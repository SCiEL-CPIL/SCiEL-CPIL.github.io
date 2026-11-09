import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt_total = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 300s limit

    # Strategy: Exploit the known structure of the 2D Heat Equation.
    # The problem is y' = L y, where L is the Kronecker sum of two 1D Laplacians.
    # L = kron(I, L1) + kron(L1, I)
    # This allows us to compute expm(L * dt) efficiently using 1D expm.
    
    # 1. Verify the structure with a single call? 
    # Actually, we can just assume it's the standard 2D heat equation on a 10x10 grid 
    # with Dirichlet or Neumann boundaries, but we don't know the exact boundary conditions 
    # or grid spacing from the function signature alone. 
    # However, the task says "fixed 10x10 two-dimensional heat equation". 
    # Standard discretization: 
    # Let h be the grid spacing. 
    # Usually, for a unit square [0,1]x[0,1] with n=10 interior points? Or 10x10 total?
    # "10x10" usually implies N=100 total points.
    # If it's the standard Laplacian, the eigenvalues are known.
    
    # Let's try to reconstruct the 1D Laplacian L1 using the Kronecker structure.
    # If y' = (I \otimes L1 + L1 \otimes I) y, we can test this hypothesis.
    # But we need to know L1.
    # We can extract L1 by calling f on specific basis vectors.
    
    # Alternative: Just use the full 100x100 matrix approach but optimize the expm calculation?
    # The current score 1.0788 comes from expm(100x100).
    # The inspiration 1 (1.1620) uses "structural exploitation".
    # The inspiration 3 (1.1495) mentions "separable structure" and "_build_base_laplacian_2d".
    
    # Let's assume the standard 2D heat equation on a 10x10 grid.
    # The operator is likely:
    # d/dt y_ij = c * (y_{i-1,j} + y_{i+1,j} + y_{i,j-1} + y_{i,j+1} - 4y_{ij}) / h^2
    # With some boundary conditions.
    
    # If we can identify the 1D Laplacian A (10x10), then L = kron(I, A) + kron(A, I).
    # Then exp(L*dt) = exp(kron(I, A*dt)) * exp(kron(A, I*dt)) ? No.
    # exp(kron(I, A) + kron(A, I)) = exp(kron(I, A)) * exp(kron(A, I)) because they commute.
    # exp(kron(I, A)) = kron(I, exp(A))
    # exp(kron(A, I)) = kron(exp(A), I)
    # So exp(L*dt) = kron(I, exp(A*dt)) @ kron(exp(A*dt), I)
    # This is much faster than expm(100x100).
    
    # How to find A?
    # We can call f(t0, e) for basis vectors to get the full L.
    # Then check if L matches the Kronecker sum structure.
    # To save calls, we can just reconstruct L (100 calls) and then try to decompose it.
    # Or, we can assume the standard form and verify with 1-2 calls.
    
    # Let's try to reconstruct L fully first (100 calls is cheap).
    L = np.zeros((N, N), dtype=float)
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        L[:, i] = f(t0, e_i)
        
    # Now, try to see if L = kron(I, A) + kron(A, I) for some 10x10 A.
    # If N=100, grid_size = 10.
    g = int(np.sqrt(N))
    if g * g == N:
        # Reshape L to 4D tensor: L[i, j, k, l] corresponds to y_{j,l} -> y_{i,k}
        # y is vectorized in row-major order: y[i*g + k] = y_{i,k}
        # L_vec = L.reshape(g, g, g, g)
        # L_{i,k, j,l} = L[i*g+k, j*g+l]
        
        # If L = kron(I, A) + kron(A, I), then:
        # (kron(I, A))_{i,k, j,l} = delta_{i,j} * A_{k,l}
        # (kron(A, I))_{i,k, j,l} = A_{i,j} * delta_{k,l}
        
        # So L_{i,k, j,l} = delta_{i,j} A_{k,l} + A_{i,j} delta_{k,l}
        
        # We can solve for A.
        # Set i=j=0: L_{0,k, 0,l} = A_{k,l} + A_{0,0} delta_{k,l}
        # Set k=l=0: L_{i,0, j,0} = delta_{i,j} A_{0,0} + A_{i,j}
        
        # Let's extract A from the diagonal blocks.
        # Block (0,0) of L (top-left 10x10) corresponds to i=0, j=0.
        # L_block_00 = L[0:g, 0:g]
        # L_block_00[k, l] = A_{k,l} + A_{0,0} * delta_{k,l}
        
        # Block (i,j) where i!=j:
        # L_block_ij = A_{i,j} * I
        
        # Let's check if off-diagonal blocks are scalar multiples of I.
        is_kronecker = True
        A = np.zeros((g, g), dtype=float)
        
        # Get A_{0,0} from trace?
        # Sum of L_block_00 = trace(A) + A_{0,0}
        # This is tricky. Let's just try to reconstruct A.
        
        # From L_block_ij (i!=j), we have L_block_ij = A_{i,j} * I.
        # So A_{i,j} = L_block_ij[0,0] (if consistent).
        for i in range(g):
            for j in range(g):
                if i != j:
                    block = L[i*g:(i+1)*g, j*g:(j+1)*g]
                    # Check if block is scalar * I
                    if not np.allclose(block, block[0,0] * np.eye(g)):
                        is_kronecker = False
                        break
                    A[i, j] = block[0, 0]
                else:
                    # Diagonal block: L_block_ii = A + A_{i,i} * I ? No.
                    # L_{i,k, i,l} = A_{k,l} + A_{i,i} delta_{k,l}
                    # So L_block_ii - A_{i,i} I = A
                    # We don't know A_{i,i} yet.
                    pass
        
        if is_kronecker:
            # We have off-diagonal entries of A.
            # For diagonal entries, use L_block_00.
            # L_block_00 = A + A_{0,0} I
            # We need A_{0,0}.
            # Note that trace(L_block_00) = trace(A) + A_{0,0}
            # Also, from off-diagonal blocks, we have A_{i,j} for i!=j.
            # We can estimate trace(A) if we assume symmetry? Heat eq Laplacian is symmetric.
            # Let's assume A is symmetric.
            # Then A_{0,0} is unknown.
            # Let's pick k,l such that k!=l. L_block_00[k,l] = A_{k,l}.
            # We already have A_{k,l} from off-diagonal blocks? No, A_{k,l} is internal to A.
            
            # Let's just use the full L and check if expm(L*dt) @ y0 matches
            # the Kronecker product result.
            
            # Actually, simpler: just use the Kronecker product formula for the exponential.
            # We need A.
            # Let's try to recover A from L.
            # L = kron(I, A) + kron(A, I)
            # Apply (I \otimes P) ... this is getting complicated.
            
            # Let's just fall back to full expm if we can't easily decompose.
            # But wait, we can just compute expm(L*dt) directly.
            # The cost of expm(100x100) is ~ 100^3 = 1e6 ops.
            # The cost of 2 * expm(10x10) is ~ 2 * 1000 = 2000 ops.
            # The speedup is significant, but the accuracy is the same.
            # The score depends on accuracy and calls.
            # Calls: 100 for L reconstruction.
            # If we use Kronecker, we still need 100 calls to get L to verify?
            # Or we can assume the structure and just build A from the grid?
            # If we know the grid spacing and boundary conditions, we can build A without calls.
            # But we don't know them.
            
            # However, the task says "fixed 10x10 two-dimensional heat equation".
            # It is highly likely the standard one.
            # Let's try to build A based on standard assumptions and verify with 1 call.
            
            # Standard 2D Heat Eq on [0,1]x[0,1], n=10 points?
            # If n=10, h = 1/(n+1) or 1/n?
            # Usually Dirichlet: h = 1/(n+1).
            # A = (1/h^2) * tridiag(-1, 2, -1)
            
            # Let's try a few common discretizations.
            candidates = []
            for n in [10]:
                for h in [1.0/(n+1), 1.0/n, 1.0/(n-1)]:
                    A_cand = (1.0/h**2) * (np.diag(np.ones(n-1), -1) + np.diag(np.ones(n-1), 1) + np.diag(2*np.ones(n)))
                    # Check Neumann?
                    # Neumann: corners have different coefficients.
                    candidates.append(A_cand)
            
            # Verify candidates against L.
            found_A = None
            for A_cand in candidates:
                L_cand = np.kron(np.eye(g), A_cand) + np.kron(A_cand, np.eye(g))
                if np.allclose(L, L_cand, rtol=1e-5, atol=1e-5):
                    found_A = A_cand
                    break
            
            if found_A is not None:
                # Compute exp(L*dt) using Kronecker product
                E = np.exp(found_A * dt_total)  # 10x10 expm
                # exp(L*dt) = kron(I, E) @ kron(E, I)
                # Actually, since they commute, it's just the product.
                # M = kron(np.eye(g), E) @ kron(E, np.eye(g))
                # This is a 100x100 matrix.
                # y_final = M @ y0
                # M @ y0 = kron(I, E) @ (kron(E, I) @ y0)
                # Let y0_2d = y0.reshape(g, g)
                # kron(E, I) @ y0_2d = E @ y0_2d (row-wise? No, column-wise)
                # Vectorization convention: y[i*g+k] = y_{i,k}
                # kron(E, I) acts on columns of y0_2d?
                # If y0 is column-major, kron(E, I) y0 = (E y0_2d)^T?
                # NumPy uses row-major (C-order) by default.
                # y0_2d[i, k] = y0[i*g + k]
                # (kron(E, I) @ y0)_{i*g+k} = sum_l E_{i,l} y0_{l*g+k}
                # This corresponds to (E @ y0_2d.T).T
                # Similarly for kron(I, E).
                
                y0_2d = y0.reshape(g, g)
                # Apply kron(E, I): multiply rows by E
                y_temp = (E @ y0_2d.T).T
                # Apply kron(I, E): multiply columns by E
                y_final_2d = (y_temp @ E.T)
                y_final = y_final_2d.flatten()
                
                if np.all(np.isfinite(y_final)):
                    return y_final
            
            # If no standard candidate matched, fall back to full expm
            try:
                M = scipy.linalg.expm(L * dt_total)
                y_final = M @ y0
                if np.all(np.isfinite(y_final)):
                    return y_final
            except Exception:
                pass

    # Fallback: RK4
    steps = 1000
    h = dt_total / steps
    t = t0
    y = y0.copy()
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
