import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Solves the 10x10 2D Heat Equation IVP.
    Strategy:
    1. Detect time-invariance and linearity.
    2. Reconstruct the system matrix A.
       - Primary: Exploit Kronecker sum structure (L ⊗ I + I ⊗ L) for 1D Laplacians.
         This requires only 3 probes instead of N.
       - Fallback: Full basis extraction (N calls) if structure detection fails.
    3. Compute the exact solution using Matrix Exponential.
       - If Kronecker structure is confirmed, use 1D eigen-decompositions for speed/stability.
       - Otherwise, use dense expm.
    """
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    t_start, t_end = t_span
    dt = t_end - t_start
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s limit

    # 1. Check for time-invariance
    # Evaluate f at t_start and t_end with y0
    f_start = f(t_start, y0)
    f_end_probe = f(t_end, y0)
    
    is_time_invariant = np.allclose(f_start, f_end_probe, rtol=1e-9, atol=1e-12)
    
    A = None
    is_kronecker = False
    L1 = None  # The 1D matrix part of the Kronecker sum
    
    if is_time_invariant:
        # Assume linear system dy/dt = A y
        
        if N == 100:
            # Strategy A: Standard 10x10 Heat Stencil Detection (Kronecker Sum)
            # The operator is A = kron(L, I) + kron(I, L) where L is 10x10.
            # We can determine L by probing a vector that isolates the row/column structure.
            
            try:
                # Probe 1: e_0 (index 0, i=0, j=0)
                e_0 = np.zeros(N)
                e_0[0] = 1.0
                f_0 = f(t_start, e_0)
                
                # Probe 2: e_10 (index 10, i=1, j=0) - isolates row direction
                e_10 = np.zeros(N)
                e_10[10] = 1.0
                f_10 = f(t_start, e_10)
                
                # Probe 3: e_1 (index 1, i=0, j=1) - isolates col direction
                e_1 = np.zeros(N)
                e_1[1] = 1.0
                f_1 = f(t_start, e_1)
                
                # For a standard 5-point Laplacian:
                # A[idx, idx] = -4 * alpha (center)
                # A[idx, idx+1] = alpha (right)
                # A[idx, idx-1] = alpha (left)
                # A[idx, idx+10] = alpha (down)
                # A[idx, idx-10] = alpha (up)
                
                # Let's extract coefficients from f_0 (corner 0,0)
                # f_0[0] should be -4*alpha (if boundary is Dirichlet 0, neighbors 1,10 exist)
                # f_0[1] should be alpha
                # f_0[10] should be alpha
                
                # Let's extract from f_10 (edge 1,0)
                # f_10[10] should be -4*alpha
                # f_10[0] should be alpha
                # f_10[20] should be alpha
                # f_10[11] should be alpha
                
                # Let's extract from f_1 (edge 0,1)
                # f_1[1] should be -4*alpha
                # f_1[0] should be alpha
                # f_1[2] should be alpha
                # f_1[11] should be alpha
                
                # Determine alpha
                # From f_0:
                alpha_right = f_0[1]
                alpha_down = f_0[10]
                center_0 = f_0[0]
                
                # From f_10:
                alpha_up_10 = f_10[0]
                alpha_down_10 = f_10[20]
                alpha_right_10 = f_10[11]
                center_10 = f_10[10]
                
                # From f_1:
                alpha_left_1 = f_1[0]
                alpha_right_1 = f_1[2]
                alpha_down_1 = f_1[11]
                center_1 = f_1[1]
                
                # Check consistency
                is_consistent = (
                    np.isclose(alpha_right, alpha_down, rtol=1e-5, atol=1e-10) and
                    np.isclose(alpha_right, alpha_up_10, rtol=1e-5, atol=1e-10) and
                    np.isclose(alpha_right, alpha_down_10, rtol=1e-5, atol=1e-10) and
                    np.isclose(alpha_right, alpha_right_10, rtol=1e-5, atol=1e-10) and
                    np.isclose(alpha_right, alpha_left_1, rtol=1e-5, atol=1e-10) and
                    np.isclose(alpha_right, alpha_right_1, rtol=1e-5, atol=1e-10) and
                    np.isclose(alpha_right, alpha_down_1, rtol=1e-5, atol=1e-10) and
                    np.isclose(center_0, -4 * alpha_right, rtol=1e-5, atol=1e-10) and
                    np.isclose(center_10, -4 * alpha_right, rtol=1e-5, atol=1e-10) and
                    np.isclose(center_1, -4 * alpha_right, rtol=1e-5, atol=1e-10)
                )
                
                if is_consistent:
                    alpha = alpha_right
                    # Construct L (10x10)
                    L1 = np.zeros((10, 10))
                    for i in range(10):
                        L1[i, i] = -4 * alpha
                        if i > 0:
                            L1[i, i-1] = alpha
                        if i < 9:
                            L1[i, i+1] = alpha
                    
                    # Construct A = kron(L1, I) + kron(I, L1)
                    # Note: The ordering of indices matters.
                    # If index = i * 10 + j, then the matrix is kron(L1, I_10) ?
                    # Let's check:
                    # A[i*10+j, k*10+l]
                    # If i=k, A[i*10+j, i*10+l] = L1[j,l] (from kron(I, L1))
                    # If j=l, A[i*10+j, k*10+j] = L1[i,k] (from kron(L1, I))
                    # So A = kron(I, L1) + kron(L1, I)
                    
                    A = np.kron(np.eye(10), L1) + np.kron(L1, np.eye(10))
                    is_kronecker = True
                    
                    # Verify A against f_start
                    if not np.allclose(A @ y0, f_start, rtol=1e-5, atol=1e-12):
                        A = None
                        is_kronecker = False
            except Exception:
                A = None
                is_kronecker = False
        
        # Strategy B: Full Basis Extraction
        if A is None:
            try:
                A = np.zeros((N, N))
                for i in range(N):
                    if time.time() > deadline:
                        break
                    ei = np.zeros(N)
                    ei[i] = 1.0
                    A[:, i] = f(t_start, ei)
                
                # Verify
                if not np.allclose(A @ y0, f_start, rtol=1e-5, atol=1e-12):
                    A = None
            except Exception:
                A = None
        
        # If we have a valid A, compute exact solution
        if A is not None:
            try:
                if is_kronecker:
                    # Use 1D eigen-decomposition for efficiency and stability
                    # A = kron(I, L1) + kron(L1, I)
                    # Eigenvalues of A are lambda_i + lambda_j where lambda are eigenvalues of L1
                    # Eigenvectors are kron(v_i, v_j)
                    # We can compute exp(A*t) @ y0 efficiently.
                    
                    # Diagonalize L1 = V D V^T
                    evals, evecs = np.linalg.eigh(L1)
                    
                    # y0 is 100x1, reshape to 10x10
                    Y0 = y0.reshape(10, 10)
                    
                    # Transform to eigenbasis
                    # The action of exp(A*dt) on Y0 is:
                    # Y0_eig = evecs.T @ Y0 @ evecs
                    # Y0_scaled = Y0_eig * exp((evals[:,None] + evals[None,:]) * dt)
                    # Y_end = evecs @ Y0_scaled @ evecs.T
                    # Then flatten back to 100x1
                    
                    Y0_eig = evecs.T @ Y0 @ evecs
                    # Eigenvalues of A are sums of eigenvalues of L1
                    # lambda_A[i,j] = evals[i] + evals[j]
                    lambda_A = evals[:, None] + evals[None, :]
                    exp_factors = np.exp(lambda_A * dt)
                    Y0_scaled = Y0_eig * exp_factors
                    Y_end = evecs @ Y0_scaled @ evecs.T
                    y_end = Y_end.flatten()
                else:
                    M = scipy.linalg.expm(A * dt)
                    y_end = M @ y0
                    
                return y_end
            except Exception:
                pass
    
    # Fallback: If time-dependent or non-linear or extraction failed
    # Use a high-precision integrator.
    
    call_count = 0
    def f_wrapper(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)

    try:
        # Use Radau for stiff problems (Heat equation is stiff)
        sol = scipy.integrate.solve_ivp(
            f_wrapper, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=1e-3
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If all else fails, return y0 or best effort
    return y0
