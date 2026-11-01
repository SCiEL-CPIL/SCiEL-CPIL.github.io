"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Searches for a rank-20 decomposition of M_{2,2,6} using linear algebra.
    Falls back to the rank-21 Strassen decomposition if no rank-20 is found.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, build the rank-21 Strassen decomposition as fallback
    rank21 = _build_rank21_strassen(n, m, p)
    
    # Build the matrix multiplication tensor T in flattened form
    # T[i, j, k] = 1 if output[i,k] depends on A[i,j]*B[j,k]
    # For our problem: A is 2x2, B is 2x6, output is 2x6
    # T has shape (n*m, m*p, p*n) = (4, 12, 12)
    # But we work with the bilinear form: for each output position (i,k),
    # the contribution is sum_j A[i,j] * B[j,k]
    
    # The tensor is defined by: T[a_idx, b_idx, w_idx] 
    # where a_idx = i*m + j (row i, col j of A)
    # b_idx = j*p + k (row j, col k of B)  
    # w_idx = k*n + i (row i, col k of output, trace convention)
    
    # Actually, let me think about this more carefully.
    # The matrix multiplication tensor M_{n,m,p} is a 3-tensor.
    # For A (n x m), B (m x p), C = A @ B (n x p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In the tensor decomposition, we need:
    # U: (n*m) x rank
    # V: (m*p) x rank  
    # W: (p*n) x rank
    # such that for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and all other entries are 0.
    
    # Wait, the standard formulation is:
    # The multiplication tensor has entries T_{(i,j),(j,k),(i,k)} = 1 for valid indices
    # where A has index (i,j), B has index (j,k), and output C has index (i,k).
    
    # Let me define the tensor properly.
    # A: n x m, indexed by (i, j) with i in [0,n), j in [0,m)
    # B: m x p, indexed by (j, k) with j in [0,m), k in [0,p)
    # C: n x p, indexed by (i, k) with i in [0,n), k in [0,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Flatten: A_flat[i*m + j], B_flat[j*p + k], C_flat[k*n + i] (trace convention)
    
    # The tensor T is a 3D array with T[A_flat_idx, B_flat_idx, C_flat_idx]
    # T is nonzero only when the indices are consistent:
    # A_flat_idx = i*m + j, B_flat_idx = j*p + k, C_flat_idx = k*n + i
    
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Now search for rank-20 decomposition
    # Strategy: start from rank-21, try to eliminate one term
    # by expressing it as a linear combination of others
    
    # For a rank-r decomposition, we have matrices U (nm x r), V (mp x r), W (pn x r)
    # The condition is: for all (a, b, c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a system of bilinear equations.
    
    # Approach: Fix U and V, solve for W (linear in W for fixed U,V)
    # Or use alternating minimization
    
    # Let's try: start with rank-21 solution, randomly perturb, 
    # then try to project to rank-20
    
    # Better approach: Use the fact that if we have a rank-21 decomposition,
    # we can check if any rank-20 sub-decomposition exists by looking at
    # the linear dependencies.
    
    # Let me try a direct search: parameterize a rank-20 decomposition
    # and solve the resulting system.
    
    # For rank r, we have r * (nm + mp + pn) = r * (4 + 12 + 12) = 28r variables
    # The tensor has n*m*p = 2*2*6 = 24 nonzero entries, but the full tensor
    # has (n*m)*(m*p)*(p*n) = 4*12*12 = 576 entries, most of which are 0.
    
    # The constraint is: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # This gives 576 equations (but many are trivially satisfied if T=0).
    
    # Non-trivial equations: 24 (where T=1)
    # Plus: for all (a,b,c) where T=0, the sum must be 0.
    # Number of zero entries: 576 - 24 = 552
    
    # Total: 576 equations, 28r unknowns.
    # For r=20: 560 unknowns, 576 equations. Overdetermined but close.
    # For r=21: 588 unknowns, 576 equations. Underdetermined.
    
    # Let me try to solve for rank-20 using least squares + verification.
    
    def try_rank_r(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using iterative optimization."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        best_residual = np.inf
        best_solution = None
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Initialize randomly
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            
            # Normalize to avoid scale issues
            for col in range(r):
                norm = np.sqrt(np.sum(U[:, col]**2) + np.sum(V[:, col]**2) + np.sum(W[:, col]**2))
                if norm > 1e-10:
                    U[:, col] /= norm
                    V[:, col] /= norm
                    W[:, col] /= norm
            
            # Alternating least squares
            for iteration in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Solve for U given V, W
                # For each a: U[a, :] should satisfy sum_r U[a,r] * (V[:,r] * W[:,r])_b = T[a,b,c] for all b,c
                # This is: U[a, :] @ (V*W)^T should match T[a, :, :] reshaped
                # Actually: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # For fixed a: sum_r U[a,r] * (V[:,r] * W[:,r]) = T[a, :, :] (reshaped to vector of length mp*pn)
                
                # Let's build the linear system for U[a, :]
                # Variables: U[a, 0], ..., U[a, r-1]
                # Equations: for each (b, c): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                
                # This is a linear system: M_a @ U[a, :] = t_a
                # where M_a has shape (mp*pn, r) and t_a has shape (mp*pn,)
                
                # But this is over-constrained. Use least squares.
                
                for a in range(n * m):
                    # Build M_a
                    M_a = np.zeros((m * p * p * n, r))
                    t_a = np.zeros(m * p * p * n)
                    idx = 0
                    for b in range(m * p):
                        for c in range(p * n):
                            for r_idx in range(r):
                                M_a[idx, r_idx] = V[b, r_idx] * W[c, r_idx]
                            t_a[idx] = T[a, b, c]
                            idx += 1
                    
                    # Solve least squares
                    sol, residuals, rank_a, sv = np.linalg.lstsq(M_a, t_a, rcond=None)
                    U[a, :] = sol
                
                # Solve for V given U, W
                for b in range(m * p):
                    M_b = np.zeros((n * m * p * n, r))
                    t_b = np.zeros(n * m * p * n)
                    idx = 0
                    for a in range(n * m):
                        for c in range(p * n):
                            for r_idx in range(r):
                                M_b[idx, r_idx] = U[a, r_idx] * W[c, r_idx]
                            t_b[idx] = T[a, b, c]
                            idx += 1
                    
                    sol, residuals, rank_b, sv = np.linalg.lstsq(M_b, t_b, rcond=None)
                    V[b, :] = sol
                
                # Solve for W given U, V
                for c in range(p * n):
                    M_c = np.zeros((n * m * m * p, r))
                    t_c = np.zeros(n * m * m * p)
                    idx = 0
                    for a in range(n * m):
                        for b in range(m * p):
                            for r_idx in range(r):
                                M_c[idx, r_idx] = U[a, r_idx] * V[b, r_idx]
                            t_c[idx] = T[a, b, c]
                            idx += 1
                    
                    sol, residuals, rank_c, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
                    W[c, :] = sol
                
                # Check residual
                residual = 0.0
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            residual += (val - T[a, b, c])**2
                
                if residual < best_residual:
                    best_residual = residual
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if residual < 1e-12:
                    break
        
        if best_solution is not None and best_residual < 1e-8:
            return best_solution
        return None
    
    # Try rank 20
    result20 = try_rank_r(20)
    if result20 is not None:
        U, V, W = result20
        # Convert to integer/half-integer if possible
        U_int = _to_rational(U)
        V_int = _to_rational(V)
        W_int = _to_rational(W)
        if U_int is not None and V_int is not None and W_int is not None:
            return {"U": U_int, "V": V_int, "W": W_int}
    
    # Fallback to rank 21
    return rank21

def _build_rank21_strassen(n, m, p):
    """Build the rank-21 Strassen decomposition."""
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 0 * p + 2 * b
                elif l == 1:
                    v_idx = 0 * p + 2 * b + 1
                elif l == 2:
                    v_idx = 1 * p + 2 * b
                elif l == 3:
                    v_idx = 1 * p + 2 * b + 1
                V[v_idx][r] = val
            
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            
            r += 1
    
    return {"U": U, "V": V, "W": W}

def _to_rational(matrix, tol=1e-6):
    """Try to convert a matrix to integer/half-integer values."""
    if matrix is None:
        return None
    
    result = []
    for row in matrix:
        new_row = []
        for val in row:
            # Try to round to nearest half-integer
            rounded = round(val * 2) / 2
            if abs(val - rounded) < tol:
                new_row.append(rounded)
            else:
                return None  # Not close to half-integer
        result.append(new_row)
    return result
# EVOLVE-BLOCK-END
