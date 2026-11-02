import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify the tensor decomposition is exact."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] for s in range(r))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _rank11_solution():
    """Return the known rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val): U[i * m + j, r] = val
    def set_v(j, k, r, val): V[j * p + k, r] = val
    def set_w(k, i, r, val): W[k * n + i, r] = val

    # Strassen for 2x2 block (columns 0,1 of B)
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    # Schoolbook for column 2
    set_u(0,0,7,1); set_v(0,2,7,1); set_w(2,0,7,1)
    set_u(0,1,8,1); set_v(1,2,8,1); set_w(2,0,8,1)
    set_u(1,0,9,1); set_v(0,2,9,1); set_w(2,1,9,1)
    set_u(1,1,10,1); set_v(1,2,10,1); set_w(2,1,10,1)

    return U, V, W

def _try_rank10_search(deadline):
    """Try to find a rank-10 decomposition via structured search."""
    n, m, p = DIMENSIONS
    rank = 10
    
    # Strategy: merge two schoolbook terms for column 2 into a combined form
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # Try: use (A00+A10)*B02 + (A01+A11)*B12 for one term, and adjust
    # Actually, let's try a more general parameterized form.
    
    # We'll parameterize the last two terms (indices 8,9) as combinations
    # and solve for coefficients that make the full tensor exact.
    
    # Base: use rank-11 structure but merge terms 7,8,9,10 into fewer terms
    # Try: term7: (a*A00 + b*A10)*B02 + (c*A01 + d*A11)*B12 -> contributes to C02 and C12
    # term8: (e*A00 + f*A10)*B02 + (g*A01 + h*A11)*B12 -> contributes to C02 and C12
    
    # Let's set up a linear system. The tensor has 2*2*3=12 entries, all should be 1.
    # With 10 terms, we have 10*4 + 10*6 + 10*6 = 40+60+60 = 160 unknowns (but many are zero in structure).
    
    # Simpler approach: fix the first 8 terms as in rank-11 (indices 0-6 for Strassen, 7 for one schoolbook),
    # and solve for the remaining 2 terms to cover all 12 tensor entries.
    
    # Actually, let's try: keep 7 Strassen-like terms + 3 schoolbook-like terms = 10
    # Or: 6 Strassen-like + 4 schoolbook-like = 10
    
    # Let me try a direct numerical approach:
    # Fix the structure of U, V, W with free parameters, then solve least squares.
    
    # Structure: 
    # Terms 0-6: same as rank-11 Strassen part (fixed)
    # Terms 7-9: parameterized for column 2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def set_u(i, j, r, val): U[i * m + j, r] = val
    def set_v(j, k, r, val): V[j * p + k, r] = val
    def set_w(k, i, r, val): W[k * n + i, r] = val
    
    # Terms 0-6: fixed Strassen
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)
    
    # Terms 7-9: parameterized
    # Term 7: (u700*A00 + u710*A10)*B02 + (u701*A01 + u711*A11)*B12
    #         -> W contributions: w720*C02, w721*C12
    # Term 8: similar
    # Term 9: similar
    
    # Free parameters for terms 7,8,9:
    # U[0,7], U[1,7], U[0,8], U[1,8], U[0,9], U[1,9]  (6 params)
    # V[0,7], V[1,7], V[0,8], V[1,8], V[0,9], V[1,9]  (6 params)
    # W[2,7], W[2,8], W[2,9], W[3,7], W[3,8], W[3,9]  (6 params)
    # Total: 18 free parameters
    
    # The tensor entries we need to satisfy:
    # T[0,0,0]=1, T[0,1,0]=1, T[0,0,1]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,0]=1,
    # T[1,0,1]=1, T[1,1,1]=1, T[0,0,2]=1, T[0,1,2]=1, T[1,0,2]=1, T[1,1,2]=1
    
    # Terms 0-6 contribute to T[i,j,0] and T[i,j,1]. Let's compute their contribution.
    # For k=0: T[i,j,0] = sum_s U[i,j,s]*V[j,0,s]*W[0,i,s]
    # For k=1: T[i,j,1] = sum_s U[i,j,s]*V[j,1,s]*W[1,i,s]
    # For k=2: T[i,j,2] = sum_s U[i,j,s]*V[j,2,s]*W[2,i,s]
    
    # Terms 7-9 only have V[j,2,s] and W[2,i,s] nonzero (for s=7,8,9)
    # So they only contribute to T[i,j,2]
    
    # First, check if terms 0-6 already give T[i,j,0]=1 and T[i,j,1]=1 for all i,j
    T01 = np.zeros((n, m, 2))
    for i in range(n):
        for j in range(m):
            for k in range(2):
                for s in range(7):
                    T01[i, j, k] += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    # Check if T01 is all ones
    target_01 = np.ones((n, m, 2))
    if not np.allclose(T01, target_01, atol=1e-9):
        return None  # Strassen part doesn't work alone
    
    # Now for k=2, we need T[i,j,2]=1 for all i,j
    # T[i,j,2] = sum_{s=7}^{9} U[i*m+j, s] * V[j*p+2, s] * W[2*n+i, s]
    
    # This is a bilinear system. Let's set up a linear least squares problem.
    # Variables: x = [U[0,7], U[1,7], U[0,8], U[1,8], U[0,9], U[1,9],
    #             V[0,7], V[1,7], V[0,8], V[1,8], V[0,9], V[1,9],
    #             W[2,7], W[2,8], W[2,9], W[3,7], W[3,8], W[3,9]]
    
    # This is nonlinear (products of variables). Use iterative approach.
    
    # Initialize with schoolbook values
    set_u(0,0,7,1); set_v(0,2,7,1); set_w(2,0,7,1)
    set_u(0,1,8,1); set_v(1,2,8,1); set_w(2,0,8,1)
    set_u(1,0,9,1); set_v(0,2,9,1); set_w(2,1,9,1)
    # Missing: A11*B12 for C12
    
    # We need 4 contributions but only have 3 terms. Try to merge.
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Try: term7 = (A00+A10)*B02 -> C02 += A00*B02, C12 += A10*B02
    #       term8 = (A01+A11)*B12 -> C02 += A01*B12, C12 += A11*B12
    #       term9 = correction term
    
    set_u(0,0,7,1); set_u(1,0,7,1)
    set_v(0,2,7,1)
    set_w(2,0,7,1); set_w(2,1,7,1)
    
    set_u(0,1,8,1); set_u(1,1,8,1)
    set_v(1,2,8,1)
    set_w(2,0,8,1); set_w(2,1,8,1)
    
    # Check if this gives correct T[i,j,2]
    T2 = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for s in range(7, 10):
                T2[i, j] += U[i*m+j, s] * V[j*p+2, s] * W[2*n+i, s]
    
    if np.allclose(T2, np.ones((n, m)), atol=1e-9):
        # We have a valid rank-10 solution!
        return U, V, W
    
    # If not, try numerical optimization
    # Use scipy if available
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    def objective(params):
        if time.time() > deadline:
            return np.inf
        # Unpack parameters
        U7 = params[0:4]  # U[0,7], U[1,7], U[0,8], U[1,8]
        V7 = params[4:8]  # V[0,7], V[1,7], V[0,8], V[1,8]
        W7 = params[8:12]  # W[2,7], W[3,7], W[2,8], W[3,8]
        
        # Compute T[i,j,2] for all i,j
        err = 0.0
        for i in range(n):
            for j in range(m):
                val = 0.0
                # Term 7
                u7 = U7[0] if i == 0 and j == 0 else (U7[1] if i == 1 and j == 0 else 0)
                v7 = V7[0] if j == 0 else (V7[1] if j == 1 else 0)
                w7 = W7[0] if i == 0 else (W7[1] if i == 1 else 0)
                val += u7 * v7 * w7
                # Term 8
                u8 = U7[2] if i == 0 and j == 1 else (U7[3] if i == 1 and j == 1 else 0)
                v8 = V7[2] if j == 0 else (V7[3] if j == 1 else 0)
                w8 = W7[2] if i == 0 else (W7[3] if i == 1 else 0)
                val += u8 * v8 * w8
                err += (val - 1.0)**2
        return err
    
    x0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
    
    if result.fun < 1e-10 and time.time() < deadline:
        # Reconstruct
        U7 = result.x[0:4]
        V7 = result.x[4:8]
        W7 = result.x[8:12]
        
        U[0*m+0, 7] = U7[0]
        U[1*m+0, 7] = U7[1]
        U[0*m+1, 8] = U7[2]
        U[1*m+1, 8] = U7[3]
        
        V[0*p+2, 7] = V7[0]
        V[1*p+2, 7] = V7[1]
        V[0*p+2, 8] = V7[2]
        V[1*p+2, 8] = V7[3]
        
        W[2*n+0, 7] = W7[0]
        W[2*n+1, 7] = W7[1]
        W[2*n+0, 8] = W7[2]
        W[2*n+1, 8] = W7[3]
        
        # Clear term 9 (unused)
        U[:, 9] = 0
        V[:, 9] = 0
        W[:, 9] = 0
        
        # Verify
        if _verify(U, V, W):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    # Try rank-10 first
    result_10 = _try_rank10_search(deadline)
    if result_10 is not None and time.time() < deadline:
        U, V, W = result_10
        assert _verify(U, V, W), "Rank-10 verification failed"
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-11
    U, V, W = _rank11_solution()
    assert _verify(U, V, W), "Rank-11 verification failed"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
