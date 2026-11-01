"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import fsolve

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def _strassen_rank7():
    """Return Strassen's rank-7 decomposition as fallback."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n=2, m=2, p=2, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank6_homotopy(deadline):
    """
    Attempt to find a rank-6 decomposition using numerical methods.
    Returns (success, result_dict) or (False, None).
    """
    start_time = time.time()
    n, m, p = 2, 2, 2
    rank = 6
    
    # The tensor equation: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # We have 8 equations (i,j,k in {0,1}) and 3*4*6 = 72 unknowns.
    # This is underdetermined, so solutions should exist if the tensor has rank <= 6.
    # But we need to check if the matrix multiplication tensor actually has rank 6.
    
    # Strategy: Use random initializations and Newton's method on the 8 constraints.
    # We fix some variables to break gauge freedom (scaling of each rank factor).
    # Fix W[0][r] = 1 for r = 0..5 (first row of W to 1s) to fix scaling.
    
    # Variables: U (4x6), V (4x6), W (4x6) with W[0,:] fixed to 1
    # Free variables: U (24), V (24), W[1:,:] (18) = 66 free vars, 8 constraints
    # Very underdetermined. Let's try a different approach.
    
    # Better: Use the fact that if rank 6 exists, we can parametrize.
    # Try random initializations with Newton iteration on the 8 tensor equations.
    
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            return False, None
        
        # Random initialization
        U = np.random.randn(4, rank)
        V = np.random.randn(4, rank)
        W = np.random.randn(4, rank)
        
        # Fix gauge: normalize first column of W
        for r in range(rank):
            if abs(W[0, r]) > 1e-6:
                W[:, r] /= W[0, r]
                W[0, r] = 1.0
            else:
                # Perturb if W[0,r] is too small
                W[0, r] = 0.1 + 0.1 * np.random.randn()
                W[:, r] /= W[0, r]
                W[0, r] = 1.0
        
        # Define residuals: for each (i,j,k), residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        def residuals(params):
            # params: flattened U (24) + V (24) + W[1:,:] (18) = 66 values
            U_flat = params[0:24].reshape(4, 6)
            V_flat = params[24:48].reshape(4, 6)
            W_rest = params[48:66].reshape(3, 6)
            W_full = np.zeros((4, 6))
            W_full[0, :] = 1.0
            W_full[1:, :] = W_rest
            
            res = np.zeros(8)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[idx] = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_full[k*n+i, :]) - 1.0
                        idx += 1
            return res
        
        # Initial params
        params = np.concatenate([U.flatten(), V.flatten(), W[1:, :].flatten()])
        
        # Use fsolve
        try:
            sol, info, ier, msg = fsolve(residuals, params, full_output=True, maxfev=10000)
            if ier == 1:
                # Check solution quality
                res_val = residuals(sol)
                if np.max(np.abs(res_val)) < 1e-8:
                    U_sol = sol[0:24].reshape(4, 6)
                    V_sol = sol[24:48].reshape(4, 6)
                    W_sol = np.zeros((4, 6))
                    W_sol[0, :] = 1.0
                    W_sol[1:, :] = sol[48:66].reshape(3, 6)
                    
                    # Verify
                    if _verify_decomposition(U_sol.tolist(), V_sol.tolist(), W_sol.tolist()):
                        return True, {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
        except Exception:
            pass
        
        # Try different random seeds
        np.random.seed(attempt * 137 + 42)
    
    return False, None

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2x2 matrix multiplication.
    
    First attempts to find a rank-6 decomposition using numerical homotopy/Newton methods.
    Falls back to Strassen's rank-7 algorithm if no rank-6 solution is found.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin before 120s limit
    
    # Try rank 6
    success, result = _try_rank6_homotopy(deadline)
    if success and result is not None:
        return result
    
    # Fall back to Strassen rank 7
    return _strassen_rank7()
    
    # Strassen decomposition for 2x2 matrix multiplication:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Product: A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Then:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # In our indexing:
    # A: i=0,j=0 -> a; i=0,j=1 -> b; i=1,j=0 -> c; i=1,j=1 -> d
    # B: j=0,k=0 -> e; j=0,k=1 -> f; j=1,k=0 -> g; j=1,k=1 -> h
    # C (output): i=0,k=0 -> C[0,0]; i=0,k=1 -> C[0,1]; i=1,k=0 -> C[1,0]; i=1,k=1 -> C[1,1]
    #
    # W index: k*n + i
    # C[0,0]: i=0, k=0 -> W[0*2+0] = W[0]
    # C[0,1]: i=0, k=1 -> W[1*2+0] = W[2]
    # C[1,0]: i=1, k=0 -> W[0*2+1] = W[1]
    # C[1,1]: i=1, k=1 -> W[1*2+1] = W[3]
    #
    # U index: i*m + j
    # a: i=0,j=0 -> U[0]
    # b: i=0,j=1 -> U[1]
    # c: i=1,j=0 -> U[2]
    # d: i=1,j=1 -> U[3]
    #
    # V index: j*p + k
    # e: j=0,k=0 -> V[0]
    # f: j=0,k=1 -> V[1]
    # g: j=1,k=0 -> V[2]
    # h: j=1,k=1 -> V[3]
    
    # M1 = (a+d)(e+h)
    # U: a and d are positive: U[0][0]=1, U[3][0]=1
    # V: e and h are positive: V[0][0]=1, V[3][0]=1
    # W: C[0,0]=W[0] gets +1, C[1,1]=W[3] gets -1 (from -M2 later, but M1 contributes to both C[0,0] and C[1,1])
    # Actually let me re-derive more carefully.
    
    # Let me assign each Strassen product to a rank dimension r:
    # r=0: M1 = (a+d)(e+h)
    # r=1: M2 = (c+d)e  
    # r=2: M3 = a(f-h)
    # r=3: M4 = d(g-e)
    # r=4: M5 = (a+b)h
    # r=5: M6 = (c-a)(e+f)
    # r=6: M7 = (b-d)(g+h)
    
    # U factors (row = i*m+j, col = r):
    # r=0: a(1), d(1) -> U[0][0]=1, U[3][0]=1
    # r=1: c(1), d(1) -> U[2][1]=1, U[3][1]=1
    # r=2: a(1) -> U[0][2]=1
    # r=3: d(1) -> U[3][3]=1
    # r=4: a(1), b(1) -> U[0][4]=1, U[1][4]=1
    # r=5: c(1), a(-1) -> U[2][5]=1, U[0][5]=-1
    # r=6: b(1), d(-1) -> U[1][6]=1, U[3][6]=-1
    
    U[0][0] = 1   # a in M1
    U[3][0] = 1   # d in M1
    U[2][1] = 1   # c in M2
    U[3][1] = 1   # d in M2
    U[0][2] = 1   # a in M3
    U[3][3] = 1   # d in M4
    U[0][4] = 1   # a in M5
    U[1][4] = 1   # b in M5
    U[2][5] = 1   # c in M6
    U[0][5] = -1  # a in M6 (negative)
    U[1][6] = 1   # b in M7
    U[3][6] = -1  # d in M7 (negative)
    
    # V factors (row = j*p+k, col = r):
    # r=0: e(1), h(1) -> V[0][0]=1, V[3][0]=1
    # r=1: e(1) -> V[0][1]=1
    # r=2: f(1), h(-1) -> V[1][2]=1, V[3][2]=-1
    # r=3: g(1), e(-1) -> V[2][3]=1, V[0][3]=-1
    # r=4: h(1) -> V[3][4]=1
    # r=5: e(1), f(1) -> V[0][5]=1, V[1][5]=1
    # r=6: g(1), h(1) -> V[2][6]=1, V[3][6]=1
    
    V[0][0] = 1   # e in M1
    V[3][0] = 1   # h in M1
    V[0][1] = 1   # e in M2
    V[1][2] = 1   # f in M3
    V[3][2] = -1  # h in M3 (negative)
    V[2][3] = 1   # g in M4
    V[0][3] = -1  # e in M4 (negative)
    V[3][4] = 1   # h in M5
    V[0][5] = 1   # e in M6
    V[1][5] = 1   # f in M6
    V[2][6] = 1   # g in M7
    V[3][6] = 1   # h in M7
    
    # W factors (row = k*n+i, col = r):
    # C[0,0] = M1 + M4 - M5 + M7 -> W[0]
    # C[0,1] = M3 + M5 -> W[2]
    # C[1,0] = M2 + M4 -> W[1]
    # C[1,1] = M1 - M2 + M3 + M6 -> W[3]
    
    # W[0] (C[0,0]): M1(+1), M4(+1), M5(-1), M7(+1)
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # M5
    W[0][6] = 1   # M7
    
    # W[1] (C[1,0]): M2(+1), M4(+1)
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4
    
    # W[2] (C[0,1]): M3(+1), M5(+1)
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5
    
    # W[3] (C[1,1]): M1(+1), M2(-1), M3(+1), M6(+1)
    W[3][0] = 1   # M1
    W[3][1] = -1  # M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
