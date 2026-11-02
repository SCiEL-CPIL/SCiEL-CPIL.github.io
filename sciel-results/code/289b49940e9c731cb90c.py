"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_tensor_identity(U, V, W, n, m, p):
    """Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k)."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9, max_err

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return U, V, W

def _try_rank_R(R, n, m, p, time_budget=30.0):
    """Try to find a rank-R decomposition using random initialization and gradient-free search.
    Returns (U, V, W) or None."""
    start = time.time()
    
    # Number of equations: n*m*p
    num_eq = n * m * p
    # Number of unknowns per rank: n*m + m*p + p*n
    dims = [n*m, m*p, p*n]
    total_unknowns = sum(dims)
    
    # Build the linear system structure: for each (i,j,k), 
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of n*m*p equations in R*(n*m + m*p + p*n) unknowns.
    
    # Strategy: fix W to a random matrix, solve for U and V iteratively.
    # Or use the approach: for fixed V and W, U is determined by a linear system.
    
    # Let's use a simpler approach: random init + iterative refinement
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for attempt in range(20):
        if time.time() - start > time_budget:
            break
        
        np.random.seed(attempt * 1000 + 42)
        
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        # Iterative refinement: alternate fixing two factors and solving for the third
        for it in range(50):
            if time.time() - start > time_budget:
                break
            
            # Fix V, W; solve for U
            # For each (i,j): sum_r U[i*m+j, r] * (V[j*p+*, r] * W[*n+i, r]) = 1 for all k
            # This is: for each (i,j), we have p equations in R unknowns for U[i*m+j, :]
            # A_{(i,j), k, r} = V[j*p+k, r] * W[k*n+i, r]
            # sum_r U[i*m+j, r] * A_{(i,j), k, r} = 1 for all k
            
            for i in range(n):
                for j in range(m):
                    # Build p x R matrix
                    A_mat = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Solve A_mat @ U[i*m+j, :] = ones(p)
                    if R >= p:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        U[i*m+j, :] = sol
                    else:
                        # Underdetermined: use lstsq for minimum norm
                        sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        U[i*m+j, :] = sol
            
            # Fix U, W; solve for V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                    V[j*p+k, :] = sol
            
            # Fix U, V; solve for W
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                    W[k*n+i, :] = sol
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = max(err, abs(s - 1.0))
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10:
        # Convert to lists with integer/half-integer entries if possible
        U_list = [[0]*R for _ in range(n*m)]
        V_list = [[0]*R for _ in range(m*p)]
        W_list = [[0]*R for _ in range(p*n)]
        for i in range(n*m):
            for r in range(R):
                U_list[i][r] = round(best_U[i, r] * 2) / 2
        for i in range(m*p):
            for r in range(R):
                V_list[i][r] = round(best_V[i, r] * 2) / 2
        for i in range(p*n):
            for r in range(R):
                W_list[i][r] = round(best_W[i, r] * 2) / 2
        return U_list, V_list, W_list
    
    return None

def _build_rank33_decomposition():
    """Build a rank-33 decomposition for 2x3 @ 3x6 matrix multiplication.
    Uses 3 independent blocks of 2x3 @ 3x2, each using rank 11."""
    n, m, p = 2, 3, 6
    
    # For 2x3 @ 3x2 sub-problem:
    # i in {0,1}, j in {0,1,2}, k in {0,1}
    # Need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Schoolbook rank 12. Let's try to find rank 11.
    # If rank 11 doesn't work, fall back to rank 12 per block = 36 total.
    
    # Let me try to construct rank 11 for 2x3 @ 3x2.
    # The tensor has 12 entries all equal to 1.
    # With rank 11, we have 11*(6+6+4) = 154 unknowns and 12 equations.
    # This is highly overdetermined in the nonlinear sense but let's try.
    
    # Actually, let me just use the known terms from the current program and verify them.
    # The current program's terms use local indices:
    # u: indexed by (i,j) where i in {0,1}, j in {0,1,2} -> flat: i*3+j
    # v: indexed by (j,k) where j in {0,1,2}, k in {0,1} -> flat: j*2+k  
    # w: indexed by (k,i) where k in {0,1}, i in {0,1} -> flat: k*2+i
    
    # Let me check the current terms against the identity:
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    # Verify: for each (i,j,k) in {0,1}x{0,1,2}x{0,1}, check sum = 1
    # u flat index: i*3+j
    # v flat index: j*2+k
    # w flat index: k*2+i
    
    ok = True
    for i in range(2):
        for j in range(3):
            for k in range(2):
                s = 0.0
                for term in terms:
                    u_val = term['u'].get(i*3+j, 0)
                    v_val = term['v'].get(j*2+k, 0)
                    w_val = term['w'].get(k*2+i, 0)
                    s += u_val * v_val * w_val
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if ok:
        # Rank 11 per block works! Build the full rank-33 decomposition.
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(3):
            for t_idx, term in enumerate(terms):
                col = block * 11 + t_idx
                for idx, val in term['u'].items():
                    U[idx][col] = val
                for l_idx, val in term['v'].items():
                    j_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = j_local * p + (block * 2 + k_local)
                    V[g_idx][col] = val
                for l_idx, val in term['w'].items():
                    i_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = (block * 2 + k_local) * n + i_local
                    W[g_idx][col] = val
        
        return U, V, W
    else:
        # Fall back to schoolbook rank 36
        return _build_schoolbook(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # First try the rank-33 decomposition
    U, V, W = _build_rank33_decomposition()
    rank = len(U[0])
    
    # Verify the decomposition
    valid, max_err = _verify_tensor_identity(U, V, W, n, m, p)
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # If rank 33 doesn't work, try to find a lower rank decomposition
    # using the iterative method
    for R in range(35, 32, -1):
        if time.time() > deadline - 10:
            break
        result = _try_rank_R(R, n, m, p, time_budget=20.0)
        if result is not None:
            U, V, W = result
            valid, max_err = _verify_tensor_identity(U, V, W, n, m, p)
            if valid:
                return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook rank 36
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
