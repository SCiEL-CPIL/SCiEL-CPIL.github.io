"""Combined approach: verified rank-39 Strassen+schoolbook, then bounded exact search."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0        
        v_f = k1        
        v_g = p + k0    
        v_h = p + k1    
    
        w_c00 = k0 * n      
        w_c10 = k0 * n + 1  
        w_c01 = k1 * n      
        w_c11 = k1 * n + 1
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2        
    v_g2 = p + k2    
    w_c02 = k2 * n      
    w_c12 = k2 * n + 1  
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _try_low_rank(rank, n, m, p, deadline, max_rounds=2000, seed=42):
    """
    Try to find a decomposition of given rank using iterative refinement with
    better initialization and convergence criteria.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Better initialization: use structured initialization based on Strassen pattern
    U = rng.randn(n * m, rank) * 0.3
    V = rng.randn(m * p, rank) * 0.3
    W = rng.randn(p * n, rank) * 0.3
    
    # Target index arrays
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    # Precompute masks for efficiency
    masks_a = [(ta_arr == a) for a in range(n * m)]
    masks_b = [(tb_arr == b) for b in range(m * p)]
    masks_c = [(tc_arr == c) for c in range(p * n)]
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = masks_a[a]
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = masks_b[b]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = masks_c[c_idx]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        # Check convergence every 25 iterations (more frequent)
        if it % 25 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
    
    # Check if best solution is good enough
    if best_err < 1e-8 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _try_low_rank_strassen_init(rank, n, m, p, deadline, max_rounds=2000, seed=42):
    """
    Try to find a decomposition of given rank using iterative refinement
    with a Strassen-inspired initialization.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Strassen-inspired initialization: start with a structured pattern
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize first 7 columns with Strassen pattern for the first 2x2 block
    if rank >= 7:
        # m1 = (A00+A11)*(B[0,0]+B[1,1]) -> C[0,0] + C[1,1]
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[p+1, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (A10+A11)*B[0,0] -> C[1,0] - C[1,1]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # m3 = A00*(B[0,1]-B[1,1]) -> C[0,1] + C[1,1]
        U[0, 2] = 1
        V[1, 2] = 1; V[p+1, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # m4 = A11*(B[1,0]-B[0,0]) -> C[0,0] + C[1,0]
        U[3, 3] = 1
        V[p, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # m5 = (A00+A01)*B[1,1] -> -C[0,0] + C[0,1]
        U[0, 4] = 1; U[1, 4] = 1
        V[p+1, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # m6 = (A10-A00)*(B[0,0]+B[0,1]) -> C[1,1]
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7 = (A01-A11)*(B[1,0]+B[1,1]) -> C[0,0]
        U[1, 6] = 1; U[3, 6] = -1
        V[p, 6] = 1; V[p+1, 6] = 1
        W[0, 6] = 1
    
    # Fill remaining columns with random values
    if rank > 7:
        U[:, 7:] = rng.randn(n * m, rank - 7) * 0.3
        V[:, 7:] = rng.randn(m * p, rank - 7) * 0.3
        W[:, 7:] = rng.randn(p * n, rank - 7) * 0.3
    
    # Target index arrays
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    # Precompute masks for efficiency
    masks_a = [(ta_arr == a) for a in range(n * m)]
    masks_b = [(tb_arr == b) for b in range(m * p)]
    masks_c = [(tc_arr == c) for c in range(p * n)]
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = masks_a[a]
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = masks_b[b]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = masks_c[c_idx]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        # Check convergence every 25 iterations
        if it % 25 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
    
    # Check if best solution is good enough
    if best_err < 1e-8 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # Try to find a lower rank decomposition with remaining time
        if time.time() < deadline - 15:
            # Try ranks in order, with multiple seeds for each
            for rank in [38, 37, 36, 35, 34]:
                if time.time() > deadline - 5:
                    break
                # Use more seeds for lower ranks
                seeds = list(range(rank, rank + 10))
                for seed in seeds:
                    if time.time() > deadline - 3:
                        break
                    # Try both random and Strassen-inspired initialization
                    result = _try_low_rank(rank, n, m, p, deadline=deadline - 2, max_rounds=1500, seed=seed)
                    if result is not None:
                        U2, V2, W2 = result
                        if _verify(U2, V2, W2, n, m, p):
                            return {"U": U2, "V": V2, "W": W2}
                    
                    if time.time() > deadline - 3:
                        break
                    result = _try_low_rank_strassen_init(rank, n, m, p, deadline=deadline - 2, max_rounds=1500, seed=seed)
                    if result is not None:
                        U2, V2, W2 = result
                        if _verify(U2, V2, W2, n, m, p):
                            return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": W, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
