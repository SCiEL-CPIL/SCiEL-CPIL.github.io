"""Combined approach: verified rank-39 baseline, then iterative refinement for lower rank with full constraint verification."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact for ALL entries."""
    rank = len(U[0])
    # Precompute target tensor
    # T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(k',i') with j=j', k=k', i=i'
    # a = i*m + j
    # b = j'*p + k
    # c = k'*n + i'
    
    # We can iterate over all a, b, c and check
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for jp in range(m * p):
                # b = jp
                # j' = jp // p
                # k = jp % p
                j_prime = jp // p
                k = jp % p
                if j != j_prime:
                    # Must be 0
                    s = 0.0
                    for c_idx in range(p * n):
                        # c = k'*n + i'
                        # k' = c_idx // n
                        # i' = c_idx % n
                        k_prime = c_idx // n
                        i_prime = c_idx % n
                        if k != k_prime or i != i_prime:
                            # This is a zero entry
                            for r in range(rank):
                                s += U[a][r] * V[jp][r] * W[c_idx][r]
                            if abs(s) > tol:
                                return False
                else:
                    # j == j'
                    for c_idx in range(p * n):
                        k_prime = c_idx // n
                        i_prime = c_idx % n
                        if k == k_prime and i == i_prime:
                            # This is a 1 entry
                            s = 0.0
                            for r in range(rank):
                                s += U[a][r] * V[jp][r] * W[c_idx][r]
                            if abs(s - 1.0) > tol:
                                return False
                        else:
                            # This is a 0 entry
                            s = 0.0
                            for r in range(rank):
                                s += U[a][r] * V[jp][r] * W[c_idx][r]
                            if abs(s) > tol:
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

def _try_low_rank(rank, n, m, p, deadline, max_rounds=2000, seed=42, tol=1e-8):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Minimizes error over ALL tensor entries.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Precompute indices for all tensor entries
    # a in 0..3, b in 0..21, c in 0..21
    # Target T[a,b,c]
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T=1 if j=j', k=k', i=i'
    
    a_vals, b_vals, c_vals, targets = [], [], [], []
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for jp in range(m * p):
                j_prime = jp // p
                k = jp % p
                for cp in range(p * n):
                    k_prime = cp // n
                    i_prime = cp % n
                    a_vals.append(a)
                    b_vals.append(jp)
                    c_vals.append(cp)
                    if j == j_prime and k == k_prime and i == i_prime:
                        targets.append(1.0)
                    else:
                        targets.append(0.0)
                        
    a_arr = np.array(a_vals)
    b_arr = np.array(b_vals)
    c_arr = np.array(c_vals)
    t_arr = np.array(targets)
    
    best_err = float('inf')
    best_state = None
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = (a_arr == a)
            if not np.any(mask):
                continue
            b_sub = b_arr[mask]
            c_sub = c_arr[mask]
            t_sub = t_arr[mask]
            V_sub = V[b_sub, :]
            W_sub = W[c_sub, :]
            prod = V_sub * W_sub
            num = (prod * t_sub[:, None]).sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = (b_arr == b)
            if not np.any(mask):
                continue
            a_sub = a_arr[mask]
            c_sub = c_arr[mask]
            t_sub = t_arr[mask]
            U_sub = U[a_sub, :]
            W_sub = W[c_sub, :]
            prod = U_sub * W_sub
            num = (prod * t_sub[:, None]).sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = (c_arr == c_idx)
            if not np.any(mask):
                continue
            a_sub = a_arr[mask]
            b_sub = b_arr[mask]
            t_sub = t_arr[mask]
            U_sub = U[a_sub, :]
            V_sub = V[b_sub, :]
            prod = U_sub * V_sub
            num = (prod * t_sub[:, None]).sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        # Check convergence every 10 iterations
        if it % 10 == 0:
            sums = np.zeros(len(a_arr))
            for c in range(rank):
                sums += U[a_arr, c] * V[b_arr, c] * W[c_arr, c]
            err = np.sum((sums - t_arr) ** 2)
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            if err < tol:
                # Check if it's close to integers/half-integers
                # And verify exactly
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_full(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    return U_list, V_list, W_list
    
    # Return best state if it's close enough
    if best_state is not None and best_err < 1e-6:
        U_b, V_b, W_b = best_state
        U_list = U_b.tolist()
        V_list = V_b.tolist()
        W_list = W_b.tolist()
        if _verify_full(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # First, return the verified rank-39 construction
    U39, V39, W39 = _construct_rank39()
    if not _verify_full(U39, V39, W39, n, m, p):
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
    
    # Try to find lower rank with remaining time
    best_result = None
    if time.time() < deadline - 10:
        # Try rank 38
        for seed in [42, 123, 999, 111]:
            if time.time() > deadline - 5:
                break
            result = _try_low_rank(38, n, m, p, deadline=deadline - 5, max_rounds=500, seed=seed)
            if result is not None:
                U2, V2, W2 = result
                best_result = (U2, V2, W2)
                break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    return {"U": U39, "V": V39, "W": W39}
# EVOLVE-BLOCK-END
