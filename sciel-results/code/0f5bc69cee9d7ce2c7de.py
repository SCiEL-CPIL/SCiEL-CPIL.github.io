"""Matrix multiplication 2x3 by 3x9 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix mult.
    
    Returns dict with U (6 x rank), V (27 x rank), W (18 x rank) where:
    - U[i*m + j, c] corresponds to first matrix element a[i,j]
    - V[j*p + k, c] corresponds to second matrix element b[j,k]
    - W[k*n + i, c] corresponds to output element c[k,i] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    nm = n * m   # 6
    mp = m * p   # 27
    np_ = n * p  # 18
    
    # Trivial solution: rank = n*m*p = 54
    def trivial_solution():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((np_, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = trivial_solution()
    
    # Strategy: The tensor for 2x3 * 3x9 is the direct sum of three 2x3 * 3x3 tensors.
    # We search for a low-rank decomposition of the 2x3 * 3x3 sub-tensor.
    # If we find rank r_sub, the total rank is 3 * r_sub.
    # Known: rank(2x3 * 3x3) = 11. Total rank = 33.
    # Trivial rank(2x3 * 3x3) = 18. Total rank = 54.
    
    def verify_sub_decomposition(U_sub, V_sub, W_sub, n_s, m_s, p_s):
        """Verify decomposition for n_s x m_s * m_s x p_s."""
        r = U_sub.shape[1]
        # Check non-zero entries
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    a = i * m_s + j
                    b = j * p_s + k
                    c = k * n_s + i
                    val = np.sum(U_sub[a, :] * V_sub[b, :] * W_sub[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check zero entries (sample)
        np.random.seed(42)
        for _ in range(100):
            a = np.random.randint(n_s * m_s)
            b = np.random.randint(m_s * p_s)
            c = np.random.randint(p_s * n_s)
            i = a // m_s
            j = a % m_s
            j2 = b // p_s
            k = b % p_s
            k2 = c // n_s
            i2 = c % n_s
            if j == j2 and i == i2 and k == k2:
                continue
            val = np.sum(U_sub[a, :] * V_sub[b, :] * W_sub[c, :])
            if abs(val) > 1e-9:
                return False
        return True

    def find_sub_decomposition(n_s, m_s, p_s, target_rank, max_time=100):
        """Search for rank decomposition of n_s x m_s * m_s x p_s."""
        start = time.time()
        nm_s = n_s * m_s
        mp_s = m_s * p_s
        np_s = n_s * p_s
        
        # Try random initializations
        for attempt in range(20):
            if time.time() > start + max_time:
                break
            if time.time() > deadline:
                break
            
            np.random.seed(attempt)
            U = np.random.randn(nm_s, target_rank)
            V = np.random.randn(mp_s, target_rank)
            W = np.random.randn(np_s, target_rank)
            
            # Local search
            best_err = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for it in range(5000):
                if time.time() > start + max_time or time.time() > deadline:
                    break
                
                # Compute error
                err = 0.0
                for i in range(n_s):
                    for j in range(m_s):
                        for k in range(p_s):
                            a = i * m_s + j
                            b = j * p_s + k
                            c = k * n_s + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                
                # Sample zeros
                for _ in range(20):
                    a = np.random.randint(nm_s)
                    b = np.random.randint(mp_s)
                    c = np.random.randint(np_s)
                    i = a // m_s
                    j = a % m_s
                    j2 = b // p_s
                    k = b % p_s
                    k2 = c // n_s
                    i2 = c % n_s
                    if j == j2 and i == i2 and k == k2:
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err += val ** 2
                
                if err < best_err:
                    best_err = err
                    best_state = (U.copy(), V.copy(), W.copy())
                    if best_err < 1e-8:
                        break
                
                # Perturbation
                if np.random.rand() < 0.33:
                    U[np.random.randint(nm_s), np.random.randint(target_rank)] += np.random.randn() * 0.5
                elif np.random.rand() < 0.66:
                    V[np.random.randint(mp_s), np.random.randint(target_rank)] += np.random.randn() * 0.5
                else:
                    W[np.random.randint(np_s), np.random.randint(target_rank)] += np.random.randn() * 0.5
            
            if best_err < 1e-8:
                U, V, W = best_state
                # Round to half-integers
                U_h = np.round(U * 2) / 2
                V_h = np.round(V * 2) / 2
                W_h = np.round(W * 2) / 2
                if verify_sub_decomposition(U_h, V_h, W_h, n_s, m_s, p_s):
                    return U_h, V_h, W_h
        
        return None

    # Search for rank 11 sub-decomposition
    U_sub, V_sub, W_sub = find_sub_decomposition(2, 3, 3, 11, max_time=100)
    
    if U_sub is not None:
        # Construct full solution
        r_sub = U_sub.shape[1]
        r_full = r_sub * 3
        U_full = np.zeros((nm, r_full), dtype=float)
        V_full = np.zeros((mp, r_full), dtype=float)
        W_full = np.zeros((np_, r_full), dtype=float)
        
        # U is shared (same for all 3 blocks)
        U_full[:, 0:r_sub] = U_sub
        U_full[:, r_sub:2*r_sub] = U_sub
        U_full[:, 2*r_sub:3*r_sub] = U_sub
        
        # V: j in 0..2, k in 0..8
        # Block 0: k=0,1,2. V[j, 0:3] = V_sub[j, 0:3]
        # Block 1: k=3,4,5. V[j, 3:6] = V_sub[j, 0:3]
        # Block 2: k=6,7,8. V[j, 6:9] = V_sub[j, 0:3]
        for block in range(3):
            k_start = block * 3
            for j in range(3):
                V_full[j * 9 + k_start : j * 9 + k_start + 3, block * r_sub : (block + 1) * r_sub] = V_sub[j * 3 : j * 3 + 3, :]
        
        # W: k in 0..8, i in 0..1
        # Block 0: k=0,1,2. W[0:3, i] = W_sub[0:3, i]
        # Block 1: k=3,4,5. W[3:6, i] = W_sub[0:3, i]
        # Block 2: k=6,7,8. W[6:9, i] = W_sub[0:3, i]
        for block in range(3):
            k_start = block * 3
            W_full[k_start : k_start + 3, :] = W_sub[:, block * r_sub : (block + 1) * r_sub] # Wait, W_sub is (6 x r_sub).
            # W_full rows are k*n+i.
            # W_sub rows are k_s*n_s+i_s.
            # W_full[k_start + k_s, i] = W_sub[k_s*n_s+i, block*r_sub + r]
            
            # Correct mapping:
            for k_s in range(3):
                for i in range(2):
                    idx_w_full = (k_start + k_s) * 2 + i
                    idx_w_sub = k_s * 2 + i
                    W_full[idx_w_full, block * r_sub : (block + 1) * r_sub] = W_sub[idx_w_sub, :]
        
        # Verify full solution
        def verify_full(U, V, W):
            r = U.shape[1]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
        
        if verify_full(U_full, V_full, W_full):
            best_result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Convert to list format and round to nearest half-integer if possible
    def to_half_integer(mat):
        mat = np.round(np.array(mat) * 2) / 2
        return mat.tolist()
    
    U_list = to_half_integer(best_result["U"])
    V_list = to_half_integer(best_result["V"])
    W_list = to_half_integer(best_result["W"])
    
    return {"U": U_list, "V": V_list, "W": W_list}
