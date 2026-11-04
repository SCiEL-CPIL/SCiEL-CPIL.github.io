"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time as _time
import numpy as np

def _strassen_2x2x2():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    
    Convention: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k in {0,1}
    U is 2x2, V is 2x2, W is 2x2 (W indexed as W[k,i])
    """
    # Standard Strassen:
    # M1 = (A11-A22)(B21-B22) -> contributes to C11(+1), C22(+1)
    # M2 = (A11+A12)B22       -> contributes to C11(-1), C22(+1)  [C11 = M1+M4-M2+M6, C22 = M1-M3+M5+M7]
    # Wait, let me use the standard formulas:
    # C11 = M1 + M4 - M2 + M6
    # C12 = M3 + M5
    # C21 = M7 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # M1 = (A00-A11)(B10-B11)
    # U1: A00-A11 -> [[1,0],[0,-1]]
    # V1: B10-B11 -> [[0,0],[1,-1]]
    # W1: contributes +1 to C00, +1 to C11 -> W1[k,i]: W1[0,0]=1, W1[1,1]=1
    terms = []
    terms.append(([1,0,0,-1], [0,0,1,-1], [1,0,0,1]))
    
    # M2 = (A00+A01)B11
    # U2: A00+A01 -> [[1,1],[0,0]]
    # V2: B11 -> [[0,0],[0,1]]
    # W2: contributes -1 to C00, +1 to C11 -> W2[0,0]=-1, W2[1,1]=1
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,0,1]))
    
    # M3 = A00(B00-B11)
    # U3: A00 -> [[1,0],[0,0]]
    # V3: B00-B11 -> [[1,0],[0,-1]]
    # W3: contributes +1 to C01, -1 to C11 -> W3[0,1]=1, W3[1,1]=-1
    terms.append(([1,0,0,0], [1,0,0,-1], [0,1,-1,0]))
    
    # M4 = A11(B00+B11)
    # U4: A11 -> [[0,0],[0,1]]
    # V4: B00+B11 -> [[1,0],[0,1]]
    # W4: contributes +1 to C00, +1 to C10 -> W4[0,0]=1, W4[0,1]=1
    terms.append(([0,0,0,1], [1,0,0,1], [1,1,0,0]))
    
    # M5 = (A01-A11)(B00+B01)
    # U5: A01-A11 -> [[0,1],[0,-1]]
    # V5: B00+B01 -> [[1,1],[0,0]]
    # W5: contributes +1 to C01, +1 to C11 -> W5[0,1]=1, W5[1,1]=1
    terms.append(([0,1,0,-1], [1,1,0,0], [0,1,0,1]))
    
    # M6 = (A00+A11)B00
    # U6: A00+A11 -> [[1,0],[0,1]]
    # V6: B00 -> [[1,0],[0,0]]
    # W6: contributes +1 to C00 -> W6[0,0]=1
    terms.append(([1,0,0,1], [1,0,0,0], [1,0,0,0]))
    
    # M7 = (A10-A00)(B01+B11)
    # U7: A10-A00 -> [[-1,0],[1,0]]
    # V7: B01+B11 -> [[0,1],[0,1]]
    # W7: contributes +1 to C10, +1 to C11 -> W7[0,1]=1, W7[1,1]=1
    terms.append(([-1,0,1,0], [0,1,0,1], [0,1,0,1]))
    
    return terms

def _verify_strassen():
    """Verify the Strassen decomposition is correct."""
    n = m = p = 2
    terms = _strassen_2x2x2()
    reconstructed = np.zeros((n, m, p))
    for u_flat, v_flat, w_flat in terms:
        u_mat = np.array(u_flat).reshape(n, m)
        v_mat = np.array(v_flat).reshape(m, p)
        w_mat = np.array(w_flat).reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    reconstructed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
    expected = np.ones((n, m, p))
    return np.allclose(reconstructed, expected)

def _generate_splits(total):
    """Generate all compositions of 'total' into positive integers."""
    results = []
    def helper(remaining, parts_so_far):
        if remaining == 0:
            results.append(tuple(parts_so_far))
            return
        for part in range(1, remaining + 1):
            parts_so_far.append(part)
            helper(remaining - part, parts_so_far)
            parts_so_far.pop()
    helper(total, [])
    return results

def _block_cost(n_b, m_b, p_b):
    """Cost of a block: Strassen if 2x2x2, else schoolbook."""
    if n_b == 2 and m_b == 2 and p_b == 2:
        return 7
    return n_b * m_b * p_b

def _total_rank(n_splits, m_splits, p_splits):
    total = 0
    for n_b in n_splits:
        for m_b in m_splits:
            for p_b in p_splits:
                total += _block_cost(n_b, m_b, p_b)
    return total

def construct():
    start_time = _time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Verify Strassen
    if not _verify_strassen():
        # Fallback to schoolbook if Strassen is wrong
        return _schoolbook(n, m, p)
    
    # Search for optimal splits
    n_splits_list = _generate_splits(n)
    m_splits_list = _generate_splits(m)
    p_splits_list = _generate_splits(p)
    
    best_rank = n * m * p
    best_n_splits = (n,)
    best_m_splits = (m,)
    best_p_splits = (p,)
    
    for ns in n_splits_list:
        if _time.time() > deadline:
            break
        for ms in m_splits_list:
            if _time.time() > deadline:
                break
            for ps in p_splits_list:
                r = _total_rank(ns, ms, ps)
                if r < best_rank:
                    best_rank = r
                    best_n_splits = ns
                    best_m_splits = ms
                    best_p_splits = ps
    
    n_splits = list(best_n_splits)
    m_splits = list(best_m_splits)
    p_splits = list(best_p_splits)
    
    # Calculate offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
        
    # Initialize matrices
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    rank = 0
    
    strassen_patterns = _strassen_2x2x2()
    
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                n_b = n_splits[i_b]
                m_b = m_splits[j_b]
                p_b = p_splits[k_b]
                
                i_off = n_offsets[i_b]
                j_off = m_offsets[j_b]
                k_off = p_offsets[k_b]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    for u_flat, v_flat, w_flat in strassen_patterns:
                        col = rank
                        u_mat = np.array(u_flat).reshape(n_b, m_b)
                        v_mat = np.array(v_flat).reshape(m_b, p_b)
                        w_mat = np.array(w_flat).reshape(p_b, n_b)
                        for li in range(n_b):
                            for lj in range(m_b):
                                gi = i_off + li
                                gj = j_off + lj
                                U[gi * m + gj][col] = u_mat[li, lj]
                        for lj in range(m_b):
                            for lk in range(p_b):
                                gj = j_off + lj
                                gk = k_off + lk
                                V[gj * p + gk][col] = v_mat[lj, lk]
                        for lk in range(p_b):
                            for li in range(n_b):
                                gk = k_off + lk
                                gi = i_off + li
                                W[gk * n + gi][col] = w_mat[lk, li]
                        rank += 1
                else:
                    for li in range(n_b):
                        for lj in range(m_b):
                            for lk in range(p_b):
                                col = rank
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                rank += 1
    
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Pure schoolbook decomposition."""
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    rank = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][rank] = 1
                V[j*p+k][rank] = 1
                W[k*n+i][rank] = 1
                rank += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
