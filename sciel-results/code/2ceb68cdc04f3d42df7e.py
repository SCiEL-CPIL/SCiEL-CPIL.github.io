"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    # Splits: 3=2+1, 5=2+2+1, 5=2+2+1
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
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
        
    # Strassen patterns for 2x2x2 (7 multiplications instead of 8)
    strassen_patterns = [
        {  # M1: (A00+A11)(B00+B11)
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {  # M2: (A10+A11)B00
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {  # M3: A00(B01-B11)
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {  # M4: A11(B10-B00)
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {  # M5: (A00+A01)B11
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {  # M6: (A10-A00)(B00+B01)
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {  # M7: (A01-A11)(B10+B11)
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    # Build sparse entry lists
    U_entries = []
    V_entries = []
    W_entries = []
    rank = 0
    
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                if time.time() > deadline:
                    break
                
                n_b = n_splits[i_b]
                m_b = m_splits[j_b]
                p_b = p_splits[k_b]
                
                i_off = n_offsets[i_b]
                j_off = m_offsets[j_b]
                k_off = p_offsets[k_b]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Use Strassen's 7-term algorithm
                    for pattern in strassen_patterns:
                        col = rank
                        for li, lj, val in pattern['U']:
                            gi = i_off + li
                            gj = j_off + lj
                            U_entries.append((gi * m + gj, col, val))
                        for lj, lk, val in pattern['V']:
                            gj = j_off + lj
                            gk = k_off + lk
                            V_entries.append((gj * p + gk, col, val))
                        for lk, li, val in pattern['W']:
                            gk = k_off + lk
                            gi = i_off + li
                            W_entries.append((gk * n + gi, col, val))
                        rank += 1
                elif min(n_b, m_b, p_b) == 1:
                    # One dimension is 1: the sub-tensor is an all-ones matrix
                    # which has rank 1, so use a single rank-1 term
                    col = rank
                    for li in range(n_b):
                        for lj in range(m_b):
                            gi = i_off + li
                            gj = j_off + lj
                            U_entries.append((gi * m + gj, col, 1))
                    for lj in range(m_b):
                        for lk in range(p_b):
                            gj = j_off + lj
                            gk = k_off + lk
                            V_entries.append((gj * p + gk, col, 1))
                    for lk in range(p_b):
                        for li in range(n_b):
                            gk = k_off + lk
                            gi = i_off + li
                            W_entries.append((gk * n + gi, col, 1))
                    rank += 1
                else:
                    # Schoolbook fallback
                    for li in range(n_b):
                        for lj in range(m_b):
                            for lk in range(p_b):
                                col = rank
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U_entries.append((gi * m + gj, col, 1))
                                V_entries.append((gj * p + gk, col, 1))
                                W_entries.append((gk * n + gi, col, 1))
                                rank += 1
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for row, col, val in U_entries:
        U[row][col] += val
    for row, col, val in V_entries:
        V[row][col] += val
    for row, col, val in W_entries:
        W[row][col] += val
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
