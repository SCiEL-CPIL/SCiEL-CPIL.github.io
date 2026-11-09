"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Splits
    n_splits = [(0, 2), (2, 1)]
    m_splits = [(0, 2), (2, 2)]
    p_splits = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    # Precompute Strassen forms for 2x2x2
    # u, v, w are lists of 4 elements (local indices 0,1,2,3)
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_forms = [
        # (u, v, w)
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Calculate total rank
    total_rank = 0
    for n_start, n_len in n_splits:
        for m_start, m_len in m_splits:
            for p_start, p_len in p_splits:
                if n_len == 2 and m_len == 2 and p_len == 2:
                    total_rank += 7
                else:
                    total_rank += n_len * m_len * p_len
                    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_col = 0
    
    for n_start, n_len in n_splits:
        for m_start, m_len in m_splits:
            for p_start, p_len in p_splits:
                if n_len == 2 and m_len == 2 and p_len == 2:
                    # Use Strassen
                    for u_local, v_local, w_local in strassen_forms:
                        # Map local indices to global
                        # u_local index l corresponds to i', j'
                        # i' = l // 2, j' = l % 2
                        for l in range(4):
                            val = u_local[l]
                            if val != 0:
                                i_prime = l // 2
                                j_prime = l % 2
                                i_global = n_start + i_prime
                                j_global = m_start + j_prime
                                U[i_global * m + j_global][current_col] += val
                                
                            val = v_local[l]
                            if val != 0:
                                j_prime = l // 2
                                k_prime = l % 2
                                j_global = m_start + j_prime
                                k_global = p_start + k_prime
                                V[j_global * p + k_global][current_col] += val
                                
                            val = w_local[l]
                            if val != 0:
                                k_prime = l // 2
                                i_prime = l % 2
                                k_global = p_start + k_prime
                                i_global = n_start + i_prime
                                W[k_global * n + i_global][current_col] += val
                                
                        current_col += 1
                else:
                    # Use Schoolbook
                    for i_prime in range(n_len):
                        for j_prime in range(m_len):
                            for k_prime in range(p_len):
                                i_global = n_start + i_prime
                                j_global = m_start + j_prime
                                k_global = p_start + k_prime
                                
                                U[i_global * m + j_global][current_col] = 1
                                V[j_global * p + k_global][current_col] = 1
                                W[k_global * n + i_global][current_col] = 1
                                
                                current_col += 1
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
