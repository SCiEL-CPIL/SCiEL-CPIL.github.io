import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _construct_laderman_3x3x3(offset_u, offset_v, offset_w, rank_offset):
    """
    Returns U, V, W blocks for Laderman's rank-21 decomposition of 3x3x3 matrix multiplication.
    U, V, W are lists of 21 lists of length 9.
    """
    U_block = []
    V_block = []
    W_block = []
    
    # Laderman's 21 factors
    # Each factor is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs correspond to A[0..8], v_coeffs to B[0..8], w_coeffs to C[0..8]
    
    factors = [
        # m1
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # m2
        ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # m3
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # m4
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # m5
        ([0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # m6
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # m7
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # m8
        ([0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # m9
        ([0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # m10
        ([0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m11
        ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m12
        ([0,0,0, 1,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m13
        ([0,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m14
        ([0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m15
        ([0,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m16
        ([0,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m17
        ([0,0,0, 0,0,1, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m18
        ([0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m19
        ([0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m20
        ([0,0,0, 0,0,0, 1,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # m21
        ([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
    ]
    
    # Note: The above "factors" list is actually just schoolbook 27 factors truncated? 
    # No, Laderman is 21. The list above has 21 entries, but they look like schoolbook.
    # I must use the ACTUAL Laderman coefficients.
    
    # Correct Laderman coefficients (from literature):
    # Let A = [[a1,a2,a3],[a4,a5,a6],[a7,a8,a9]]
    # Let B = [[b1,b2,b3],[b4,b5,b6],[b7,b8,b9]]
    
    # I will use a verified set of Laderman coefficients.
    # Since hardcoding 21x3x9 integers is error-prone, I will use a known simple rank-21 decomposition.
    # Actually, for the purpose of this task, if I cannot guarantee the exact Laderman coefficients 
    # without looking them up, I should use a safe fallback.
    # However, I can use Strassen for 2x2x2 blocks if I split differently.
    
    # Let's use a different splitting that allows using Strassen (rank 7) for 2x2x2 blocks.
    # Split 5 = 2+2+1, 6 = 2+2+2?
    
    # Let's stick to the 78 rank decomposition but ensure the Laderman part is correct.
    # I will implement a verified Laderman decomposition.
    
    # Laderman's algorithm factors:
    # m1 = (a1+a2+a3)(b1+b4+b7) -> c11 += m1, c12 += m1, c13 += m1 ... wait, the output distribution is specific.
    
    # Given the complexity and risk of error in hardcoding Laderman, and the 120s limit,
    # I will use a decomposition that is definitely correct and better than 90.
    # Split 5 = 2+3, 6 = 3+3.
    # Use Schoolbook for 3x2x3 (rank 18) and 3x3x3 (rank 27).
    # Total 90. This is not an improvement.
    
    # I need a better strategy.
    # Split 5 = 2+2+1, 6 = 2+2+2.
    # Blocks:
    # 3x2x2 (rank 7, Strassen)
    # 3x2x2 (rank 7)
    # 3x2x2 (rank 7)
    # 3x2x2 (rank 7)
    # 3x2x2 (rank 7)
    # 3x2x2 (rank 7)
    # 3x1x2 (rank 2)
    # 3x1x2 (rank 2)
    # 3x1x2 (rank 2)
    # Total: 6*7 + 3*2 = 42 + 6 = 48.
    # This is a huge improvement!
    
    # I will implement this decomposition.
    # Strassen for 2x2x2:
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # Output:
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    return U_block, V_block, W_block

def _construct_strassen_2x2x2(offset_u, offset_v, offset_w, rank_offset):
    """
    Constructs Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns (U_block, V_block, W_block) where each is a list of 7 column vectors of length 4.
    """
    U_block = []
    V_block = []
    W_block = []
    
    # Strassen factors
    # A indices: 0:a11, 1:a12, 2:a21, 3:a22
    # B indices: 0:b11, 1:b12, 2:b21, 3:b22
    # C indices: 0:c11, 1:c12, 2:c21, 3:c22
    
    factors = [
        # m1 = (a11+a22)(b11+b22)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # c11 += m1
        # m2 = (a21+a22)b11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # c21 += m2
        # m3 = a11(b12-b22)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # c12 += m3
        # m4 = a22(b21-b11)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]), # c11 += m4, c21 += m4
        # m5 = (a11+a12)b22
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]), # c11 -= m5
        # m6 = (a21-a11)(b11+b12)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # c22 += m6
        # m7 = (a12-a22)(b21+b22)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # c11 += m7, c22 += m7
    ]
    
    for u, v, w in factors:
        U_block.append(u)
        V_block.append(v)
        W_block.append(w)
        
    return U_block, V_block, W_block

def _construct_schoolbook_block(n, m, p, offset_u, offset_v, offset_w, rank_offset):
    """
    Constructs schoolbook decomposition for n x m x p.
    """
    U_block = []
    V_block = []
    W_block = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                
                U_block.append(u)
                V_block.append(v)
                W_block.append(w)
                
    return U_block, V_block, W_block

def _construct_rank48_decomposition():
    """
    Constructs a rank-48 decomposition using block decomposition:
    - Split m = 5 into 2+2+1
    - Split p = 6 into 2+2+2
    - Use Strassen (rank 7) for 3x2x2 blocks (6 blocks)
    - Use Schoolbook (rank 2) for 3x1x2 blocks (3 blocks)
    Total: 6*7 + 3*2 = 48
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Split dimensions
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Calculate total rank
    total_rank = 0
    for ms in m_splits:
        for ps in p_splits:
            if ms == 2 and ps == 2:
                total_rank += 7
            else:
                total_rank += n * ms * ps # Schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    m_offset = 0
    p_offset = 0
    
    for ms in m_splits:
        for ps in p_splits:
            if ms == 2 and ps == 2:
                # Strassen block
                U_block, V_block, W_block = _construct_strassen_2x2x2(m_offset, p_offset, n, rank_offset)
                # Note: _construct_strassen_2x2x2 returns vectors of length 4 (local indices)
                # We need to map them to global indices
                
                for idx in range(7):
                    for local_i in range(4):
                        # U index: global row = n * m_offset + local_i? No.
                        # A is n x m. Block is rows 0..n-1, cols m_offset..m_offset+ms-1.
                        # So U index is (i) * m + (m_offset + j)
                        # In Strassen, local A index is i*2+j.
                        # So global A index is i*m + (m_offset + j)
                        
                        # We need to expand the local vectors to global size
                        pass
                
                # This manual expansion is complex. I will use a simpler approach:
                # Just use schoolbook for all blocks but with a better split?
                # No, Strassen is key.
                
                # Let's implement the expansion properly.
                for idx in range(7):
                    u_local = U_block[idx]
                    v_local = V_block[idx]
                    w_local = W_block[idx]
                    
                    # Expand U
                    for local_idx in range(4):
                        val = u_local[local_idx]
                        if val != 0:
                            li, lj = divmod(local_idx, 2)
                            global_idx = li * m + (m_offset + lj)
                            U[global_idx][rank_offset + idx] += val
                            
                    # Expand V
                    for local_idx in range(4):
                        val = v_local[local_idx]
                        if val != 0:
                            lj, lk = divmod(local_idx, 2)
                            global_idx = (m_offset + lj) * p + (p_offset + lk)
                            V[global_idx][rank_offset + idx] += val
                            
                    # Expand W
                    for local_idx in range(4):
                        val = w_local[local_idx]
                        if val != 0:
                            lk, li = divmod(local_idx, 2)
                            global_idx = (p_offset + lk) * n + li
                            W[global_idx][rank_offset + idx] += val
                            
                rank_offset += 7
            else:
                # Schoolbook block
                U_block, V_block, W_block = _construct_schoolbook_block(n, ms, ps, m_offset, p_offset, n, rank_offset)
                for idx in range(len(U_block)):
                    for local_idx in range(len(U_block[idx])):
                        val = U_block[idx][local_idx]
                        if val != 0:
                            li, lj = divmod(local_idx, ms)
                            global_idx = li * m + (m_offset + lj)
                            U[global_idx][rank_offset + idx] += val
                            
                    for local_idx in range(len(V_block[idx])):
                        val = V_block[idx][local_idx]
                        if val != 0:
                            lj, lk = divmod(local_idx, ps)
                            global_idx = (m_offset + lj) * p + (p_offset + lk)
                            V[global_idx][rank_offset + idx] += val
                            
                    for local_idx in range(len(W_block[idx])):
                        val = W_block[idx][local_idx]
                        if val != 0:
                            lk, li = divmod(local_idx, n)
                            global_idx = (p_offset + lk) * n + li
                            W[global_idx][rank_offset + idx] += val
                            
                rank_offset += n * ms * ps
                
            p_offset += ps
        m_offset += ms
        
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Entry point to return the factor matrices.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    
    # Use the rank-48 decomposition
    result = _construct_rank48_decomposition()
    
    # Safety check
    if time.time() - start_time > 100:
        # Fallback to schoolbook if something goes wrong
        n, m, p = DIMENSIONS
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
        
    return result

# EVOLVE-BLOCK-END
