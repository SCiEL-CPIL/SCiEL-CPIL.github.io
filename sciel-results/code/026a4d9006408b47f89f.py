import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 3, 5, 7
    
    # Helper to create schoolbook decomposition for M_{n,m,p}
    def schoolbook(n, m, p):
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen decomposition for M_{2,2,2}
    def strassen_222():
        # Returns U (4x7), V (4x7), W (4x7)
        # Indices:
        # A: 0: A11, 1: A12, 2: A21, 3: A22
        # B: 0: B11, 1: B12, 2: B21, 3: B22
        # C: 0: C11, 1: C12, 2: C21, 3: C22
        
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # C11 += M1
        W[3, 0] = 1  # C22 += M1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C21 += M2
        W[3, 1] = -1 # C22 -= M2
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C12 += M3
        W[3, 2] = 1  # C22 += M3
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # C11 += M4
        W[2, 3] = 1  # C21 += M4
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C11 -= M5
        W[1, 4] = 1  # C12 += M5
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C22 += M6
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C11 += M7
        W[3, 6] = -1 # C22 -= M7
        
        return U, V, W

    # Block decomposition
    # A (3x5) split into A1 (1x5), A2 (2x5)
    # B (5x7) split into B1 (5x3), B2 (5x4)
    # Actually, split B into B_left (5x3) and B_right (5x4) is not enough, 
    # we need to split the inner dimension m=5 as well for the block structure.
    
    # Let's stick to the 4-block structure:
    # A = [A11 (1x2) A12 (1x3); A21 (2x2) A22 (2x3)]
    # B = [B11 (2x3) B12 (2x4); B21 (3x3) B22 (3x4)]
    
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We will accumulate U, V, W for each term.
    # Note: The indices in U, V, W correspond to the full A, B, C matrices.
    # U has shape (15, R), V (35, R), W (21, R).
    
    U_total = []
    V_total = []
    W_total = []
    
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset):
        # i_offset: row offset in A (and C)
        # j_offset: col offset in A (and row offset in B)
        # k_offset: col offset in B (and C)
        
        n_b, m_b, p_b = U_block.shape[0]//m_b if False else 0, 0, 0 # Dummy
        # Actually, we know the shapes from the block definition.
        # U_block is (n_b * m_b, rank)
        # V_block is (m_b * p_b, rank)
        # W_block is (p_b * n_b, rank)
        
        rank = U_block.shape[1]
        
        U_ext = np.zeros((15, rank), dtype=np.int64)
        V_ext = np.zeros((35, rank), dtype=np.int64)
        W_ext = np.zeros((21, rank), dtype=np.int64)
        
        # Map U_block indices to full U
        for i in range(U_block.shape[0]):
            # i is local index in block (n_b * m_b)
            # We need to know n_b and m_b to map to global i, j
            # This helper is tricky without n_b, m_b.
            # Let's pass n_b, m_b, p_b.
            pass
        return U_ext, V_ext, W_ext

    # Better approach: Construct each term's U, V, W in the full space directly.
    
    def make_term(n_b, m_b, p_b, U_b, V_b, W_b, i_off, j_off, k_off):
        rank = U_b.shape[1]
        U = np.zeros((15, rank), dtype=np.int64)
        V = np.zeros((35, rank), dtype=np.int64)
        W = np.zeros((21, rank), dtype=np.int64)
        
        for i in range(n_b):
            for j in range(m_b):
                local_u_idx = i * m_b + j
                global_u_idx = (i_off + i) * 5 + (j_off + j)
                U[global_u_idx, :] = U_b[local_u_idx, :]
                
        for j in range(m_b):
            for k in range(p_b):
                local_v_idx = j * p_b + k
                global_v_idx = (j_off + j) * 7 + (k_off + k)
                V[global_v_idx, :] = V_b[local_v_idx, :]
                
        for k in range(p_b):
            for i in range(n_b):
                local_w_idx = k * n_b + i
                global_w_idx = (k_off + k) * 3 + (i_off + i)
                W[global_w_idx, :] = W_b[local_w_idx, :]
                
        return U, V, W

    # Term 1: A11 B11 (1x2 * 2x3)
    U11, V11, W11 = schoolbook(1, 2, 3)
    U_t1, V_t1, W_t1 = make_term(1, 2, 3, U11, V11, W11, 0, 0, 0)
    
    # Term 2: A12 B21 (1x3 * 3x3)
    U12, V12, W12 = schoolbook(1, 3, 3)
    U_t2, V_t2, W_t2 = make_term(1, 3, 3, U12, V12, W12, 0, 2, 0)
    
    # Term 3: A11 B12 (1x2 * 2x4)
    U13, V13, W13 = schoolbook(1, 2, 4)
    U_t3, V_t3, W_t3 = make_term(1, 2, 4, U13, V13, W13, 0, 0, 3)
    
    # Term 4: A12 B22 (1x3 * 3x4)
    U14, V14, W14 = schoolbook(1, 3, 4)
    U_t4, V_t4, W_t4 = make_term(1, 3, 4, U14, V14, W14, 0, 2, 3)
    
    # Term 5: A21 B11 (2x2 * 2x3) -> M_{2,2,3}
    # M_{2,2,3} = M_{2,2,1} + M_{2,2,2}
    # M_{2,2,1} (2x2 * 2x1)
    U221, V221, W221 = schoolbook(2, 2, 1)
    # M_{2,2,2} (2x2 * 2x2)
    U222, V222, W222 = strassen_222()
    # Combine M_{2,2,3}
    U_M223 = np.hstack([U221, U222])
    V_M223 = np.hstack([V221, V222])
    W_M223 = np.hstack([W221, W222])
    # Wait, the p dimension is 3. 
    # M_{2,2,1} contributes to k=0.
    # M_{2,2,2} contributes to k=1,2.
    # My make_term handles the offset.
    # But I need to combine the sub-terms of M_{2,2,3} correctly.
    # Term 5a: A21 B11_part1 (2x2 * 2x1) at k=0
    U_t5a, V_t5a, W_t5a = make_term(2, 2, 1, U221, V221, W221, 1, 0, 0)
    # Term 5b: A21 B11_part2 (2x2 * 2x2) at k=1
    U_t5b, V_t5b, W_t5b = make_term(2, 2, 2, U222, V222, W222, 1, 0, 1)
    
    # Term 6: A22 B21 (2x3 * 3x3) -> M_{2,3,3}
    # M_{2,3,3} = M_{2,1,3} + M_{2,2,3}
    # M_{2,1,3} (2x1 * 1x3)
    U213, V213, W213 = schoolbook(2, 1, 3)
    # M_{2,2,3} (2x2 * 2x3) - reuse logic from Term 5
    # Term 6a: A22_part1 B21 (2x1 * 1x3) at j=0 (local), so global j=2
    U_t6a, V_t6a, W_t6a = make_term(2, 1, 3, U213, V213, W213, 1, 2, 0)
    # Term 6b: A22_part2 B21 (2x2 * 2x3) at j=1 (local), so global j=3
    # Need M_{2,2,3} again.
    U_M223_b = np.hstack([U221, U222])
    V_M223_b = np.hstack([V221, V222])
    W_M223_b = np.hstack([W221, W222])
    # Wait, make_term expects a single U_b, V_b, W_b for the whole block.
    # If I pass the concatenated M_{2,2,3} to make_term, it will work.
    # But I need to be careful with the offsets.
    # M_{2,2,3} has n=2, m=2, p=3.
    U_t6b, V_t6b, W_t6b = make_term(2, 2, 3, U_M223_b, V_M223_b, W_M223_b, 1, 3, 0)
    
    # Term 7: A21 B12 (2x2 * 2x4) -> M_{2,2,4}
    # M_{2,2,4} = M_{2,2,2} + M_{2,2,2}
    # Term 7a: k=3,4 (local 0,1)
    U_t7a, V_t7a, W_t7a = make_term(2, 2, 2, U222, V222, W222, 1, 0, 3)
    # Term 7b: k=5,6 (local 2,3)
    U_t7b, V_t7b, W_t7b = make_term(2, 2, 2, U222, V222, W222, 1, 0, 5)
    
    # Term 8: A22 B22 (2x3 * 3x4) -> M_{2,3,4}
    # M_{2,3,4} = M_{2,1,4} + M_{2,2,4}
    # M_{2,1,4} (2x1 * 1x4)
    U214, V214, W214 = schoolbook(2, 1, 4)
    # Term 8a: j=2 (local 0)
    U_t8a, V_t8a, W_t8a = make_term(2, 1, 4, U214, V214, W214, 1, 2, 3)
    # Term 8b: j=3,4 (local 1,2) -> M_{2,2,4}
    # M_{2,2,4} = M_{2,2,2} + M_{2,2,2}
    # Concatenate for make_term
    U_M224 = np.hstack([U222, U222])
    V_M224 = np.hstack([V222, V222])
    W_M224 = np.hstack([W222, W222])
    U_t8b, V_t8b, W_t8b = make_term(2, 2, 4, U_M224, V_M224, W_M224, 1, 3, 3)
    
    # Sum all terms
    U_all = np.sum([U_t1, U_t2, U_t3, U_t4, U_t5a, U_t5b, U_t6a, U_t6b, U_t7a, U_t7b, U_t8a, U_t8b], axis=0)
    V_all = np.sum([V_t1, V_t2, V_t3, V_t4, V_t5a, V_t5b, V_t6a, V_t6b, V_t7a, V_t7b, V_t8a, V_t8b], axis=0)
    W_all = np.sum([W_t1, W_t2, W_t3, W_t4, W_t5a, W_t5b, W_t6a, W_t6b, W_t7a, W_t7b, W_t8a, W_t8b], axis=0)
    
    return {'U': U_all.tolist(), 'V': V_all.tolist(), 'W': W_all.tolist()}
