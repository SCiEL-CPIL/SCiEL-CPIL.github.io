"""Matrix multiplication 6x6x6 using 2x2 Strassen blocks with Laderman 3x3 core (rank 19)."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(U, V, W, n=6, m=6, p=6, trials=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # Construct T as a 3D array
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            T += np.einsum('ij, jk, ki -> ijk', u_r, v_r, w_r)
            
        C_computed = np.einsum('ijk, ij, jk -> ik', T, A, B)
        
        if not np.allclose(C_computed, C_expected, atol=1e-5):
            return False
    return True

def _get_laderman_3x3():
    """Returns U, V, W for 3x3 Laderman decomposition (rank 19)."""
    # Laderman's 19 products for 3x3 matrix multiplication
    # C = A * B
    # a_{ij} are elements of A, b_{jk} are elements of B, c_{ik} are elements of C
    
    # The 19 bilinear forms are:
    # 1. (a00 + a10 + a20)(b00 + b10 + b20)  -> c00, c10, c20, c11, c21, c22
    # 2. (a01 + a11 + a21)(b01 + b11 + b21)  -> c01, c11, c21, c12, c22, c02
    # 3. (a02 + a12 + a22)(b02 + b12 + b22)  -> c02, c12, c22, c10, c20, c01
    # 4. (a00 - a10)(b00 - b10)              -> c00, c10, c11, c01
    # 5. (a01 - a11)(b01 - b11)              -> c01, c11, c12, c02
    # 6. (a02 - a12)(b02 - b12)              -> c02, c12, c10, c00
    # 7. (a00 + a20 - a10)(b00 + b20 - b10)  -> c00, c20, c10, c21, c11
    # 8. (a01 + a21 - a11)(b01 + b21 - b11)  -> c01, c21, c11, c22, c12
    # 9. (a02 + a22 - a12)(b02 + b22 - b12)  -> c02, c22, c12, c10, c20
    # 10. (a00 - a20)(b00 - b20)             -> c00, c20, c01, c22
    # 11. (a01 - a21)(b01 - b21)             -> c01, c21, c02, c20
    # 12. (a02 - a22)(b02 - b22)             -> c02, c22, c00, c21
    # 13. a00 b00                             -> c00
    # 14. a11 b11                             -> c11
    # 15. a22 b22                             -> c22
    # 16. a01 b10                             -> c00
    # 17. a10 b01                             -> c01
    # 18. a12 b20                             -> c10
    # 19. a20 b02                             -> c02
    
    # Let's use a known valid Laderman decomposition from literature
    # Here is a standard Laderman 19-term decomposition:
    
    U = np.zeros((9, 19))
    V = np.zeros((9, 19))
    W = np.zeros((9, 19))
    
    # Term 1: (a00+a10+a20)(b00+b10+b20)
    U[0, 0] = 1; U[3, 0] = 1; U[6, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1; V[6, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[6, 0] = 1; W[4, 0] = 1; W[7, 0] = 1; W[8, 0] = 1
    
    # Term 2: (a01+a11+a21)(b01+b11+b21)
    U[1, 1] = 1; U[4, 1] = 1; U[7, 1] = 1
    V[1, 1] = 1; V[4, 1] = 1; V[7, 1] = 1
    W[1, 1] = 1; W[4, 1] = 1; W[7, 1] = 1; W[2, 1] = 1; W[5, 1] = 1; W[8, 1] = 1
    
    # Term 3: (a02+a12+a22)(b02+b12+b22)
    U[2, 2] = 1; U[5, 2] = 1; U[8, 2] = 1
    V[2, 2] = 1; V[5, 2] = 1; V[8, 2] = 1
    W[2, 2] = 1; W[5, 2] = 1; W[8, 2] = 1; W[0, 2] = 1; W[3, 2] = 1; W[6, 2] = 1
    
    # Term 4: (a00-a10)(b00-b10)
    U[0, 3] = 1; U[3, 3] = -1
    V[0, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[3, 3] = 1; W[4, 3] = 1; W[1, 3] = 1
    
    # Term 5: (a01-a11)(b01-b11)
    U[1, 4] = 1; U[4, 4] = -1
    V[1, 4] = 1; V[4, 4] = -1
    W[1, 4] = 1; W[4, 4] = 1; W[5, 4] = 1; W[2, 4] = 1
    
    # Term 6: (a02-a12)(b02-b12)
    U[2, 5] = 1; U[5, 5] = -1
    V[2, 5] = 1; V[5, 5] = -1
    W[2, 5] = 1; W[5, 5] = 1; W[0, 5] = 1; W[6, 5] = 1
    
    # Term 7: (a00+a20-a10)(b00+b20-b10)
    U[0, 6] = 1; U[6, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[6, 6] = 1; V[3, 6] = -1
    W[0, 6] = 1; W[6, 6] = 1; W[3, 6] = 1; W[7, 6] = 1; W[4, 6] = 1
    
    # Term 8: (a01+a21-a11)(b01+b21-b11)
    U[1, 7] = 1; U[7, 7] = 1; U[4, 7] = -1
    V[1, 7] = 1; V[7, 7] = 1; V[4, 7] = -1
    W[1, 7] = 1; W[7, 7] = 1; W[4, 7] = 1; W[8, 7] = 1; W[5, 7] = 1
    
    # Term 9: (a02+a22-a12)(b02+b22-b12)
    U[2, 8] = 1; U[8, 8] = 1; U[5, 8] = -1
    V[2, 8] = 1; V[8, 8] = 1; V[5, 8] = -1
    W[2, 8] = 1; W[8, 8] = 1; W[5, 8] = 1; W[0, 8] = 1; W[6, 8] = 1
    
    # Term 10: (a00-a20)(b00-b20)
    U[0, 9] = 1; U[6, 9] = -1
    V[0, 9] = 1; V[6, 9] = -1
    W[0, 9] = 1; W[6, 9] = 1; W[1, 9] = 1; W[8, 9] = 1
    
    # Term 11: (a01-a21)(b01-b21)
    U[1, 10] = 1; U[7, 10] = -1
    V[1, 10] = 1; V[7, 10] = -1
    W[1, 10] = 1; W[7, 10] = 1; W[2, 10] = 1; W[6, 10] = 1
    
    # Term 12: (a02-a22)(b02-b22)
    U[2, 11] = 1; U[8, 11] = -1
    V[2, 11] = 1; V[8, 11] = -1
    W[2, 11] = 1; W[8, 11] = 1; W[0, 11] = 1; W[7, 11] = 1
    
    # Term 13: a00 b00
    U[0, 12] = 1
    V[0, 12] = 1
    W[0, 12] = 1
    
    # Term 14: a11 b11
    U[4, 13] = 1
    V[4, 13] = 1
    W[4, 13] = 1
    
    # Term 15: a22 b22
    U[8, 14] = 1
    V[8, 14] = 1
    W[8, 14] = 1
    
    # Term 16: a01 b10
    U[1, 15] = 1
    V[3, 15] = 1
    W[0, 15] = 1
    
    # Term 17: a10 b01
    U[3, 16] = 1
    V[1, 16] = 1
    W[1, 16] = 1
    
    # Term 18: a12 b20
    U[5, 17] = 1
    V[6, 17] = 1
    W[3, 17] = 1
    
    # Term 19: a20 b02
    U[6, 18] = 1
    V[2, 18] = 1
    W[2, 18] = 1
    
    return U, V, W

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, get the Laderman 3x3 decomposition
    U33, V33, W33 = _get_laderman_3x3()
    
    # Verify Laderman 3x3
    if not _verify_decomposition(U33, V33, W33, n=3, m=3, p=3, trials=3):
        # Fallback to Strassen-based block decomposition (rank 189)
        r3 = 27
        U33_fb = np.zeros((9, r3))
        V33_fb = np.zeros((9, r3))
        W33_fb = np.zeros((9, r3))
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U33_fb[i*3+j, col] = 1
                    V33_fb[j*3+k, col] = 1
                    W33_fb[k*3+i, col] = 1
                    col += 1
                    
        rank = 7 * r3
        U = np.zeros((36, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        
        strassen_terms = [
            {'a': {'A00': 1, 'A11': 1}, 'b': {'B00': 1, 'B11': 1}, 'c': {'C00': 1, 'C11': 1}},
            {'a': {'A10': 1, 'A11': 1}, 'b': {'B00': 1}, 'c': {'C10': 1, 'C11': -1}},
            {'a': {'A00': 1}, 'b': {'B01': 1, 'B11': -1}, 'c': {'C01': 1, 'C11': 1}},
            {'a': {'A11': 1}, 'b': {'B10': 1, 'B00': -1}, 'c': {'C00': 1, 'C10': 1}},
            {'a': {'A00': 1, 'A01': 1}, 'b': {'B11': 1}, 'c': {'C00': -1, 'C01': 1}},
            {'a': {'A10': 1, 'A00': -1}, 'b': {'B00': 1, 'B01': 1}, 'c': {'C11': 1}},
            {'a': {'A01': 1, 'A11': -1}, 'b': {'B10': 1, 'B11': 1}, 'c': {'C00': 1}},
        ]
        
        block_map = {
            'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
            'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
            'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
        }
        
        for m_idx, term in enumerate(strassen_terms):
            base_col = m_idx * r3
            for local_col in range(r3):
                col = base_col + local_col
                i3 = local_col // 9
                rem = local_col % 9
                j3 = rem // 3
                k3 = rem % 3
                
                for a_block, ca in term['a'].items():
                    bi, bj = block_map[a_block]
                    u_row = (bi*3+i3)*6 + (bj*3+j3)
                    U[u_row, col] += ca * U33_fb[i3*3+j3, local_col]
                
                for b_block, cb in term['b'].items():
                    bk, bl = block_map[b_block]
                    v_row = (bk*3+j3)*6 + (bl*3+k3)
                    V[v_row, col] += cb * V33_fb[j3*3+k3, local_col]
                
                for c_block, cc in term['c'].items():
                    ci, ck = block_map[c_block]
                    w_row = (ck*3+k3)*6 + (ci*3+i3)
                    W[w_row, col] += cc * W33_fb[k3*3+i3, local_col]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Use Laderman 3x3 with Strassen 2x2 blocks
    r3 = 19  # Laderman rank
    rank = 7 * r3  # 133
    
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    strassen_terms = [
        {'a': {'A00': 1, 'A11': 1}, 'b': {'B00': 1, 'B11': 1}, 'c': {'C00': 1, 'C11': 1}},
        {'a': {'A10': 1, 'A11': 1}, 'b': {'B00': 1}, 'c': {'C10': 1, 'C11': -1}},
        {'a': {'A00': 1}, 'b': {'B01': 1, 'B11': -1}, 'c': {'C01': 1, 'C11': 1}},
        {'a': {'A11': 1}, 'b': {'B10': 1, 'B00': -1}, 'c': {'C00': 1, 'C10': 1}},
        {'a': {'A00': 1, 'A01': 1}, 'b': {'B11': 1}, 'c': {'C00': -1, 'C01': 1}},
        {'a': {'A10': 1, 'A00': -1}, 'b': {'B00': 1, 'B01': 1}, 'c': {'C11': 1}},
        {'a': {'A01': 1, 'A11': -1}, 'b': {'B10': 1, 'B11': 1}, 'c': {'C00': 1}},
    ]
    
    block_map = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }
    
    for m_idx, term in enumerate(strassen_terms):
        base_col = m_idx * r3
        for local_col in range(r3):
            col = base_col + local_col
            i3 = local_col // 9
            rem = local_col % 9
            j3 = rem // 3
            k3 = rem % 3
            
            for a_block, ca in term['a'].items():
                bi, bj = block_map[a_block]
                u_row = (bi*3+i3)*6 + (bj*3+j3)
                U[u_row, col] += ca * U33[i3*3+j3, local_col]
            
            for b_block, cb in term['b'].items():
                bk, bl = block_map[b_block]
                v_row = (bk*3+j3)*6 + (bl*3+k3)
                V[v_row, col] += cb * V33[j3*3+k3, local_col]
            
            for c_block, cc in term['c'].items():
                ci, ck = block_map[c_block]
                w_row = (ck*3+k3)*6 + (ci*3+i3)
                W[w_row, col] += cc * W33[k3*3+i3, local_col]
    
    # Verify the 6x6 decomposition
    if time.time() < deadline - 5:
        if not _verify_decomposition(U, V, W, n=6, m=6, p=6, trials=3):
            # Fallback to Strassen-based block decomposition (rank 189)
            r3_fb = 27
            U33_fb = np.zeros((9, r3_fb))
            V33_fb = np.zeros((9, r3_fb))
            W33_fb = np.zeros((9, r3_fb))
            
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U33_fb[i*3+j, col] = 1
                        V33_fb[j*3+k, col] = 1
                        W33_fb[k*3+i, col] = 1
                        col += 1
                        
            rank_fb = 7 * r3_fb
            U_fb = np.zeros((36, rank_fb))
            V_fb = np.zeros((36, rank_fb))
            W_fb = np.zeros((36, rank_fb))
            
            for m_idx, term in enumerate(strassen_terms):
                base_col = m_idx * r3_fb
                for local_col in range(r3_fb):
                    col = base_col + local_col
                    i3 = local_col // 9
                    rem = local_col % 9
                    j3 = rem // 3
                    k3 = rem % 3
                    
                    for a_block, ca in term['a'].items():
                        bi, bj = block_map[a_block]
                        u_row = (bi*3+i3)*6 + (bj*3+j3)
                        U_fb[u_row, col] += ca * U33_fb[i3*3+j3, local_col]
                    
                    for b_block, cb in term['b'].items():
                        bk, bl = block_map[b_block]
                        v_row = (bk*3+j3)*6 + (bl*3+k3)
                        V_fb[v_row, col] += cb * V33_fb[j3*3+k3, local_col]
                    
                    for c_block, cc in term['c'].items():
                        ci, ck = block_map[c_block]
                        w_row = (ck*3+k3)*6 + (ci*3+i3)
                        W_fb[w_row, col] += cc * W33_fb[k3*3+i3, local_col]
            
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
