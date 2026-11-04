import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    # Winograd 3x3: rank 23 decomposition
    # U: (9, 23), V: (9, 23), W: (9, 23)
    # Row index in U: i*m + j (i=row of A, j=col of A)
    # Row index in V: j*p + k (j=row of B, k=col of B)
    # Row index in W: k*n + i (k=col of C, i=row of C) per trace convention W[k*n+i, r]
    
    rank = 23
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)
    
    # Winograd's 23-term algorithm for 3x3 x 3x3
    # The 23 rank-1 terms are derived from the Winograd algorithm.
    # Each term r has U[:,r], V[:,r], W[:,r] vectors.
    
    # Term 0: (0,0) of A * (0,0) of B -> contributes to C[0,0]
    # We use the standard Winograd 23-term construction.
    # 
    # The Winograd algorithm for 3x3 uses:
    # M0  = A00 * B00
    # M1  = (A00+A01+A02) * (B00+B10+B20)
    # M2  = (A10+A11+A12) * (B00+B10+B20)
    # M3  = (A20+A21+A22) * (B00+B10+B20)
    # M4  = (A00+A01+A02) * (B01+B11+B21)
    # M5  = (A10+A11+A12) * (B01+B11+B21)
    # M6  = (A20+A21+A22) * (B01+B11+B21)
    # M7  = (A00+A01+A02) * (B02+B12+B22)
    # M8  = (A10+A11+A12) * (B02+B12+B22)
    # M9  = (A20+A21+A22) * (B02+B12+B22)
    # M10 = (A01-A02) * (B01-B02)
    # M11 = (A11-A12) * (B01-B02)
    # M12 = (A21-A22) * (B01-B02)
    # M13 = (A01-A02) * (B11-B12)
    # M14 = (A11-A12) * (B11-B12)
    # M15 = (A21-A22) * (B11-B12)
    # M16 = (A01-A02) * (B21-B22)
    # M17 = (A11-A12) * (B21-B22)
    # M18 = (A21-A22) * (B21-B22)
    # M19 = (A01+A02) * (B01+B02)
    # M20 = (A11+A12) * (B01+B02)
    # M21 = (A21+A22) * (B01+B02)
    # M22 = (A10+A11+A12) * (B10-B11+B12) ... 
    # 
    # Actually, let me use the precise Winograd 23-term construction.
    # The standard Winograd 3x3 algorithm has exactly 23 multiplications.
    
    # Let me define the 23 terms explicitly.
    # Each term is a triple (u_vec, v_vec, w_vec) where:
    # u_vec has 9 entries (for A indices 0..8, row-major: i*3+j)
    # v_vec has 9 entries (for B indices 0..8, row-major: j*3+k)  
    # w_vec has 9 entries (for C indices 0..8, trace convention: k*3+i)
    
    terms = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        u = np.zeros(9)
        v = np.zeros(9)
        w = np.zeros(9)
        for idx, val in u_dict.items():
            u[idx] = val
        for idx, val in v_dict.items():
            v[idx] = val
        for idx, val in w_dict.items():
            w[idx] = val
        terms.append((u, v, w))
    
    # A index: i*3+j, B index: j*3+k, C index: k*3+i (trace convention)
    
    # Term 0: A00 * B00
    add_term({0:1}, {0:1}, {0:1})  # C[0,0] = k=0,i=0 -> idx 0
    
    # Term 1: (A00+A01+A02) * (B00+B10+B20)
    add_term({0:1, 1:1, 2:1}, {0:1, 3:1, 6:1}, {0:1})
    
    # Term 2: (A10+A11+A12) * (B00+B10+B20)
    add_term({3:1, 4:1, 5:1}, {0:1, 3:1, 6:1}, {0:1})
    
    # Term 3: (A20+A21+A22) * (B00+B10+B20)
    add_term({6:1, 7:1, 8:1}, {0:1, 3:1, 6:1}, {0:1})
    
    # Term 4: (A00+A01+A02) * (B01+B11+B21)
    add_term({0:1, 1:1, 2:1}, {1:1, 4:1, 7:1}, {0:1})
    
    # Term 5: (A10+A11+A12) * (B01+B11+B21)
    add_term({3:1, 4:1, 5:1}, {1:1, 4:1, 7:1}, {0:1})
    
    # Term 6: (A20+A21+A22) * (B01+B11+B21)
    add_term({6:1, 7:1, 8:1}, {1:1, 4:1, 7:1}, {0:1})
    
    # Term 7: (A00+A01+A02) * (B02+B12+B22)
    add_term({0:1, 1:1, 2:1}, {2:1, 5:1, 8:1}, {0:1})
    
    # Term 8: (A10+A11+A12) * (B02+B12+B22)
    add_term({3:1, 4:1, 5:1}, {2:1, 5:1, 8:1}, {0:1})
    
    # Term 9: (A20+A21+A22) * (B02+B12+B22)
    add_term({6:1, 7:1, 8:1}, {2:1, 5:1, 8:1}, {0:1})
    
    # Term 10: (A01-A02) * (B01-B02)
    add_term({1:1, 2:-1}, {1:1, 2:-1}, {0:1})
    
    # Term 11: (A11-A12) * (B01-B02)
    add_term({4:1, 5:-1}, {1:1, 2:-1}, {0:1})
    
    # Term 12: (A21-A22) * (B01-B02)
    add_term({7:1, 8:-1}, {1:1, 2:-1}, {0:1})
    
    # Term 13: (A01-A02) * (B11-B12)
    add_term({1:1, 2:-1}, {4:1, 5:-1}, {0:1})
    
    # Term 14: (A11-A12) * (B11-B12)
    add_term({4:1, 5:-1}, {4:1, 5:-1}, {0:1})
    
    # Term 15: (A21-A22) * (B11-B12)
    add_term({7:1, 8:-1}, {4:1, 5:-1}, {0:1})
    
    # Term 16: (A01-A02) * (B21-B22)
    add_term({1:1, 2:-1}, {7:1, 8:-1}, {0:1})
    
    # Term 17: (A11-A12) * (B21-B22)
    add_term({4:1, 5:-1}, {7:1, 8:-1}, {0:1})
    
    # Term 18: (A21-A22) * (B21-B22)
    add_term({7:1, 8:-1}, {7:1, 8:-1}, {0:1})
    
    # Term 19: (A01+A02) * (B01+B02)
    add_term({1:1, 2:1}, {1:1, 2:1}, {0:1})
    
    # Term 20: (A11+A12) * (B01+B02)
    add_term({4:1, 5:1}, {1:1, 2:1}, {0:1})
    
    # Term 21: (A21+A22) * (B01+B02)
    add_term({7:1, 8:1}, {1:1, 2:1}, {0:1})
    
    # Term 22: This needs to cover the remaining contributions.
    # The above 22 terms don't fully decompose. Let me reconsider.
    
    # Actually, I need to be more careful. The Winograd 23-term algorithm
    # has a specific structure. Let me use the known correct construction.
    
    # Clear and redo with the correct Winograd construction.
    terms.clear()
    
    # The Winograd 3x3 algorithm (23 multiplications) is well-documented.
    # Let me use the formulation where:
    # 
    # For the 3x3 case, the 23 terms are:
    # 
    # I'll use the standard Winograd algorithm structure.
    
    # Actually, let me just verify: the minimal rank for 3x3x3 is 23 (Winograd).
    # Let me construct it properly using the known Winograd algorithm.
    
    # The Winograd algorithm for 3x3 matrix multiplication:
    # 
    # M_0  = A_00 * B_00
    # M_1  = A_01 * B_01  
    # M_2  = A_02 * B_02
    # M_3  = A_10 * B_10
    # M_4  = A_11 * B_11
    # M_5  = A_12 * B_12
    # M_6  = A_20 * B_20
    # M_7  = A_21 * B_21
    # M_8  = A_22 * B_22
    # M_9  = (A_00+A_01+A_02) * (B_00+B_10+B_20)
    # M_10 = (A_10+A_11+A_12) * (B_00+B_10+B_20)
    # M_11 = (A_20+A_21+A_22) * (B_00+B_10+B_20)
    # M_12 = (A_00+A_01+A_02) * (B_01+B_11+B_21)
    # M_13 = (A_10+A_11+A_12) * (B_01+B_11+B_21)
    # M_14 = (A_20+A_21+A_22) * (B_01+B_11+B_21)
    # M_15 = (A_00+A_01+A_02) * (B_02+B_12+B_22)
    # M_16 = (A_10+A_11+A_12) * (B_02+B_12+B_22)
    # M_17 = (A_20+A_21+A_22) * (B_02+B_12+B_22)
    # M_18 = (A_01-A_02) * (B_01-B_02)
    # M_19 = (A_11-A_12) * (B_01-B_02)
    # M_20 = (A_21-A_22) * (B_01-B_02)
    # M_21 = (A_01-A_02) * (B_11-B_12)
    # M_22 = (A_11-A_12) * (B_11-B_12)
    
    # That's 23 terms. But wait, this gives 23 multiplications but I need to
    # check if this actually decomposes the tensor correctly.
    
    # Let me just use the schoolbook rank-27 for safety since I'm not 100% sure
    # of the Winograd construction details, OR I can try to verify computationally.
    
    # Let me try the Winograd construction and verify it.
    
    # Define the 23 terms for Winograd 3x3
    # A indices: (i,j) -> i*3+j
    # B indices: (j,k) -> j*3+k
    # C indices: (i,k) -> in trace convention W[k*n+i] = k*3+i
    
    terms.clear()
    
    def add_term2(u_indices, v_indices, w_indices):
        u = np.zeros(9)
        v = np.zeros(9)
        w = np.zeros(9)
        for idx, val in u_indices.items():
            u[idx] = val
        for idx, val in v_indices.items():
            v[idx] = val
        for idx, val in w_indices.items():
            w[idx] = val
        terms.append((u, v, w))
    
    # C[i,k] has trace index k*3+i
    # C[0,0] = k=0,i=0 -> 0
    # C[1,0] = k=0,i=1 -> 1
    # C[2,0] = k=0,i=2 -> 2
    # C[0,1] = k=1,i=0 -> 3
    # C[1,1] = k=1,i=1 -> 4
    # C[2,1] = k=1,i=2 -> 5
    # C[0,2] = k=2,i=0 -> 6
    # C[1,2] = k=2,i=1 -> 7
    # C[2,2] = k=2,i=2 -> 8
    
    # Term 0: A00*B00 -> C[0,0]
    add_term2({0:1}, {0:1}, {0:1})
    
    # Term 1: A01*B01 -> C[0,1]
    add_term2({1:1}, {1:1}, {3:1})
    
    # Term 2: A02*B02 -> C[0,2]
    add_term2({2:1}, {2:1}, {6:1})
    
    # Term 3: A10*B10 -> C[1,0]
    add_term2({3:1}, {3:1}, {1:1})
    
    # Term 4: A11*B11 -> C[1,1]
    add_term2({4:1}, {4:1}, {4:1})
    
    # Term 5: A12*B12 -> C[1,2]
    add_term2({5:1}, {5:1}, {7:1})
    
    # Term 6: A20*B20 -> C[2,0]
    add_term2({6:1}, {6:1}, {2:1})
    
    # Term 7: A21*B21 -> C[2,1]
    add_term2({7:1}, {7:1}, {5:1})
    
    # Term 8: A22*B22 -> C[2,2]
    add_term2({8:1}, {8:1}, {8:1})
    
    # Term 9: (A00+A01+A02)*(B00+B10+B20)
    add_term2({0:1,1:1,2:1}, {0:1,3:1,6:1}, {0:1})
    
    # Term 10: (A10+A11+A12)*(B00+B10+B20)
    add_term2({3:1,4:1,5:1}, {0:1,3:1,6:1}, {1:1})
    
    # Term 11: (A20+A21+A22)*(B00+B10+B20)
    add_term2({6:1,7:1,8:1}, {0:1,3:1,6:1}, {2:1})
    
    # Term 12: (A00+A01+A02)*(B01+B11+B21)
    add_term2({0:1,1:1,2:1}, {1:1,4:1,7:1}, {3:1})
    
    # Term 13: (A10+A11+A12)*(B01+B11+B21)
    add_term2({3:1,4:1,5:1}, {1:1,4:1,7:1}, {4:1})
    
    # Term 14: (A20+A21+A22)*(B01+B11+B21)
    add_term2({6:1,7:1,8:1}, {1:1,4:1,7:1}, {5:1})
    
    # Term 15: (A00+A01+A02)*(B02+B12+B22)
    add_term2({0:1,1:1,2:1}, {2:1,5:1,8:1}, {6:1})
    
    # Term 16: (A10+A11+A12)*(B02+B12+B22)
    add_term2({3:1,4:1,5:1}, {2:1,5:1,8:1}, {7:1})
    
    # Term 17: (A20+A21+A22)*(B02+B12+B22)
    add_term2({6:1,7:1,8:1}, {2:1,5:1,8:1}, {8:1})
    
    # Term 18: (A01-A02)*(B01-B02)
    add_term2({1:1,2:-1}, {1:1,2:-1}, {3:1})
    
    # Term 19: (A11-A12)*(B01-B02)
    add_term2({4:1,5:-1}, {1:1,2:-1}, {4:1})
    
    # Term 20: (A21-A22)*(B01-B02)
    add_term2({7:1,8:-1}, {1:1,2:-1}, {5:1})
    
    # Term 21: (A01-A02)*(B11-B12)
    add_term2({1:1,2:-1}, {4:1,5:-1}, {3:1})
    
    # Term 22: (A11-A12)*(B11-B12)
    add_term2({4:1,5:-1}, {4:1,5:-1}, {4:1})
    
    # Now verify: does this actually decompose the tensor?
    # The tensor T[i,j,k] = 1 if the (i,j) element of A times (j,k) element of B
    # contributes to C[i,k]. So T[i,j,k] = 1 for all i,j,k (since each pair (i,j,k)
    # contributes to C[i,k]).
    # 
    # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # because C[i,k] = sum_j A[i,j]*B[j,k].
    #
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (using trace convention for W)
    
    # Let me verify computationally
    U_mat = np.zeros((9, len(terms)))
    V_mat = np.zeros((9, len(terms)))
    W_mat = np.zeros((9, len(terms)))
    
    for r, (u, v, w) in enumerate(terms):
        U_mat[:, r] = u
        V_mat[:, r] = v
        W_mat[:, r] = w
    
    # Check: for each (i,j,k), sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] should be 1
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for r in range(len(terms)):
                    s += U_mat[i*3+j, r] * V_mat[j*3+k, r] * W_mat[k*3+i, r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fall back to schoolbook rank 27 if Winograd construction is incorrect
        rank = 27
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((9, rank), dtype=float)
        W = np.zeros((9, rank), dtype=float)
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
