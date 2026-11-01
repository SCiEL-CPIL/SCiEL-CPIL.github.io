import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Uses Strassen's rank-7 algorithm for each of the 7 blocks of 2x2 columns
    in the B matrix. Total rank = 7 * 7 = 49.
    
    The multiplication tensor T_{(i,j),(j,k),(i,k)} = 1 is decomposed as:
    T = sum_s U_s ⊗ V_s ⊗ W_s
    
    where:
    - U has shape (n*m, rank) = (4, 49)
    - V has shape (m*p, rank) = (28, 49)  
    - W has shape (p*n, rank) = (28, 49)
    
    Index conventions:
    - U[i*m + j, col] for A[i,j]
    - V[j*p + k, col] for B[j,k]
    - W[k*n + i, col] for C[i,k] (following trace convention W[k*n+i, r])
    """
    t_start = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Number of 2x2 blocks in the 2x14 matrix
    num_blocks = p // 2  # 7
    
    # Strassen's 7-term decomposition for 2x2 * 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen terms: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict {A_idx: coeff} where A_idx = i*m+j
    # b_coeffs: dict {B_idx: coeff} where B_idx = j*2+b_col (local to block)
    # c_coeffs: dict {C_idx: coeff} where C_idx = i*2+c_col (local to block)
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10 + A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00 + A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    rank = num_blocks * 7  # 49
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for b in range(num_blocks):
        for s in range(7):
            col = b * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            # U: U[i*m + j, col] = coefficient of A[i,j]
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            # V: V[j*p + k, col] = coefficient of B[j,k]
            # Block b covers columns 2*b and 2*b+1 of B
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * b + b_col
                V[b_j * p + b_k, col] = coeff
            
            # W: W[k*n + i, col] = coefficient of C[i,k]
            # Block b covers columns 2*b and 2*b+1 of C
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * b + c_col
                W[c_k * n + c_i, col] = coeff
    
    # Verify correctness with a quick check
    def verify_decomposition(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # T_{(i,j),(j,k),(i,k)} should equal 1
                    # = sum_s U[u_idx,s] * V[v_idx,s] * W[w_idx,s]
                    val = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook rank-56 decomposition
        rank_fb = n * m * p  # 56
        U = np.zeros((nm, rank_fb))
        V = np.zeros((mp, rank_fb))
        W = np.zeros((pn, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    # Convert to lists with integer entries where possible
    def to_list_with_ints(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
    
    U_list = to_list_with_ints(U)
    V_list = to_list_with_ints(V)
    W_list = to_list_with_ints(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
