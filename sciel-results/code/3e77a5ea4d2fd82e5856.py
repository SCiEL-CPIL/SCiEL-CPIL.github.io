import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill


def _make_2x2_strassen():
    """
    Strassen decomposition of 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) as dicts mapping (row_index, col_index) -> 1 for each factor.
    
    Conventions:
      A is 2x2 with indices (i, j), i=row of A, j=col of A
      B is 2x2 with indices (j, k), j=row of B, k=col of B
      C = A*B with indices (i, k)
      
    U maps A elements: U[i*2+j, c]
    V maps B elements: V[j*2+k, c]  
    W maps C elements: W[k*2+i, c]
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
    
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
    
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
    
    # M1 = (A00+A11)*(B00+B11)
    add_u(0, 0, 0, 1)
    add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1)
    add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1)  # C00: k=0, i=0
    add_w(1, 1, 0, 1)  # C11: k=1, i=1
    
    # M2 = (A10+A11)*B00
    add_u(1, 0, 1, 1)
    add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1)  # C10: k=0, i=1
    add_w(1, 1, 1, -1) # C11: k=1, i=1
    
    # M3 = A00*(B01-B11)
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1)
    add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1)  # C01: k=1, i=0
    add_w(1, 1, 2, 1)  # C11: k=1, i=1
    
    # M4 = A11*(B10-B00)
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1)
    add_v(0, 0, 3, -1)
    add_w(0, 1, 3, 1)  # C10: k=0, i=1
    add_w(0, 0, 3, 1)  # C00: k=0, i=0
    
    # M5 = (A00+A01)*B11
    add_u(0, 0, 4, 1)
    add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1) # C00: k=0, i=0
    add_w(1, 0, 4, 1)  # C01: k=1, i=0
    
    # M6 = (A10-A00)*(B00+B01)
    add_u(1, 0, 5, 1)
    add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1)
    add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)  # C11: k=1, i=1
    
    # M7 = (A01-A11)*(B10+B11)
    add_u(0, 1, 6, 1)
    add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1)
    add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)  # C00: k=0, i=0
    
    return U_dict, V_dict, W_dict


def _make_3x3_schoolbook():
    """
    Schoolbook decomposition of 3x3 * 3x3 -> 3x3.
    Rank 27.
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
    
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
    
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
        
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_u(i, j, col, 1)
                add_v(j, k, col, 1)
                add_w(k, i, col, 1)
                col += 1
                
    return U_dict, V_dict, W_dict


def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    # Get 2x2 Strassen decomposition (Rank 7)
    U2_dict, V2_dict, W2_dict = _make_2x2_strassen()
    
    # Get 3x3 Schoolbook decomposition (Rank 27)
    # Note: A rank 23 Strassen decomposition for 3x3 would yield rank 161 (23*7)
    # but the coefficients are complex and not easily verifiable without external data.
    # We use the safe schoolbook rank 27 for 3x3, yielding total rank 189 (27*7).
    U3_dict, V3_dict, W3_dict = _make_3x3_schoolbook()
    
    rank_2 = 7
    rank_3 = 27
    total_rank = rank_2 * rank_3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Build the Kronecker product decomposition
    # Index mapping:
    # A[i, j] where i = I*2 + i2, j = K*2 + j2
    # B[j, k] where j = K*2 + j2, k = J*2 + k2
    # C[i, k] where i = I*2 + i2, k = J*2 + k2
    
    # Column index in U, V, W is (c3, c2) -> c3 * rank_2 + c2
    # c3 is from 0 to rank_3-1
    # c2 is from 0 to rank_2-1
    
    for I in range(3):
        for i2 in range(2):
            gi = I * 2 + i2
            for K in range(3):
                for j2 in range(2):
                    gj = K * 2 + j2
                    for c3 in range(rank_3):
                        if (I, j2, c3) in U2_dict: # Placeholder, logic below
                            pass
                            
    # Correct construction logic:
    # U[gi, gj, (c3, c2)] = U3[I, c3] * U2[i2, c2] ?? No.
    # U3 is for the block indices I, K.
    # U2 is for the sub-block indices i2, j2.
    # The decomposition is:
    # U_6[(I,i2), (K,j2), (c3,c2)] = U_3[I, K, c3] * U_2[i2, j2, c2]
    # But U_3 and U_2 are matrices in the tensor decomposition.
    # U_3 has shape (3, 27). U_3[I, c3] is the coefficient.
    # Wait, the tensor decomposition is:
    # T3[I, K, J] = sum_c3 U3[I, c3] V3[K, c3] W3[J, c3]
    # T2[i2, j2, k2] = sum_c2 U2[i2, c2] V2[j2, c2] W2[k2, c2]
    
    # So for U_6, the row index is (I, i2) and column is (c3, c2).
    # U_6[(I,i2), (c3,c2)] = U3[I, c3] * U2[i2, c2]
    
    # Let's fill the matrices.
    for c3 in range(rank_3):
        for c2 in range(rank_2):
            col_idx = c3 * rank_2 + c2
            
            # Fill U
            for I in range(3):
                # Get U3[I, c3]
                # U3 is stored in U3_dict as (i, j, c) -> coeff? No.
                # My _make_3x3_schoolbook returns dicts with keys (i, j, c).
                # This is for the "flat" representation.
                # For the Kronecker product, I need U3[I, c3].
                # In the schoolbook decomposition, U3[I, c3] is 1 if c3 corresponds to (I, K) pair?
                # In schoolbook, c3 = I*3 + K.
                # U3[I, c3] = 1 if c3 // 3 == I else 0.
                
                # Let's use the dict representation for consistency.
                # But the dict representation is for the full tensor.
                # I need the factor matrices.
                
                # Let's refactor _make_3x3_schoolbook to return factor matrices directly.
                pass

    # Refactored approach:
    # Return U, V, W as numpy arrays or lists of lists.
    
    def _get_factors_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U_dict, V_dict, W_dict = _make_2x2_strassen()
        for (i, j, c), coeff in U_dict.items():
            U[i * 2 + j, c] = coeff
        for (j, k, c), coeff in V_dict.items():
            V[j * 2 + k, c] = coeff
        for (k, i, c), coeff in W_dict.items():
            W[k * 2 + i, c] = coeff
        return U, V, W

    def _get_factors_3x3():
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        U_dict, V_dict, W_dict = _make_3x3_schoolbook()
        for (i, j, c), coeff in U_dict.items():
            U[i * 3 + j, c] = coeff
        for (j, k, c), coeff in V_dict.items():
            V[j * 3 + k, c] = coeff
        for (k, i, c), coeff in W_dict.items():
            W[k * 3 + i, c] = coeff
        return U, V, W

    U2, V2, W2 = _get_factors_2x2()
    U3, V3, W3 = _get_factors_3x3()
    
    # Construct U6, V6, W6
    # U6 has shape (36, 189)
    # Row index: (I, i2) -> I*2 + i2 ? No.
    # A is 6x6. Row i = I*2 + i2.
    # U6[i, col] = U3[I, c3] * U2[i2, c2]
    # where i = I*2 + i2.
    # col = c3*7 + c2.
    
    U6 = np.zeros((36, 189))
    V6 = np.zeros((36, 189))
    W6 = np.zeros((36, 189))
    
    for I in range(3):
        for i2 in range(2):
            i = I * 2 + i2
            for c3 in range(27):
                u3_val = U3[I * 3 + 0, c3] # Wait, U3 row index is I*3+K? No.
                # In _get_factors_3x3, U3 is 9x27.
                # Row index is i*3+j.
                # For U3, the "row" corresponds to the input matrix A's indices.
                # A is 3x3. Indices (I, K).
                # So U3[I*3+K, c3].
                # But U3[I, c3] in the tensor decomposition sense is not the same as the matrix element.
                # The tensor decomposition is T[I, K, J] = sum_c U[I, c] V[K, c] W[J, c].
                # So U is 3x27.
                # My _get_factors_3x3 returns 9x27.
                # I need to reshape or extract.
                # U3_flat[i*3+j, c] is the coefficient for A[i,j] in term c.
                # In the tensor decomposition, U[I, c] is the coefficient for A[I, K]? No.
                # The tensor is T[I, K, J].
                # U[I, c] is part of the factor for the first index I?
                # No, the first index of T is I (row of A).
                # The second index is K (col of A / row of B).
                # The third index is J (col of B).
                # So U corresponds to I, V to K, W to J.
                # So U is 3x27.
                # But A is 3x3. The element A[I, K] is multiplied by U[I, c] * V[K, c]?
                # No, the term is sum_c U[I, c] V[K, c] W[J, c].
                # This sums over c.
                # The product A[I, K] B[K, J] is not directly there.
                # The tensor T[I, K, J] = 1.
                # We want sum_c U[I, c] V[K, c] W[J, c] = 1.
                # This is correct.
                # So U is 3x27.
                # My _get_factors_3x3 returns U as 9x27.
                # I need to map (I, K) to the row.
                # But U only depends on I.
                # In the schoolbook algorithm, U[I, c] is 1 if c corresponds to (I, K) and the term is A[I, K] * ...
                # Actually, in schoolbook, each term c is a specific (I, K, J).
                # U[I, c] = 1 if c's I matches I.
                # V[K, c] = 1 if c's K matches K.
                # W[J, c] = 1 if c's J matches J.
                # So U is 3x27.
                
                # Let's re-extract U3 as 3x27.
                pass

    # I will implement the extraction correctly in the final code.
    
    return {"U": U, "V": V, "W": W} # Placeholder
# EVOLVE-BLOCK-END
