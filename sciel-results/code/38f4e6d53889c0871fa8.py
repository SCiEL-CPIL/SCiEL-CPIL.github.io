import numpy as np

def construct():
    n, m, p = 3, 3, 6
    
    # Fallback: Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    # Try Rank 14 using two Bini 3x3x3 rank-7 decompositions
    # Bini 3x3x3 rank-7 coefficients
    # U_bini (9x7), V_bini (9x7), W_bini (9x7)
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    U_bini = np.zeros((9, 7), dtype=int)
    V_bini = np.zeros((9, 7), dtype=int)
    W_bini = np.zeros((9, 7), dtype=int)
    
    # Product 1
    # M1 = (A11+A22+A33)(B11+B22+B33)
    # A part: A11, A22, A33 -> indices 0, 4, 8
    U_bini[0, 0] += 1
    U_bini[4, 0] += 1
    U_bini[8, 0] += 1
    # B part: B11, B22, B33 -> indices 0, 4, 8
    V_bini[0, 0] += 1
    V_bini[4, 0] += 1
    V_bini[8, 0] += 1
    # W part: C11, C22, C33 get contribution?
    # W index is k*3+i.
    # The product contributes to all C_ik?
    # No, the W matrix distributes the product to the output.
    # For M1, it contributes to C11, C12, C13, C21, ... C33?
    # Actually, W_bini[k*3+i, r] determines how much product r contributes to C_ik.
    # In Strassen/Bini, W is usually identity-like or sparse.
    # For M1 = (sum A)(sum B), it contributes to all C_ik?
    # No, that's not right.
    # The decomposition is C = sum_r (U_r A) (V_r B) W_r^T?
    # No, the form is C_{ik} = sum_j sum_l A_{ij} B_{jl} T_{ijk}.
    # The tensor decomposition is T_{ijk} = sum_r U_{ij}^r V_{jk}^r W_{ki}^r.
    # So W_{ki}^r is the coefficient for C_{ik} from product r.
    # In Bini 3x3x3, the W matrix is also involved in the linear combination.
    
    # I am not confident in the W matrix for Bini 3x3x3.
    # Therefore, I will NOT attempt the rank-14 solution due to high risk of incorrect W matrix.
    # I will return the schoolbook solution.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
