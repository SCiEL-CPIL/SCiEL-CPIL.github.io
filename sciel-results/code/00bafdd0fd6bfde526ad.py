import numpy as np

def construct():
    n, m, p = 3, 5, 6
    rank = n * m + m * p - m
    # rank = 15 + 30 - 5 = 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The decomposition:
    # C_ik = sum_j A_ij B_jk
    # = sum_{j=0}^{m-2} A_ij B_jk + A_{i, m-1} B_{m-1, k}
    # We can write A_ij B_jk = (A_ij + A_{i, m-1}) B_jk - A_{i, m-1} B_jk
    # Summing over j:
    # C_ik = sum_{j=0}^{m-2} (A_ij + A_{i, m-1}) B_jk - sum_{j=0}^{m-2} A_{i, m-1} B_jk + A_{i, m-1} B_{m-1, k}
    # C_ik = sum_{j=0}^{m-2} (A_ij + A_{i, m-1}) B_jk + A_{i, m-1} (B_{m-1, k} - sum_{j=0}^{m-2} B_jk)
    
    # This uses:
    # 1. Terms for (A_ij + A_{i, m-1}) B_jk for j=0..m-2, i=0..n-1, k=0..p-1?
    #    No, (A_ij + A_{i, m-1}) is a single factor.
    #    We need to represent this in U, V, W.
    #    U_{ij, c} corresponds to A_ij.
    #    We can't sum A_ij and A_{i, m-1} in U easily without extra terms.
    
    # Alternative construction:
    # Rank nm + mp - m
    # For each i, j (j < m-1), we have a term.
    # For each j, k, we have a term.
    
    # Let's just use the schoolbook if 40 is too complex to code correctly in time.
    # But 40 is a significant improvement.
    
    # I will implement the simple rank 1 check first? No, rank 1 is wrong.
    # I will implement rank 40.
    
    # Construction details:
    # Columns 0 to n*(m-1)-1:
    #   For i in 0..n-1, j in 0..m-2:
    #     c = i*(m-1) + j
    #     U[i*m+j, c] = 1
    #     U[i*m+(m-1), c] = 1  # This adds A_{i, m-1}
    #     V[j*p+k, c] = 1 for all k? No, V is per k.
    #     This term is (A_ij + A_{i, m-1}) * B_jk.
    #     So V[j*p+k, c] = 1 for all k?
    #     If V[j*p+k, c] = 1 for all k, then the term is (A_ij + A_{i, m-1}) * sum_k B_jk W_{ki}.
    #     We need W_{ki, c} to pick out B_jk.
    #     This doesn't work because W is shared.
    
    # Correct Construction for Rank nm + mp - m:
    # Terms 1: For each i, j (j < m-1), k:
    #   This is n*(m-1)*p terms. Too many.
    
    # Let's use the known result that rank <= nm + mp - m.
    # I will hardcode a valid rank 40 decomposition if I can find one, 
    # otherwise I will return the schoolbook (90) to be safe.
    # Given the risk, I will return the schoolbook.
    
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
