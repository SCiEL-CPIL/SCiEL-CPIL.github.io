"""
Program for rank-minimal exact tensor decomposition of 2x4 * 4x8 matrix multiplication.

Approach:
- Decompose the multiplication into independent sub-problems based on the intermediate index j.
- Split j (size 4) into two groups of 2: jg in {0, 1}.
- For each j-group, we have a 2x2 * 2x8 multiplication.
- Split the output index k (size 8) into four groups of 2: kg in {0, 1, 2, 3}.
- For each (jg, kg) pair, we have a 2x2 * 2x2 multiplication.
- Use Strassen's rank-7 algorithm for each 2x2 * 2x2 sub-problem.
- Total rank = 2 (j-groups) * 4 (k-groups) * 7 (Strassen rank) = 56.

The factor matrices U, V, W are constructed such that:
  C[i, k] = sum_j A[i, j] * B[j, k]
is exactly reproduced via the tensor decomposition:
  C[i, k] = sum_r U[i, m+j, r] * V[m+j, p+k, r] * W[p+k, n+i, r] 
  (Note: The interface expects U, V, W to be flattened or indexed appropriately. 
   The standard tensor decomposition for matrix multiplication C = AB is:
   C_{ik} = sum_{j,r} A_{ij} B_{jk} \delta_{...} 
   The factor matrices provided in the prompt are:
   U: shape (n*m, rank) -> corresponds to A indices (i, j)
   V: shape (m*p, rank) -> corresponds to B indices (j, k)
   W: shape (p*n, rank) -> corresponds to C indices (k, i) with trace convention W[k*n+i, r]
   
   The decomposition is:
   C_{ik} = sum_{j, r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
   This is not quite right. The standard rank decomposition of the multiplication tensor T_{ijk} is:
   T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}
   Here U has indices (i, j), V has (j, k), W has (k, i).
   The prompt specifies:
   U: (n*m, rank) -> index i*m+j
   V: (m*p, rank) -> index j*p+k
   W: (p*n, rank) -> index k*n+i
   
   So the reconstruction is:
   C_{ik} = sum_{j, r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
   Wait, if we sum over j and r, the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
   This matches the tensor structure.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen's 7 intermediate products for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Define Strassen coefficients
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    strassen_A = [
        {0: 1, 3: 1},   # M1: a+d
        {2: 1, 3: 1},   # M2: c+d
        {0: 1},         # M3: a
        {3: 1},         # M4: d
        {0: 1, 1: 1},   # M5: a+b
        {2: 1, 0: -1},  # M6: c-a
        {1: 1, 3: -1},  # M7: b-d
    ]
    strassen_B = [
        {0: 1, 3: 1},   # M1: e+h
        {0: 1},         # M2: e
        {1: 1, 3: -1},  # M3: f-h
        {2: 1, 0: -1},  # M4: g-e
        {3: 1},         # M5: h
        {0: 1, 1: 1},   # M6: e+f
        {2: 1, 3: 1},   # M7: g+h
    ]
    strassen_C = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C11: M1+M4-M5+M7
        {2: 1, 4: 1},                # C12: M3+M5
        {1: 1, 3: 1},                # C21: M2+M4
        {0: 1, 1: -1, 2: 1, 5: 1},  # C22: M1-M2+M3+M6
    ]
    
    # Total rank
    # 2 j-groups * 4 k-groups * 7 Strassen terms
    rank = 2 * 4 * 7
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # n*m = 8
    V = [[0] * rank for _ in range(m * p)]  # m*p = 32
    W = [[0] * rank for _ in range(p * n)]  # p*n = 16
    
    # Iterate over j-groups, k-groups, and Strassen terms
    for jg in range(2):
        for kg in range(4):
            for s in range(7):
                c = (jg * 4 + kg) * 7 + s
                
                # Fill U: A[i, jg*2 + j_local]
                # Index in U: i * m + (jg*2 + j_local)
                for i in range(2):
                    for j_local in range(2):
                        a_idx = i * 2 + j_local
                        if a_idx in strassen_A[s]:
                            U[i * m + jg * 2 + j_local][c] += strassen_A[s][a_idx]
                
                # Fill V: B[jg*2 + j_local, kg*2 + k_local]
                # Index in V: (jg*2 + j_local) * p + (kg*2 + k_local)
                for j_local in range(2):
                    for k_local in range(2):
                        b_idx = j_local * 2 + k_local
                        if b_idx in strassen_B[s]:
                            V[(jg * 2 + j_local) * p + kg * 2 + k_local][c] += strassen_B[s][b_idx]
                
                # Fill W: C[i, kg*2 + k_local]
                # Index in W: (kg*2 + k_local) * n + i
                for i in range(2):
                    for k_local in range(2):
                        c_idx = i * 2 + k_local
                        if s in strassen_C[c_idx]:
                            W[(kg * 2 + k_local) * n + i][c] += strassen_C[c_idx][s]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
