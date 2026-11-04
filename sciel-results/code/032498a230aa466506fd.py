"""Rewritten program: M_{2,3,10} rank-55 decomposition via Strassen on 2x2 sub-blocks."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    U_rows = n * m   # 6
    V_rows = m * p   # 30
    W_rows = p * n   # 20
    
    # Per pair of k-values, we use 7 (Strassen) + 4 (schoolbook for j=2) = 11 products
    per_pair = 11
    num_pairs = p // 2  # 5
    total_rank = per_pair * num_pairs  # 55
    
    U_mat = [[0] * total_rank for _ in range(U_rows)]
    V_mat = [[0] * total_rank for _ in range(V_rows)]
    W_mat = [[0] * total_rank for _ in range(W_rows)]
    
    # Strassen formulas for 2x2 x 2x2:
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # Result: [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # For each pair (k1, k2), we multiply:
    # A_pair = [[A[0,j1], A[0,j2]], [A[1,j1], A[1,j2]]]  (2x2, j1=0, j2=1)
    # B_pair = [[B[j1,k1], B[j1,k2]], [B[j2,k1], B[j2,k2]]]  (2x2)
    # C_pair = A_pair @ B_pair (2x2)
    # Plus schoolbook for j=2: C[i,k1] += A[i,2]*B[2,k1], C[i,k2] += A[i,2]*B[2,k2]
    
    for pair_idx in range(num_pairs):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base = pair_idx * per_pair
        
        # Strassen products (indices 0-6 in local)
        # m1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k2])
        r = base + 0
        U_mat[0*3+0][r] += 1  # A[0,0]
        U_mat[1*3+1][r] += 1  # A[1,1]
        V_mat[0*10+k1][r] += 1  # B[0,k1]
        V_mat[1*10+k2][r] += 1  # B[1,k2]
        W_mat[k1*2+0][r] += 1  # C[0,k1]
        W_mat[k2*2+1][r] += 1  # C[1,k2]
        
        # m2 = (A[1,0]+A[1,1]) * B[0,k1]
        r = base + 1
        U_mat[1*3+0][r] += 1  # A[1,0]
        U_mat[1*3+1][r] += 1  # A[1,1]
        V_mat[0*10+k1][r] += 1  # B[0,k1]
        W_mat[k1*2+1][r] += 1  # C[1,k1]
        
        # m3 = A[0,0] * (B[0,k2]-B[1,k1])
        r = base + 2
        U_mat[0*3+0][r] += 1  # A[0,0]
        V_mat[0*10+k2][r] += 1  # B[0,k2]
        V_mat[1*10+k1][r] -= 1  # -B[1,k1]
        W_mat[k1*2+0][r] += 1  # C[0,k1]
        W_mat[k2*2+0][r] += 1  # C[0,k2]
        
        # m4 = A[1,1] * (B[1,k1]-B[0,k1])
        r = base + 3
        U_mat[1*3+1][r] += 1  # A[1,1]
        V_mat[1*10+k1][r] += 1  # B[1,k1]
        V_mat[0*10+k1][r] -= 1  # -B[0,k1]
        W_mat[k1*2+0][r] += 1  # C[0,k1]
        W_mat[k1*2+1][r] += 1  # C[1,k1]
        
        # m5 = (A[0,0]+A[0,1]) * B[1,k2]
        r = base + 4
        U_mat[0*3+0][r] += 1  # A[0,0]
        U_mat[0*3+1][r] += 1  # A[0,1]
        V_mat[1*10+k2][r] += 1  # B[1,k2]
        W_mat[k1*2+0][r] -= 1  # -C[0,k1]
        W_mat[k2*2+0][r] += 1  # C[0,k2]
        
        # m6 = (A[1,0]-A[0,0]) * (B[0,k1]+B[0,k2])
        r = base + 5
        U_mat[1*3+0][r] += 1  # A[1,0]
        U_mat[0*3+0][r] -= 1  # -A[0,0]
        V_mat[0*10+k1][r] += 1  # B[0,k1]
        V_mat[0*10+k2][r] += 1  # B[0,k2]
        W_mat[k2*2+1][r] += 1  # C[1,k2]
        
        # m7 = (A[0,1]-A[1,1]) * (B[1,k1]+B[1,k2])
        r = base + 6
        U_mat[0*3+1][r] += 1  # A[0,1]
        U_mat[1*3+1][r] -= 1  # -A[1,1]
        V_mat[1*10+k1][r] += 1  # B[1,k1]
        V_mat[1*10+k2][r] += 1  # B[1,k2]
        W_mat[k1*2+0][r] += 1  # C[0,k1]
        
        # Schoolbook for j=2 (indices 7-10 in local)
        # s1 = A[0,2] * B[2,k1] -> C[0,k1]
        r = base + 7
        U_mat[0*3+2][r] += 1  # A[0,2]
        V_mat[2*10+k1][r] += 1  # B[2,k1]
        W_mat[k1*2+0][r] += 1  # C[0,k1]
        
        # s2 = A[0,2] * B[2,k2] -> C[0,k2]
        r = base + 8
        U_mat[0*3+2][r] += 1  # A[0,2]
        V_mat[2*10+k2][r] += 1  # B[2,k2]
        W_mat[k2*2+0][r] += 1  # C[0,k2]
        
        # s3 = A[1,2] * B[2,k1] -> C[1,k1]
        r = base + 9
        U_mat[1*3+2][r] += 1  # A[1,2]
        V_mat[2*10+k1][r] += 1  # B[2,k1]
        W_mat[k1*2+1][r] += 1  # C[1,k1]
        
        # s4 = A[1,2] * B[2,k2] -> C[1,k2]
        r = base + 10
        U_mat[1*3+2][r] += 1  # A[1,2]
        V_mat[2*10+k2][r] += 1  # B[2,k2]
        W_mat[k2*2+1][r] += 1  # C[1,k2]
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
