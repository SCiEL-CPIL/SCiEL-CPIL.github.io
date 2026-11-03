"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-18 exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy: View the 3x9 matrix B as three 3x3 blocks. Use a rank-6 decomposition
    for 2x3x3 matrix multiplication, applied to each of the 3 blocks, giving rank 18.
    
    For 2x3x3: A is 2x3, B is 3x3, C is 2x3.
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor T[a, b, c] where a = i*3+j, b = j*3+k, c = k*2+i.
    
    Rank-6 decomposition for 2x3x3:
    For each (j, k) in {0,1,2} x {0,1,2}, create one term.
    U[i*3+j, r] = 1 for i in {0,1}
    V[j*3+k, r] = 1
    W[k*2+i, r] = 1 for i in {0,1}
    
    Wait, this has the same issue as before. Let me verify for 2x3x3:
    a = i_a*3+j_a, b = j_b*3+k_b, c = k_c*2+i_c
    Term r=(j,k): U[a,r]=1 iff j_a=j, V[b,r]=1 iff j_b=j and k_b=k, W[c,r]=1 iff k_c=k and i_c=i
    So product = [j_a=j_b] * [k_b=k_c] * [i_a=i_c]... but we need all three to match simultaneously.
    
    Actually for the rank-6: U[a,r]=1 iff j_a=j(r), V[b,r]=1 iff (j_b,k_b)=(j(r),k(r)), W[c,r]=1 iff (k_c,i_c)=(k(r),i) where i is free.
    
    So sum over r: [j_a=j(r)]*[j_b=j(r) and k_b=k(r)]*[k_c=k(r) and i_c=i for some i]
    = [j_a=j_b] * [k_b=k_c] * [i_a exists such that i_c=i_a]
    
    Since i ranges over {0,1} freely in U and W, the condition is just j_a=j_b and k_b=k_c.
    But we also need i_a=i_c. This is missing again!
    
    So rank-6 doesn't work for 2x3x3 either. The schoolbook for 2x3x3 is rank 18.
    
    Let me just use schoolbook rank 54 for 2x3x9.
    """
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
# EVOLVE-BLOCK-END
