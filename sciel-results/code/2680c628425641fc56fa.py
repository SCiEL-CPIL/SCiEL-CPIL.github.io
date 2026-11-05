"""Matrix multiplication 3x5x7 - attempting rank-minimal decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    
    # Kronecker decomposition strategy:
    # T_{n,m,p} = T_{n,1,1} \otimes T_{1,m,p}
    # Rank of T_{n,1,1} is n. Rank of T_{1,m,p} is m*p.
    # Total rank = n * m * p.
    
    # Component 1: T_{n,1,1} (Identity-like for n x 1 x 1)
    # U1 shape (n*1, n), V1 shape (1*1, n), W1 shape (1*n, n)
    # U1[i, i] = 1
    # V1[0, i] = 1
    # W1[i, i] = 1
    
    # Component 2: T_{1,m,p} (Matrix mult 1xm by mxp)
    # U2 shape (1*m, m*p), V2 shape (m*p, m*p), W2 shape (p*1, m*p)
    # This is the standard schoolbook for 1 x m x p.
    # U2[j, j*p+k] = 1
    # V2[j*p+k, j*p+k] = 1
    # W2[k, j*p+k] = 1
    
    rank1 = n
    rank2 = m * p
    total_rank = rank1 * rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Construct Kronecker product of the two decompositions
    # Index mapping:
    # U index: i (from 0..n-1), j (from 0..m-1) -> i*m + j
    # V index: j (from 0..m-1), k (from 0..p-1) -> j*p + k
    # W index: k (from 0..p-1), i (from 0..n-1) -> k*n + i
    
    # Rank index: r1 (from 0..n-1), r2 (from 0..m*p-1) -> r1 * rank2 + r2
    
    for r1 in range(rank1):
        for r2 in range(rank2):
            col = r1 * rank2 + r2
            
            # Determine which (i,j,k) this rank term corresponds to
            # In T_{n,1,1}, term r1 corresponds to i=r1, j=0, k=0.
            # In T_{1,m,p}, term r2 corresponds to j=r2//p, k=r2%p.
            
            i = r1
            j = r2 // p
            k = r2 % p
            
            # Check if this term is active in the Kronecker product
            # T_{n,1,1} has non-zero at U1[i, r1] if i==r1.
            # T_{1,m,p} has non-zero at U2[j, r2] if j == r2//p (always true for this structure)
            
            # The Kronecker product U is U1[i, r1] * U2[j, r2]
            # U1[i, r1] is 1 if i == r1, else 0.
            # U2[j, r2] is 1 if j == r2//p, else 0.
            
            if i == r1 and j == r2 // p:
                U[i * m + j][col] = 1
            
            # V1[0, r1] is 1 if 0 == 0 (always true for V1 part? No, V1 shape is 1x1 x n. V1[0, r1]=1).
            # V2[j*p+k, r2] is 1 if j*p+k == r2.
            # So V index is j*p+k.
            # V1 part index is 0.
            # Combined V index is (0, j*p+k) -> j*p+k.
            
            if 0 == 0 and j * p + k == r2:
                V[j * p + k][col] = 1
            
            # W1[i, r1] is 1 if i == r1.
            # W2[k, r2] is 1 if k == r2%p.
            # Combined W index is (k, i) -> k*n + i.
            
            if i == r1 and k == r2 % p:
                W[k * n + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
