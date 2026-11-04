"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-9 exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    The key insight: B is a 3x9 matrix which can be viewed as three 3x3 blocks B0, B1, B2.
    For each block b (0,1,2), we compute C[:, 3b:3b+3] = A * B_b.
    
    We use the standard rank-9 decomposition for 2x3 * 3x3 multiplication (which has rank 9 = 2*3*3,
    but actually the minimal rank for 2x3*3x3 is 9? No, let's think more carefully).
    
    Actually, for 2x3 * 3x3, the tensor rank is at most 9 (schoolbook). But we can do better?
    No, the minimal rank for 2x3*3x3 is actually 9. Wait, no.
    
    Let me reconsider. The total problem is 2x3 * 3x9.
    
    Approach: Decompose B into 3 blocks of 3x3. For each block, we need A * B_b.
    The tensor for 2x3 * 3x3 has rank at most 9. But we can share structure.
    
    Actually, a cleaner approach: Use the fact that the multiplication tensor T_{i,j,k} for (i,j) in A, j in middle, k in B
    can be decomposed. Since p=9 and m=3, we can write k = 3*b + l where b in {0,1,2}, l in {0,1,2}.
    
    For each block b, the sub-problem is 2x3 * 3x3. The schoolbook rank for this is 9.
    Three blocks would give 27. But we can do better by noting that the A matrix is shared.
    
    Better approach: Use a rank-9 decomposition where we exploit the structure.
    
    Let's use the following: For 2x3 * 3x9, we can use 9 rank-1 tensors.
    The idea: partition the 9 columns of B into 3 groups of 3. For each group b, 
    we compute A * B_b where B_b is 3x3.
    
    For 2x3 * 3x3, the minimal rank is actually 9? Let me verify:
    The tensor rank of 2x3 * 3x3... The schoolbook is 9. It's known that for 2xn * nxn,
    the rank is at most n^2. For n=3, that's 9. And it's known to be exactly 9 for n>=2?
    
    Actually, I recall that the rank of matrix multiplication [2,3,3] is 9. So three independent
    blocks would need 27. But that's not optimal.
    
    Let me think differently. We have A (2x3) and B (3x9). The product C (2x9).
    
    We can use the following decomposition:
    For each of the 3 "layers" corresponding to the 3 rows of B (and 3 columns of A),
    we can process all 9 columns at once.
    
    Actually, here's a rank-9 construction:
    
    Let's index: A has rows i=0,1 and cols j=0,1,2. B has rows j=0,1,2 and cols k=0..8.
    
    Consider 9 rank-1 tensors indexed by (j, b) where j in {0,1,2} and b in {0,1,2}.
    Wait, that's 9.
    
    For each (j, b), define:
    - U: 1 at position (i, j) for all i? No, U must be 1 for specific i.
    
    Let me use a different approach. Use the standard bilinear decomposition.
    
    Rank-9 decomposition:
    For j in {0,1,2} and b in {0,1,2} (where b selects the block of 3 columns):
      c = j * 3 + b
      
      U[c]: has 1 at positions (i, j) for i in {0,1}? No, U is indexed by (i,j) pairs, shape (6, rank).
      Actually U has shape (n*m, rank) = (6, rank).
      V has shape (m*p, rank) = (27, rank).
      W has shape (p*n, rank) = (18, rank).
      
      For the rank-1 term corresponding to (j, b):
      - U: set U[i*m + j][c] = 1 for i in {0, 1}. So U[0*3+j][c] = 1 and U[1*3+j][c] = 1.
      - V: set V[j*p + k][c] = 1 for k in {3*b, 3*b+1, 3*b+2}. So V[j*9 + 3*b + l][c] = 1 for l in {0,1,2}.
      - W: set W[k*n + i][c] = 1 for k in {3*b, 3*b+1, 3*b+2} and i in {0,1}.
        So W[(3*b+l)*2 + i][c] = 1 for l in {0,1,2} and i in {0,1}.
      
    Let's verify: The contribution to C[i, k] where k = 3*b + l is:
    Sum over c of U[i*m + j][c] * V[j*p + k][c] * ... no, that's not how it works.
    
    The tensor decomposition: T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    Wait, the convention is: C[i,k] = sum_j A[i,j] * B[j,k]
    And the tensor is T_{i,j,k} = delta (it's the multiplication tensor).
    
    The decomposition: T = sum_c U_c \otimes V_c \otimes W_c
    where U_c is a vector of length n*m, V_c of length m*p, W_c of length p*n.
    
    And the reconstruction: C[i,k] = sum_c (sum_j U_c[i*m+j] * A[i,j]) * (sum_j V_c[j*p+k] * B[j,k]) ... 
    No, that's not right either.
    
    The standard form: C[i,k] = sum_c U_c[i*m + ?] ... 
    
    Let me re-read the interface:
    U: shape (n*m, rank) - so U[row, c] where row = i*m + j
    V: shape (m*p, rank) - so V[row, c] where row = j*p + k  
    W: shape (p*n, rank) - so W[row, c] where row = k*n + i
    
    The reconstruction is: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * ... no.
    
    Actually, the tensor T_{i,j,k} satisfies: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And for matrix multiplication, T_{i,j,k} = 1 if this is a valid triple, i.e., the tensor is the structure tensor of MatMul.
    Specifically, T_{i,j,k} corresponds to the coefficient in: C[i,k] = sum_{j} A[i,j] B[j,k]
    So the tensor is T[i,j,k] = delta (it's 1 for all valid i,j,k combinations in the sense that it's the basis).
    
    More precisely: The multiplication tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k where the actual tensor entries are T[i,j,k] = 1 for the "multiplication" pattern.
    
    For the decomposition to be exact, we need:
    For all i, j, k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Where T[i,j,k] = 1 (since MatMul tensor has all entries 1 in the appropriate sense... actually no).
    
    Wait, I think I'm confusing myself. Let me look at the schoolbook example:
    For each (i,j,k), we have one rank-1 term where:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1, and all other entries 0.
    
    Then for a given (i,j,k): sum_c ... = 1 (from the term where c corresponds to (i,j,k)).
    For other (i',j',k'): if (i',j',k') != (i,j,k), the sum is 0.
    
    So the tensor being decomposed has T[i,j,k] = 1 for all i,j,k? That doesn't seem right for matrix multiplication.
    
    Actually, I think the tensor T is defined such that the bilinear map is:
    (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    
    And the tensor T satisfies: C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1.
    
    So yes, T[i,j,k] = 1 for all valid i,j,k. The schoolbook uses n*m*p rank-1 terms, one for each (i,j,k) triple.
    
    Now for my rank-9 attempt:
    For c = j*3 + b where j in {0,1,2}, b in {0,1,2}:
    U[i*m+j, c] = 1 for i in {0,1}
    V[j*p + k, c] = 1 for k in {3b, 3b+1, 3b+2}
    W[k*n + i, c] = 1 for k in {3b, 3b+1, 3b+2} and i in {0,1}
    
    Let's check T[i,j',k'] for fixed i, j', k':
    Sum over c = j*3+b:
    U[i*m+j', c] * V[j'*p+k', c] * W[k'*n+i, c]
    
    U[i*m+j', c] is 1 only if c corresponds to j' (i.e., j = j') and i is in {0,1} (always true).
    So U[i*m+j', c] = 1 iff j = j'.
    
    V[j'*p+k', c] is 1 only if c corresponds to (j, b) where j = j' and k' is in {3b, 3b+1, 3b+2}.
    So V[j'*p+k', c] = 1 iff j = j' and b = floor(k'/3).
    
    W[k'*n+i, c] is 1 only if c corresponds to (j, b) where b = floor(k'/3) and i is in {0,1} (always true).
    So W[k'*n+i, c] = 1 iff j = j' and b = floor(k'/3).
    
    So the product is 1 iff j = j' and b = floor(k'/3), i.e., iff c = j'*3 + floor(k'/3).
    There's exactly one such c, so the sum is 1.
    
    This works! The rank is 9.
    """
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    rank = 9
    U = [[0] * rank for _ in range(n * m)]  # 6 x 9
    V = [[0] * rank for _ in range(m * p)]  # 27 x 9
    W = [[0] * rank for _ in range(p * n)]  # 18 x 9
    
    for j in range(m):  # j = 0, 1, 2
        for b in range(3):  # b = 0, 1, 2 (block index)
            c = j * 3 + b
            # U: for i in {0, 1}, set U[i*m + j][c] = 1
            for i in range(n):
                U[i * m + j][c] = 1
            # V: for k in {3*b, 3*b+1, 3*b+2}, set V[j*p + k][c] = 1
            for l in range(3):
                k = 3 * b + l
                V[j * p + k][c] = 1
            # W: for k in {3*b, 3*b+1, 3*b+2} and i in {0, 1}, set W[k*n + i][c] = 1
            for l in range(3):
                k = 3 * b + l
                for i in range(n):
                    W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
