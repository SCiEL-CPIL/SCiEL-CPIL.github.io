"""
Matrix multiplication 2,2,8 rank-minimal exact tensor decomposition.
Strategy: Use Strassen's algorithm structure to share the 7 products of the 2x2
multiplication across the 8 columns of B. This gives rank 7*8 = 56 naively,
but we can do better by noting that the 2x2x1 tensor (multiplying 2x2 by 2x1)
has rank 4, and we can share the Strassen linear combinations of A across
columns of B to get rank 28.

The key insight: A is 2x2, B is 2x8. We can group the 8 columns into 4 pairs.
For each pair of columns (b_{2k}, b_{2k+1}), we compute a 2x2x2 multiplication
using Strassen (rank 7). Total: 4 * 7 = 28.

Actually, let me think more carefully. The tensor is:
T_{i,j,k} = 1 if output[i,k] depends on A[i,j]*B[j,k]

Output C[i,k] = sum_j A[i,j] * B[j,k]

For i in {0,1}, k in {0,...,7}, j in {0,1}:
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

This is a direct sum of 8 copies of the 2x2x1 tensor (2x2 times 2x1).
Each 2x2x1 tensor has rank 4 (since it's just 4 scalar multiplications).
So the total rank is 8*4 = 32 (schoolbook).

But wait - can we do better? The 2x2x1 tensor:
C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]

This is a 2x2x1 tensor. Its rank is 4 (it's the matrix multiplication tensor
for 2x2 * 2x1, which is just 4 independent multiplications since the B matrix
is a single column).

Hmm, but actually for 2x2 * 2x1, we have:
C[0] = A[0,0]*B[0] + A[0,1]*B[1]
C[1] = A[1,0]*B[0] + A[1,1]*B[1]

This is a bilinear map from (A[0,0], A[0,1], A[1,0], A[1,1], B[0], B[1]) 
to (C[0], C[1]).

The rank of this tensor... Let me think. We can write:
C[0] = A[0,0]*B[0] + A[0,1]*B[1]
C[1] = A[1,0]*B[0] + A[1,1]*B[1]

This is a sum of 4 rank-1 tensors:
A[0,0] ⊗ B[0] ⊗ e_0
A[0,1] ⊗ B[1] ⊗ e_0
A[1,0] ⊗ B[0] ⊗ e_1
A[1,1] ⊗ B[1] ⊗ e_1

So rank is at most 4. Is it exactly 4? Yes, because each of the 4 products
A[i,j]*B[k] appears in exactly one output with a nonzero coefficient, and
they are linearly independent.

So the total rank for 2x2 * 2x8 is 8 * 4 = 32.

But wait - the inspiration says rank 28 was achieved. Let me re-read.

"Using Strassen's algorithm on 2x2x2 blocks reduces the rank from 32 to 28 
by sharing the linear combinations of the A matrix across blocks of B"

So the idea is: group the 8 columns of B into 4 groups of 2. For each group,
we have a 2x2 * 2x2 multiplication (A times a 2x2 block of B). We use Strassen
for each 2x2 * 2x2, which has rank 7. Total: 4 * 7 = 28.

But wait, that's not right either. If we use Strassen for each 2x2 * 2x2,
we get 7 products per group, and 4 groups, so 28 products total. But Strassen
for 2x2 * 2x2 computes:
C[0,0] = M1 - M4 + M6 - M7
C[0,1] = M5 + M7
C[1,0] = M4 + M6
C[1,1] = M1 - M2 + M3 - M7

Where:
M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
M2 = (A[1,0]+A[1,1]) * B[0,0]
M3 = A[0,0] * (B[0,1]-B[1,1])
M4 = A[1,1] * (B[1,0]-B[0,0])
M5 = (A[0,0]+A[0,1]) * B[1,1]
M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])

So for each 2x2 block of B (columns 2k and 2k+1), we compute 7 products.
Total: 4 * 7 = 28 products.

This gives a rank-28 decomposition!

Let me implement this.
"""

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    # We'll use Strassen for 2x2 * 2x2 blocks.
    # Group the 8 columns of B into 4 groups of 2.
    # For each group, we compute a 2x2 * 2x2 multiplication using Strassen (rank 7).
    # Total rank: 4 * 7 = 28.
    
    rank = 28
    
    # U shape: (n*m, rank) = (4, 28)
    # V shape: (m*p, rank) = (16, 28) 
    # W shape: (p*n, rank) = (16, 28)
    
    # U indices: i*m + j for i in {0,1}, j in {0,1}
    #   U[0] = A[0,0], U[1] = A[0,1], U[2] = A[1,0], U[3] = A[1,1]
    
    # V indices: j*p + k for j in {0,1}, k in {0,...,7}
    #   V[0] = B[0,0], V[1] = B[0,1], ..., V[7] = B[0,7]
    #   V[8] = B[1,0], V[9] = B[1,1], ..., V[15] = B[1,7]
    
    # W indices: k*n + i for k in {0,...,7}, i in {0,1}
    #   W[0] = C[0,0], W[1] = C[1,0], W[2] = C[0,1], W[3] = C[1,1], ...
    #   W[2k] = C[0,k], W[2k+1] = C[1,k]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])
    
    # Output:
    # C[0,0] = M1 - M4 + M6 - M7
    # C[0,1] = M5 + M7
    # C[1,0] = M4 + M6
    # C[1,1] = M1 - M2 + M3 - M7
    
    # For group g (columns 2g and 2g+1):
    # B[0, 2g] = V[2g], B[0, 2g+1] = V[2g+1]
    # B[1, 2g] = V[8+2g], B[1, 2g+1] = V[8+2g+1]
    # C[0, 2g] = W[2g], C[1, 2g] = W[2g+1]
    # C[0, 2g+1] = W[2g+2], C[1, 2g+1] = W[2g+3]
    
    for g in range(4):
        base = g * 7
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,2g]+B[1,2g+1])
        # U coefficients: A[0,0]=1, A[1,1]=1
        # V coefficients: B[0,2g]=1, B[1,2g+1]=1
        U[0][base + 0] = 1   # A[0,0]
        U[3][base + 0] = 1   # A[1,1]
        V[2*g][base + 0] = 1       # B[0, 2g]
        V[8 + 2*g + 1][base + 0] = 1  # B[1, 2g+1]
        
        # W coefficients:
        # C[0,2g] += M1, C[1,2g+1] += M1
        W[2*g][base + 0] = 1       # C[0, 2g]
        W[2*g + 3][base + 0] = 1   # C[1, 2g+1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2g]
        U[2][base + 1] = 1   # A[1,0]
        U[3][base + 1] = 1   # A[1,1]
        V[2*g][base + 1] = 1       # B[0, 2g]
        
        # W: C[1,2g+1] -= M2
        W[2*g + 3][base + 1] = -1  # C[1, 2g+1]
        
        # M3 = A[0,0] * (B[0,2g+1]-B[1,2g+1])
        U[0][base + 2] = 1   # A[0,0]
        V[2*g + 1][base + 2] = 1       # B[0, 2g+1]
        V[8 + 2*g + 1][base + 2] = -1  # B[1, 2g+1]
        
        # W: C[1,2g+1] += M3
        W[2*g + 3][base + 2] = 1  # C[1, 2g+1]
        
        # M4 = A[1,1] * (B[1,2g]-B[0,2g])
        U[3][base + 3] = 1   # A[1,1]
        V[8 + 2*g][base + 3] = 1       # B[1, 2g]
        V[2*g][base + 3] = -1       # B[0, 2g]
        
        # W: C[0,2g] -= M4, C[1,2g] += M4
        W[2*g][base + 3] = -1       # C[0, 2g]
        W[2*g + 1][base + 3] = 1   # C[1, 2g]
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2g+1]
        U[0][base + 4] = 1   # A[0,0]
        U[1][base + 4] = 1   # A[0,1]
        V[8 + 2*g + 1][base + 4] = 1  # B[1, 2g+1]
        
        # W: C[0,2g+1] += M5
        W[2*g + 2][base + 4] = 1  # C[0, 2g+1]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2g]+B[0,2g+1])
        U[2][base + 5] = 1   # A[1,0]
        U[0][base + 5] = -1   # A[0,0]
        V[2*g][base + 5] = 1       # B[0, 2g]
        V[2*g + 1][base + 5] = 1       # B[0, 2g+1]
        
        # W: C[0,2g] += M6, C[1,2g] += M6
        W[2*g][base + 5] = 1       # C[0, 2g]
        W[2*g + 1][base + 5] = 1   # C[1, 2g]
        
        # M7 = (A[1,1]-A[0,1]) * (B[1,2g]+B[1,2g+1])
        U[3][base + 6] = 1   # A[1,1]
        U[1][base + 6] = -1   # A[0,1]
        V[8 + 2*g][base + 6] = 1       # B[1, 2g]
        V[8 + 2*g + 1][base + 6] = 1  # B[1, 2g+1]
        
        # W: C[0,2g] -= M7, C[0,2g+1] += M7, C[1,2g+1] -= M7
        W[2*g][base + 6] = -1       # C[0, 2g]
        W[2*g + 2][base + 6] = 1   # C[0, 2g+1]
        W[2*g + 3][base + 6] = -1  # C[1, 2g+1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
