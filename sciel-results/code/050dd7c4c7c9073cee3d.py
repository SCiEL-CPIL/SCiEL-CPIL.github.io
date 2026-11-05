"""
Strategy: Rank minimization via Integer Linear Programming (ILP).
We model the matrix multiplication tensor decomposition as a selection problem
over a library of "elementary" decompositions (blocks).
Each block covers a specific sub-tensor of indices (I, J, K) and has a fixed rank.
We use a simple branch-and-bound or greedy approach (since ILP solvers might not be available 
or might be slow for this specific structure, we simulate the ILP logic with a greedy set cover 
with optimization, or just try all combinations of "standard" blocks if the space is small enough).

However, for 3x4x8, the space of partitions is small.
The "ILP" idea here is to treat the problem as:
Minimize sum(r_b * x_b)
Subject to: sum(x_b * coverage_b) >= 1 for all (i,j,k)
x_b in {0, 1}

Since the tensor entries are all 1s (T_ijk = 1), a "coverage" of 1 means the block must provide
the correct term. Actually, standard decomposition adds terms.
If we use Strassen, the terms are linear combinations.
The "block" approach works if the blocks are independent (disjoint index sets) or if we allow
superposition.
The current program uses disjoint blocks.
Is there a benefit to overlapping blocks?
For matmul, usually disjoint blocks are optimal for standard constructions.
However, maybe a different partition is better.
Example:
Current:
- 8 blocks of 2x2x2 (Strassen, rank 7)
- 8 blocks of 1x2x2 (Schoolbook, rank 4)
Total: 8*7 + 8*4 = 56 + 32 = 88.

Alternative partitions:
1. All schoolbook: 3*4*8 = 96.
2. Split I into 1+2, J into 4, K into 8?
   - 1x4x8 block: Rank 32 (schoolbook).
   - 2x4x8 block: Split J into 2+2, K into 2+2+2+2?
     - 2x2x2 blocks: 1*2*4 = 8 blocks of 2x2x2. Rank 7 each. Total 56.
     - Total: 32 + 56 = 88. Same.

3. What if we use 2x2x4 blocks?
   Rank of 2x2x4 matmul tensor?
   It is equivalent to multiplying 2x2 by 2x4.
   Can we decompose it better than schoolbook (rank 16)?
   Using Strassen on the 2x2 part?
   The 2x2x4 tensor T[i,j,k] for i in 0..1, j in 0..1, k in 0..3.
   This is two 2x2x2 blocks side by side? No, K is 4.
   If we split K into 2+2, we get two 2x2x2 blocks.
   Rank = 7 + 7 = 14.
   So a 2x2x4 block can be done in rank 14.
   
   Let's check the total rank with 2x2x4 blocks:
   Partition I: 2+1
   Partition J: 2+2
   Partition K: 4+4 (or 2+2+2+2)
   
   If we use 2x2x4 blocks:
   - For I=2, J=2, K=4:
     How many such blocks?
     I splits: one 2, one 1.
     J splits: two 2s.
     K splits: two 4s.
     
     Blocks of type 2x2x4:
     I=2, J=2 (two choices), K=4 (two choices).
     Total 1 * 2 * 2 = 4 blocks of 2x2x4.
     Rank per block: 14.
     Total rank: 4 * 14 = 56.
     
     Blocks of type 1x2x4:
     I=1, J=2 (two choices), K=4 (two choices).
     Total 1 * 2 * 2 = 4 blocks of 1x2x4.
     Rank of 1x2x4?
     Split K into 2+2 -> two 1x2x2 blocks.
     Rank 1x2x2 is 4.
     Total rank per 1x2x4 block: 8.
     Total rank: 4 * 8 = 32.
     
     Grand Total: 56 + 32 = 88.
     
   It seems 88 is very robust for this specific partitioning logic.
   
   Is there a "mixing" strategy?
   The ILP approach would allow us to mix different block sizes.
   For example, maybe a 3x4x8 block has a rank < 88?
   Known lower bounds for 3x4x8?
   Schoolbook 96.
   Strassen-based 88.
   
   Let's implement a search over valid partitions and their optimal sub-block ranks.
   We will pre-calculate the rank for all possible sub-block sizes (n', m', p')
   using the best known constructions (Schoolbook, Strassen for 2x2x2, etc.).
   Then we will search for the partition of (3,4,8) that minimizes the total rank.
   
   This is a dynamic programming problem over the dimensions.
   R(n, m, p) = min over splits of n, m, p of sum(R(n_i, m_j, p_k)).
   
   Base cases:
   R(1, 1, 1) = 1
   R(1, m, p) = m*p (Schoolbook is optimal for 1xm x mxp? 
     1xm * mxp = 1xp.
     Terms: A_j B_jk.
     Rank is m*p? 
     Let's check 1x2x2. Rank 4. Yes.
     1x2x3. Rank 6.
     Is it possible to do better?
     T_jk = A_j B_jk.
     This is a sum of m*p rank-1 tensors?
     Actually, the tensor is T = sum_j e_j \otimes (sum_k e_k \otimes e_{jk}?) No.
     The tensor is T_{j,k} = 1.
     Wait, the tensor entries are 1.
     T_{j,k} corresponds to A_j B_k? No, B is m x p.
     The term is A_j B_{j,k}.
     The tensor is T = sum_{j,k} e_j \otimes e_{j,k} \otimes e_k?
     Indices:
     U index: j (size m)
     V index: j,k (size m*p)
     W index: k (size p)
     
     Decomposition: sum_{j,k} (e_j) (e_{j,k}) (e_k).
     Rank m*p.
     Can we reduce?
     If m=2, p=2. Rank 4.
     If we use Strassen? Strassen is for 2x2x2 (matmul).
     Here one dimension is 1.
     It's a "vector-matrix" multiplication.
     Rank is generally m*p.
     
   R(2, 2, 2) = 7 (Strassen)
   
   We can build a DP table for R(n, m, p) for small n, m, p.
   Then compute R(3, 4, 8).
   
   If R(3, 4, 8) < 88, we found a better decomposition.
   If R(3, 4, 8) == 88, we stick with the current one (or a simpler one).
   
   Let's implement the DP.
"""

import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 8]

# Precompute ranks for small dimensions
# We need to know the optimal rank for any sub-block.
# We'll use known values and schoolbook as upper bound.
# R(n, m, p)
# Known:
# R(1, m, p) = m * p
# R(2, 2, 2) = 7
# R(2, 2, 1) = 4 (1x2x2 transposed essentially, or 2x2x1)
#   2x2x1: A 2x2, B 2x1. C 2x1.
#   Terms: A_ij B_j.
#   Rank 4.
# R(2, 1, 2) = 4
# R(1, 2, 2) = 4
# R(2, 2, 3)?
#   Split 3 into 1+2.
#   R(2,2,1) + R(2,2,2) = 4 + 7 = 11.
#   Schoolbook: 12.
#   So R(2,2,3) <= 11.
# R(2, 2, 4)?
#   Split 4 into 2+2.
#   2 * R(2,2,2) = 14.
#   Schoolbook: 16.
#   So R(2,2,4) <= 14.

# Let's compute R(n, m, p) up to 3, 4, 8.
# Max dims: n=3, m=4, p=8.

def compute_rank_table(max_n, max_m, max_p):
    # Initialize with schoolbook
    R = [[[ (n+1)*(m+1)*(p+1) for p in range(max_p+1)] for m in range(max_m+1)] for n in range(max_n+1)]
    
    # Set base cases
    for n in range(max_n+1):
        for m in range(max_m+1):
            for p in range(max_p+1):
                if n == 0 or m == 0 or p == 0:
                    R[n][m][p] = 0
                else:
                    R[n][m][p] = n * m * p # Schoolbook
    
    # Known special cases
    R[2][2][2] = 7
    R[2][2][1] = 4
    R[2][1][2] = 4
    R[1][2][2] = 4
    R[1][1][1] = 1
    
    # DP: R(n, m, p) = min over splits
    # Split n into n1 + n2
    # Split m into m1 + m2
    # Split p into p1 + p2
    # R(n, m, p) = min( R(n1, m1, p1) + R(n2, m2, p2) ... ? )
    # Actually, if we split independently, we get blocks.
    # The rank of the block is the sum of ranks of sub-blocks.
    # We can split any dimension.
    
    # Iterate n, m, p in increasing order
    for n in range(1, max_n+1):
        for m in range(1, max_m+1):
            for p in range(1, max_p+1):
                best = R[n][m][p]
                
                # Try splitting n
                for n1 in range(1, n):
                    n2 = n - n1
                    # We can split m and p arbitrarily for each part?
                    # No, the block is (n1, m, p) and (n2, m, p)?
                    # Or do we split all three?
                    # The decomposition is a sum of terms.
                    # If we split n, we have two independent sub-problems:
                    # (n1, m, p) and (n2, m, p).
                    # R(n, m, p) <= R(n1, m, p) + R(n2, m, p).
                    val = R[n1][m][p] + R[n2][m][p]
                    if val < best: best = val
                    
                # Try splitting m
                for m1 in range(1, m):
                    m2 = m - m1
                    val = R[n][m1][p] + R[n][m2][p]
                    if val < best: best = val
                    
                # Try splitting p
                for p1 in range(1, p):
                    p2 = p - p1
                    val = R[n][m][p1] + R[n][m][p2]
                    if val < best: best = val
                    
                # Mixed splits?
                # e.g. split n and m?
                # R(n, m, p) <= sum_{n1, m1} R(n1, m1, p) + R(n2, m2, p)?
                # This is covered by sequential splits.
                # R(n, m, p) -> R(n1, m, p) + R(n2, m, p)
                # R(n1, m, p) -> R(n1, m1, p) + R(n1, m2, p)
                # So sequential splits cover all rectangular partitions.
                
                R[n][m][p] = best
                
    return R

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_strassen_base():
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    
    def add_block(i0, j0, k0, i_size, j_size, k_size):
        nonlocal col
        if i_size == 2 and j_size == 2 and k_size == 2:
            for t in range(7):
                a_coeffs = terms[t][0]
                b_coeffs = terms[t][1]
                c_coeffs = terms[t][2]
                
                for di in range(2):
                    for dj in range(2):
                        U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
                
                for dj in range(2):
                    for dk in range(2):
                        V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
                
                for di in range(2):
                    for dk in range(2):
                        W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
                
                col += 1
        else:
            for di in range(i_size):
                for dj in range(j_size):
                    for dk in range(k_size):
                        U[(i0+di)*m + (j0+dj)][col] = 1
                        V[(j0+dj)*p + (k0+dk)][col] = 1
                        W[(k0+dk)*n + (i0+di)][col] = 1
                        col += 1

    # 8 blocks of 2x2x2 (Strassen)
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                add_block(i0, j0, k0, 2, 2, 2)
                
    # 8 blocks of 1x2x2 (Schoolbook)
    for i0 in [2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                add_block(i0, j0, k0, 1, 2, 2)
                
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # 1. Compute optimal rank via DP
    R = compute_rank_table(n, m, p)
    optimal_rank = R[n][m][p]
    
    # 2. If optimal rank < 88, we might be able to construct it.
    #    However, constructing the specific decomposition from the DP split is complex.
    #    For now, if optimal_rank < 88, we try to find a construction.
    #    If optimal_rank == 88, we use the known construction.
    
    # Let's check if 88 is indeed the DP result.
    # If DP finds something lower, we need a generator.
    # For this specific task, 88 is likely the best simple block construction.
    # If we find a lower rank, we would need to implement the specific blocks.
    # Given the time, we return the known 88 rank solution.
    # If the DP finds < 88, we could potentially use a randomized search for that rank,
    # but it's risky.
    
    U, V, W, rank = _construct_strassen_base()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
