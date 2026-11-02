import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    
    # 1. Generate Schoolbook Tensor (Rank 120)
    # T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # We represent the multiplication tensor T_{ijk} where C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor indices are (i, j, k).
    # U has shape (N, R), V (M, R), W (P, R)
    # T_{ijk} = sum_r U[i, r] V[j, r] W[k, r]
    # Note: The task specifies W[k*n+i, r]. So the index in W is k*n + i.
    # Let's map standard tensor indices (i, j, k) to matrix indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def get_schoolbook():
        R = n * m * p
        U = np.zeros((N, R))
        V = np.zeros((M, R))
        W = np.zeros((P, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return U, V, W

    U, V, W = get_schoolbook()
    current_rank = U.shape[1]
    
    # 2. Verification Function
    def verify(U, V, W):
        # Check if sum_r U[i,r] V[j,r] W[k,r] == 1 for all i,j,k
        # Efficient check: T_reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        # Target T is 1 for all i,j,k
        # However, doing full einsum for rank 120 is fast.
        # For speed, we can check a random subset or use matrix multiplication tricks.
        # T_{ijk} = (U^T V)_{ij} ? No.
        # Let's just do the full check if rank is small, or random sampling if large.
        # Since we start at 120, full check is ~120 * 4*5*6 = 14400 ops. Very fast.
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
        target = np.ones((n, m, p))
        return np.allclose(T_rec, target, atol=1e-9)

    # 3. Greedy Merging Search
    # Strategy: Try to replace two columns (r1, r2) with a single column (r_new)
    # such that the sum of the two rank-1 tensors equals the new rank-1 tensor?
    # No, that requires the sum of two rank-1 tensors to be rank 1, which implies they are parallel.
    # In schoolbook, terms are distinct (different i,j,k). So they are linearly independent.
    # We cannot simply merge two schoolbook terms into one unless we change the structure.
    
    # Alternative Structural Change: 
    # The schoolbook decomposition is not necessarily minimal.
    # For 4x5x6, the known lower bound is max(n*m, m*p, p*n) = 120? No.
    # Lower bound is max(n*m, m*p, p*n) is for border rank? 
    # Actually, the tensor rank of matrix multiplication M_{n,m,p} is at least max(n*m, m*p, p*n).
    # Wait, the trivial lower bound is max(n*m, m*p, p*n).
    # For 4x5x6: max(20, 30, 24) = 30.
    # So rank could be as low as 30?
    # But schoolbook is 120.
    # There are known decompositions better than schoolbook.
    # e.g. Winograd for 2x2x2 has rank 7 vs 8.
    # For 4x5x6, we can use block decomposition.
    
    # Let's try a Block Decomposition Strategy.
    # Split 4x5x6 into blocks.
    # 4x5 can be viewed as 2x2 blocks of 2x2? No, 4 is divisible by 2, 5 is not.
    # 4x5x6 = (2x2) x (2x2) x (2x2) + remainder?
    
    # Let's implement a generic search for "Winograd-like" improvements on 2x2x2 blocks.
    # If we can decompose a 2x2x2 block with rank 7, we save 1 rank per block.
    
    # How to apply?
    # We can partition the indices i, j, k.
    # i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}.
    
    # Let's try to find a decomposition using a known algorithm for small matrices.
    # Since we have time, we can use a randomized search to find a lower rank decomposition
    # by starting with schoolbook and applying random linear combinations to columns
    # and then checking if we can drop a column (sparsify).
    
    # Actually, a more robust approach for small sizes is to use the known fact that
    # rank(4x5x6) is likely lower than 120.
    # Let's try to construct a decomposition using Kronecker products if possible?
    # 4x5x6 is not a power of 2.
    
    # Let's try a heuristic: 
    # 1. Start with Schoolbook.
    # 2. Randomly generate a basis change for U, V, W? No, that preserves rank.
    # 3. We need to find a different decomposition.
    
    # Let's try to find a decomposition of rank R < 120 by solving a system of equations?
    # Too hard for R=119.
    
    # Let's look at the "Block Winograd" inspiration.
    # 4x5x6 can be split into:
    # 2x2x2 blocks?
    # i: 0,1 | 2,3
    # j: 0,1 | 2,3 | 4 (remainder)
    # k: 0,1 | 2,3 | 4,5 (remainder)
    
    # This is complex to implement correctly from scratch in 150s.
    
    # Let's try a simpler structural change: 
    # Use the fact that 4x5x6 contains 2x2x2 sub-problems.
    # We can define a custom tensor for 2x2x2 with rank 7.
    # Then map the 4x5x6 problem to a combination of these.
    
    # However, given the strict time limit and complexity, the safest "structural" change
    # that might yield a small improvement is to attempt to merge terms that share
    # partial indices using a greedy algorithm, but as noted, schoolbook terms are orthogonal.
    
    # Let's try a different angle: **Randomized Local Search on the Rank**.
    # We assume the optimal rank is close to the lower bound (30-40?).
    # But finding it is hard.
    
    # Let's stick to a guaranteed valid solution but try to optimize the representation.
    # The current score is 1.0 (Rank 120).
    # If we can find ANY rank < 120, we improve.
    
    # Is there a known rank for 4x5x6?
    # Strassen's algorithm is for 2x2x2 (rank 7).
    # For 4x4x4, rank is 49? (Strassen based: 7^2 = 49? No, 4x4x4 uses 2x2x2 blocks. 4x4x4 = 8 blocks of 2x2x2. 8*7 = 56? No, Strassen for 4x4x4 is rank 49? 
    # Actually, Strassen for n=2 is 7. For n=4, it's 7^2 = 49? No, the recursion is T(n) = 7 T(n/2). So T(4) = 7 T(2) = 7*7 = 49.
    # So 4x4x4 has rank <= 49.
    # Our problem is 4x5x6.
    # We can pad 4x5x6 to 8x8x8? No, that increases rank.
    # We can decompose 4x5x6 into a 4x4x4 part and a remainder.
    # 4x5x6 = 4x4x4 + 4x1x4 + 4x4x2 + 4x1x2?
    # Let's split j into 4+1 and k into 4+2.
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1 B1 + A2 B2.
    # B1 is 4x6. Split B1 into B11 (4x4) + B12 (4x2).
    # C = A1 B11 + A1 B12 + A2 B2.
    # A1 B11 is 4x4x4 multiplication. Rank <= 49 (Strassen).
    # A1 B12 is 4x4x2. Rank <= 4*4*2 = 32 (Schoolbook) or better?
    # A2 B2 is 4x1x2. Rank <= 4*1*2 = 8.
    # A1 B12: 4x4x2. Can we do better than 32?
    # 4x4x2 can be seen as two 4x4x1 multiplications.
    # 4x4x1 rank is 16? (A^T B vector). 
    # Actually, 4x4x2 is equivalent to multiplying 4x4 matrix by 4x2 matrix.
    # This is just 2 columns of the product.
    # We can use Strassen for 4x4x4 and just take the first 2 columns?
    # Strassen computes the full 4x4x4. It outputs 16 values. We only need 8 (4x2).
    # But the decomposition is fixed. We can't "discard" outputs easily without changing the structure.
    # However, if we use the Strassen decomposition for 4x4x4, the rank is 49.
    # The terms A1 B12 and A2 B2 are small.
    # Total Rank = Rank(4x4x4) + Rank(4x4x2) + Rank(4x1x2).
    # Rank(4x4x4) = 49.
    # Rank(4x4x2): We can use schoolbook for the 4x2 part of B.
    # A1 (4x4) * B12 (4x2).
    # This is 4x4x2 tensor.
    # We can decompose 4x4x2 as 2 * 4x4x1.
    # 4x4x1 rank is 16 (each output C[i,k] = sum_j A[i,j] B[j,k]).
    # So 4x4x2 rank is 32.
    # Rank(4x1x2): A2 (4x1) * B2 (1x2).
    # C[i,k] = A2[i] * B2[k].
    # This is an outer product. Rank 1?
    # C = A2 * B2^T.
    # Tensor T_{i,k} = A2_i B2_k.
    # This is rank 1 in the tensor sense?
    # U[i, r] V[j, r] W[k, r].
    # Here j is the shared dimension. j is size 1.
    # U[i, 0] = A2_i. V[0, 0] = 1. W[k, 0] = B2_k.
    # So rank 1.
    
    # Total Rank = 49 + 32 + 1 = 82.
    # 82 < 120.
    # This is a significant improvement!
    
    # Let's implement this block decomposition.
    
    def strassen_4x4x4():
        # Returns U, V, W for 4x4x4 multiplication with rank 49.
        # Strassen for 2x2x2 has rank 7.
        # 4x4x4 is 8 blocks of 2x2x2.
        # We can use the recursive structure.
        # Or simply use the known Strassen formulas for 4x4.
        # Actually, it's easier to use the 2x2x2 rank 7 decomposition and apply it to the 8 sub-blocks.
        # But the indices are interleaved.
        
        # Let's use a simpler approach: 
        # Just use schoolbook for 4x4x4? Rank 64.
        # 64 + 32 + 1 = 97.
        # 97 < 120.
        # Schoolbook is easier to implement correctly.
        
        # Let's try to get Strassen 2x2x2 rank 7 and scale it.
        # Strassen 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # We need to map this to the tensor indices.
        # For 4x4x4, we have 8 such 2x2x2 blocks.
        # The indices i,j,k in 4x4x4 can be written as i = 2*i1 + i0, etc.
        # The Strassen algorithm operates on the 2x2 blocks.
        # This requires careful indexing.
        
        # Given the time constraint, Schoolbook for the 4x4x4 block (Rank 64) is safer.
        # Total Rank 97.
        
        pass

    # Let's implement the Block Decomposition with Schoolbook for sub-blocks.
    # Split 4x5x6 into:
    # Block 1: 4x4x4 (i:0-3, j:0-3, k:0-3) -> Rank 64
    # Block 2: 4x4x2 (i:0-3, j:0-3, k:4-5) -> Rank 32
    # Block 3: 4x1x6 (i:0-3, j:4, k:0-5) -> Rank 24?
    # Wait, B is 5x6.
    # A (4x5) * B (5x6).
    # Split j into j1 (0-3) and j2 (4).
    # C = A1 B1 + A2 B2.
    # A1 is 4x4. B1 is 4x6.
    # A2 is 4x1. B2 is 1x6.
    
    # Term 1: A1 (4x4) * B1 (4x6).
    # Split k in B1 into k1 (0-3) and k2 (4-5).
    # B1 = [B11 (4x4) | B12 (4x2)].
    # C1 = A1 B11 + A1 B12.
    # A1 B11: 4x4x4. Rank 64.
    # A1 B12: 4x4x2. Rank 32.
    
    # Term 2: A2 (4x1) * B2 (1x6).
    # C2 = A2 B2.
    # This is 4x1x6.
    # Rank?
    # C[i,k] = A2[i] * B2[k].
    # This is an outer product of A2 and B2.
    # Tensor T_{i, j, k} where j is the single index (j=4).
    # U[i, r] V[j, r] W[k, r].
    # We can set V[4, 0] = 1.
    # U[i, 0] = A2[i].
    # W[k, 0] = B2[k].
    # But this is for a SINGLE column of the decomposition?
    # No, the decomposition must work for ALL inputs.
    # The tensor is fixed.
    # T_{i, 4, k} = 1 if we are looking at the term A2[i] B2[k].
    # Wait, the tensor entries are 1 for all valid i,j,k.
    # So for the sub-problem A2 * B2, the tensor is T_{i, 4, k} = 1.
    # We need to decompose this 4x1x6 tensor.
    # Rank of 4x1x6 tensor?
    # It is the outer product of vector e_i (in U space) and e_k (in W space) with scalar 1 in V space.
    # Actually, T_{i, j, k} = 1 for j=4.
    # This is a rank 1 tensor?
    # T = u * v * w^T?
    # u is 4x1 (all 1s), v is 1x1 (1), w is 6x1 (all 1s).
    # T_{i,j,k} = u_i v_j w_k.
    # If u_i=1, v_j=1, w_k=1, then T_{i,j,k}=1.
    # Yes, the 4x1x6 sub-tensor is Rank 1.
    # So we need 1 column for this part.
    
    # Total Rank = 64 (for 4x4x4) + 32 (for 4x4x2) + 1 (for 4x1x6) = 97.
    
    # Let's implement this.
    
    U_new = np.zeros((N, 97))
    V_new = np.zeros((M, 97))
    W_new = np.zeros((P, 97))
    
    col = 0
    
    # Part 1: 4x4x4 (i:0-3, j:0-3, k:0-3)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_idx = i * m + j  # i*m + j
                v_idx = j * p + k  # j*p + k
                w_idx = k * n + i  # k*n + i
                U_new[u_idx, col] = 1
                V_new[v_idx, col] = 1
                W_new[w_idx, col] = 1
                col += 1
    # col is now 64
    
    # Part 2: 4x4x2 (i:0-3, j:0-3, k:4-5)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_new[u_idx, col] = 1
                V_new[v_idx, col] = 1
                W_new[w_idx, col] = 1
                col += 1
    # col is now 64 + 32 = 96
    
    # Part 3: 4x1x6 (i:0-3, j:4, k:0-5)
    # This is rank 1.
    # We need U[i, col] * V[4, col] * W[k, col] = 1 for all i in 0-3, k in 0-5.
    # Set V[4, col] = 1.
    # Set U[i, col] = 1 for all i in 0-3.
    # Set W[k, col] = 1 for all k in 0-5.
    for i in range(4):
        u_idx = i * m + 4  # j=4
        U_new[u_idx, col] = 1
        
    v_idx = 4 * p + 0  # j=4, k=0? No, V index is j*p+k.
    # Wait, V index is j*p+k.
    # In the term A2 * B2, j is fixed at 4.
    # But V is indexed by j and k?
    # No, V is indexed by j (from A) and k (from B)?
    # Let's re-read the interface.
    # U: (n*m, rank). Index i*m+j.
    # V: (m*p, rank). Index j*p+k.
    # W: (p*n, rank). Index k*n+i.
    
    # In Part 3, the term is A2[i] * B2[k].
    # This corresponds to j=4.
    # The tensor entry is T_{i, 4, k} = 1.
    # We need U[i*m+4, r] * V[4*p+k, r] * W[k*n+i, r] = 1.
    # This must hold for all i, k.
    # If we set:
    # U[i*m+4, r] = 1
    # V[4*p+k, r] = 1
    # W[k*n+i, r] = 1
    # Then the product is 1.
    # But wait, V[4*p+k, r] depends on k.
    # If we set V[4*p+k, r] = 1 for all k, and W[k*n+i, r] = 1 for all k,
    # then for a fixed k, the product is 1*1*1 = 1.
    # This works!
    # So we just set all relevant entries to 1.
    
    for k in range(6):
        v_idx = 4 * p + k
        V_new[v_idx, col] = 1
        w_idx = k * n + 0 # Wait, W index is k*n+i.
        # We need W[k*n+i, r] = 1 for all i.
        # So we must set W[k*n+i, r] = 1 for all i for each k.
        
    # Let's correct the loop for Part 3.
    for i in range(4):
        for k in range(6):
            # We are adding to the SAME column 'col'
            # We need U[i*m+4, col] = 1
            # V[4*p+k, col] = 1
            # W[k*n+i, col] = 1
            
            # But if we set U[i*m+4, col] = 1 for multiple i, it's fine.
            # If we set V[4*p+k, col] = 1 for multiple k, it's fine.
            # If we set W[k*n+i, col] = 1 for multiple i,k, it's fine.
            
            # However, we must ensure that for a specific (i,k), the product is 1.
            # If we set all these to 1, then for any (i,k),
            # U[i*m+4, col] = 1
            # V[4*p+k, col] = 1
            # W[k*n+i, col] = 1
            # Product = 1.
            
            # So we just need to set all these entries to 1.
            
            pass # The logic above is correct, we just need to populate the arrays.

    # Populate Part 3
    for i in range(4):
        U_new[i * m + 4, col] = 1
        
    for k in range(6):
        V_new[4 * p + k, col] = 1
        for i in range(4):
            W_new[k * n + i, col] = 1
            
    col += 1
    # Total rank 97.

    # Verify
    if verify(U_new, V_new, W_new):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
