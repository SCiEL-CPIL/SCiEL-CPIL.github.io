"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 5
    
    # We will construct a rank-24 decomposition using a "row-block" approach.
    # The idea is to compute the 2x5 product C = A*B where A is 2x4 and B is 4x5.
    # We can view this as computing 5 columns of C.
    # Each column c_k = A * B[:, k] is a 2x1 vector.
    # A is 2x4, B[:, k] is 4x1.
    # The product of a 2x4 and 4x1 matrix is a 2x1 matrix.
    # The rank of 2x4x1 multiplication is 2 (it's just two dot products).
    # So for each of the 5 columns, we need rank 2. Total rank = 5 * 2 = 10?
    # Wait, the inputs are shared. We cannot just do 5 separate 2x4x1 multiplications 
    # because A is the same. But we can.
    # Let's check the tensor decomposition definition.
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # Here i in {0,1}, j in {0..3}, k in {0..4}.
    # If we use 10 terms, each term r corresponds to one "partial product" A_{i,j} * B_{j,k}.
    # No, that's rank 20 (4*5).
    # The naive decomposition uses 20 terms: for each j, sum A_{i,j} B_{j,k}.
    # Term r is indexed by j and... wait.
    # Naive: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a sum of 4 terms. Each term is a product of a 2x1 vector (col j of A) 
    # and a 1x5 vector (row j of B).
    # So the tensor is a sum of 4 rank-1 tensors (outer products).
    # Rank of 2x4x5 multiplication tensor?
    # The rank of the matrix multiplication tensor M_{m,n,p} is at most m*n*p.
    # It is also at most min(mn, np, pm) * max(m, n, p) ? No.
    # The rank is bounded by the number of terms in a bilinear algorithm.
    # Naive algorithm: sum_{j=1}^m A_{i,j} B_{j,k}. This uses m terms.
    # Each term is rank 1 in the tensor sense?
    # T = sum_j (A_{.,j} B_{j,.}^T)_{i,k}.
    # Yes, each term is a rank-1 tensor (outer product of vector A_{.,j} in R^{n*p?} no).
    # The tensor T has indices (i, j, k).
    # T_{i,j,k} = delta_{i,k?} No, T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor T has 1s at (i, j, k) if... wait.
    # The tensor T is defined such that C_{i,k} = sum_{j,r} U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # For naive: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This corresponds to rank m=4.
    # U_{i,j,r} = A_{i,j} if r==j, else 0.
    # V_{j,k,r} = B_{j,k} if r==j, else 0.
    # W_{k,i,r} = 1 if r==j, else 0.
    # So rank 4 is possible?
    # Let's check.
    # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # = sum_j (A_{i,j} if r==j else 0) * (B_{j,k} if r==j else 0) * (1 if r==j else 0)
    # = sum_j A_{i,j} B_{j,k} * 1 = C_{i,k}.
    # Yes! The rank is at most 4.
    # Is rank 4 minimal?
    # The border rank of matrix multiplication is known to be high, but the rank?
    # For 2x4x5, the rank is at most 4.
    # Can we do better? Rank 3?
    # The rank of M_{2,2,2} is 7.
    # M_{2,4,5} contains M_{2,2,2} as a sub-tensor. So rank >= 7?
    # No, the rank is not monotonic under restriction in that simple way for exact rank?
    # Actually, if T contains S as a sub-tensor, rank(T) >= rank(S).
    # So rank(M_{2,4,5}) >= rank(M_{2,2,2}) = 7.
    # So rank is between 7 and 4? Contradiction.
    # Ah, the naive algorithm I described uses rank 4?
    # Let's re-read the tensor definition.
    # T_{i,j,k} is the trilinear form?
    # The matrix multiplication tensor T_{n,m,p} has indices (i, j, k) where i in 1..n, j in 1..m, k in 1..p.
    # T_{i,j,k} = 1 if the entry (i,k) of the product depends on A_{i,j} and B_{j,k}.
    # Yes, T_{i,j,k} = 1 for all i,j,k.
    # Wait, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T is defined by C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # This is a sum of m terms. Each term is a "rank 1" tensor?
    # A rank 1 tensor is u_i v_j w_k.
    # Here the term is A_{i,j} B_{j,k}. This is NOT a product of three independent vectors.
    # The indices are coupled: A has (i,j), B has (j,k).
    # So the naive algorithm is NOT a sum of rank-1 tensors.
    # A rank-1 tensor decomposition term is U_{i,j} V_{j,k} W_{k,i}.
    # In the naive sum, for a fixed j, we have A_{i,j} * B_{j,k} * 1.
    # Let U^{(j)}_{i,j'} = delta_{j,j'}. (Vector in i, j-space? No, U is n*m x rank).
    # U_{i*m+j', r}.
    # If we set r=j, U_{i*m+j, j} = 1. All other 0.
    # V_{j'*p+k, j} = delta_{j,j'}? No.
    # V_{j'*p+k, r}. We want V_{j*p+k, j} = 1.
    # W_{k*n+i, j} = 1.
    # Then sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # = sum_j U_{i*m+j, j} V_{j*p+k, j} W_{k*n+i, j}
    # = sum_j 1 * 1 * 1 = m.
    # This gives m, not C_{i,k}.
    # I made a mistake.
    # The term A_{i,j} B_{j,k} cannot be written as U_{i,j} V_{j,k} W_{k,i} with constants.
    # It requires the values A and B.
    # In tensor decomposition, the coefficients U, V, W are FIXED.
    # The inputs A, B are the variables.
    # So C_{i,k} = sum_r (sum_{j} U_{i,j,r} A_{i,j}) (sum_{j'} V_{j',k,r} B_{j',k}) W_{k,i,r}.
    # This is the correct form.
    # So we need to find U, V, W such that:
    # sum_r W_{k,i,r} (sum_j U_{i,j,r} A_{i,j}) (sum_{j'} V_{j',k,r} B_{j',k}) = sum_j A_{i,j} B_{j,k}.
    # This is a much harder problem.
    # The rank of M_{2,4,5} is known to be at least 7 (from M_{2,2,2}).
    # And at most ... well, we can use the standard algorithm.
    # The standard algorithm for 2x4x5 can be done in rank 20?
    # Or better.
    # Let's look at the previous solution. It used rank 36.
    # My naive attempt above was wrong because I confused the bilinear complexity with rank-1 tensor terms.
    
    # Let's try to find a small rank decomposition.
    # Rank 7 is the lower bound.
    # Can we achieve rank 7?
    # Maybe not for 2,4,5.
    # Let's try a constructive approach.
    # We can use the fact that n=2 is small.
    # C_{0,k} = sum_j A_{0,j} B_{j,k}
    # C_{1,k} = sum_j A_{1,j} B_{j,k}
    # This is just two dot products.
    # We can compute this by forming the 4x5 matrix M = A^T B? No.
    # We want to compute y = A B.
    # This is equivalent to computing the product of a 2x4 and 4x5.
    
    # Let's use a known decomposition for small cases.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x4x5, we can break B into 2x2, 2x2, 2x1 blocks?
    # Or A into 1x4, 1x4 rows?
    # If we treat each row of A separately:
    # Row 0 of C: A_{0,0} B_{0,:} + A_{0,1} B_{1,:} + A_{0,2} B_{2,:} + A_{0,3} B_{3,:}
    # This is a sum of 4 terms. Each term is a 1x5 vector.
    # This doesn't help with tensor rank directly.
    
    # Let's go back to the Strassen-based approach but optimize it.
    # The previous code used rank 36.
    # 4 blocks of 2x2x2 (rank 7 each) = 28.
    # Plus 2 blocks of 2x2x1 (rank 4 each) = 8.
    # Total 36.
    # Can we reduce the rank of the 2x2x1 blocks?
    # 2x2x1 multiplication: C_0 = A_0 B_0 + A_1 B_1, C_1 = A_2 B_0 + A_3 B_1.
    # This is rank 4 (naive).
    # Can it be done in rank 3?
    # The tensor is T_{i,j,k} with i in {0,1}, j in {0,1,2,3}? No.
    # Indices: A is 2x2 (i,j), B is 2x1 (j,k), C is 2x1 (i,k).
    # T_{i,j,k} = 1 if C_i depends on A_{i,j} and B_{j,k}.
    # C_0 = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}
    # C_1 = A_{1,0} B_{0,0} + A_{1,1} B_{1,0}
    # Wait, B is 2x1. So B has entries B_{0,0} and B_{1,0}.
    # A has A_{0,0}, A_{0,1}, A_{1,0}, A_{1,1}.
    # Terms:
    # 1. A_{0,0} B_{0,0} -> C_0
    # 2. A_{0,1} B_{1,0} -> C_0
    # 3. A_{1,0} B_{0,0} -> C_1
    # 4. A_{1,1} B_{1,0} -> C_1
    # This is 4 rank-1 terms?
    # Term 1: U_{0,0}=1, V_{0,0}=1, W_{0,0}=1. (i=0, j=0, k=0).
    # Term 2: U_{0,1}=1, V_{1,0}=1, W_{0,0}=1. (i=0, j=1, k=0).
    # Term 3: U_{1,0}=1, V_{0,0}=1, W_{1,0}=1. (i=1, j=0, k=0).
    # Term 4: U_{1,1}=1, V_{1,0}=1, W_{1,0}=1. (i=1, j=1, k=0).
    # Yes, rank 4.
    # Is rank 3 possible?
    # The support of the tensor is 4 points.
    # If rank 3, we need 3 rank-1 tensors to cover these 4 points.
    # By pigeonhole, one rank-1 tensor must cover at least 2 points?
    # A rank-1 tensor U_i V_j W_k has support (i,j,k) where U_i != 0, V_j != 0, W_k != 0.
    # It's a box.
    # The points are (0,0,0), (0,1,0), (1,0,0), (1,1,0).
    # Note k is always 0. So W_k is 1 for k=0, 0 for k=1 (if B was 2x2). Here B is 2x1, so k only takes value 0.
    # So W is a scalar 1.
    # We need to decompose the 2x2 matrix of coefficients into sum of 3 rank-1 matrices.
    # The matrix is [[1, 1], [1, 1]].
    # This matrix has rank 1.
    # So it can be decomposed into 1 rank-1 term!
    # U_{0,0}=1, U_{0,1}=1, U_{1,0}=1, U_{1,1}=1.
    # V_{0,0}=1, V_{1,0}=1.
    # Then sum_j U_{i,j} V_{j,k} = 1*1 + 1*1 = 2.
    # This gives 2, not 1.
    # We need sum_j U_{i,j} V_{j,k} = delta_{i,0}? No.
    # We need the product to be 1 for all (i,j) pairs that are present.
    # The tensor T is 1 at (0,0,0), (0,1,0), (1,0,0), (1,1,0).
    # We want sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = T_{i,j,k}.
    # If we use 1 term: U_{i,j,0} V_{j,0,0} W_{0,i,0} = 1.
    # This requires U_{i,j,0} V_{j,0,0} W_{0,i,0} = 1 for all i,j.
    # Let W_{0,0,0}=1, W_{0,1,0}=1.
    # Then U_{0,0,0} V_{0,0,0} = 1
    # U_{0,1,0} V_{1,0,0} = 1
    # U_{1,0,0} V_{0,0,0} = 1
    # U_{1,1,0} V_{1,0,0} = 1
    # From 1 and 3: U_{0,0,0} = U_{1,0,0}.
    # From 2 and 4: U_{0,1,0} = U_{1,1,0}.
    # Let U_{0,0,0}=1, U_{1,0,0}=1. Then V_{0,0,0}=1.
    # Let U_{0,1,0}=1, U_{1,1,0}=1. Then V_{1,0,0}=1.
    # Check:
    # (0,0,0): 1*1*1 = 1. OK.
    # (0,1,0): 1*1*1 = 1. OK.
    # (1,0,0): 1*1*1 = 1. OK.
    # (1,1,0): 1*1*1 = 1. OK.
    # So rank 1 is sufficient for 2x2x1 multiplication?
    # Wait, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # If we use 1 term:
    # C_{i,k} = (sum_j U_{i,j} A_{i,j}) (sum_{j'} V_{j',k} B_{j',k}) W_{k,i}.
    # With U=1, V=1, W=1:
    # C_{i,k} = (A_{i,0}+A_{i,1}) (B_{0,k}+B_{1,k}) * 1.
    # This is NOT equal to A_{i,0} B_{0,k} + A_{i,1} B_{1,k}.
    # The tensor decomposition is about the COEFFICIENTS, not the values.
    # The formula C_{i,k} = sum_r ... is an identity in the variables A, B.
    # So (A_{i,0}+A_{i,1})(B_{0,k}+B_{1,k}) = A_{i,0} B_{0,k} + A_{i,0} B_{1,k} + A_{i,1} B_{0,k} + A_{i,1} B_{1,k}.
    # This has cross terms A_{i,0} B_{1,k} and A_{i,1} B_{0,k}.
    # The target C_{i,k} = A_{i,0} B_{0,k} + A_{i,1} B_{1,k} does NOT have these cross terms.
    # So rank 1 is NOT sufficient.
    # We need to cancel the cross terms.
    # This requires more terms.
    # The rank of 2x2x1 is actually 4? Or 3?
    # It is known that rank(2x2x1) = 4? No, it's a subcase of 2x2x2.
    # Actually, the rank of the matrix multiplication tensor M_{n,m,p} is at least n+m+p-1? No.
    # Let's just use the naive rank 4 for the 2x2x1 blocks, and rank 7 for 2x2x2.
    # The previous solution did this and got 36.
    # Can we do better?
    # Maybe we can share terms between blocks?
    # The blocks were:
    # 1. A_L B_11 (2x2 * 2x2)
    # 2. A_R B_21 (2x2 * 2x2)
    # 3. A_L B_12 (2x2 * 2x2)
    # 4. A_R B_22 (2x2 * 2x2)
    # 5. A_L B_15 (2x2 * 2x1)
    # 6. A_R B_25 (2x2 * 2x1)
    # Note that A_L is used in blocks 1, 3, 5.
    # A_R is used in blocks 2, 4, 6.
    # Can we combine them?
    # Instead of treating A_L and A_R separately, treat A as 2x4.
    # We want to compute A B where B is 4x5.
    # We can use a decomposition of the 4x5 multiplication.
    # But the rank is determined by the full tensor.
    
    # Let's try a different partition.
    # Partition B into 5 columns, each 4x1.
    # C = [c_0, c_1, c_2, c_3, c_4].
    # c_k = A b_k.
    # A is 2x4, b_k is 4x1.
    # This is 5 independent 2x4x1 multiplications.
    # Rank of 2x4x1?
    # C_0 = sum_{j=0}^3 A_{0,j} B_{j,k}
    # C_1 = sum_{j=0}^3 A_{1,j} B_{j,k}
    # This is similar to 2x2x1 but with 4 terms.
    # Rank of 2xn x 1?
    # It is at most 2n.
    # Can it be less?
    # For n=2, rank is 4.
    # For n=4, rank is likely 8? Or less?
    # If rank is 8 per column, total rank 40. Worse than 36.
    # If rank is 4 per column (unlikely), total 20.
    
    # Let's stick with the 36 solution but see if we can optimize the 2x2x1 part.
    # The 2x2x1 part was rank 4.
    # Is rank 3 possible for 2x2x1?
    # We need to represent T with 3 terms.
    # T has 4 non-zero entries.
    # We need 3 rank-1 tensors to cover 4 points.
    # One tensor must cover 2 points.
    # Points: (0,0,0), (0,1,0), (1,0,0), (1,1,0).
    # A rank-1 tensor covers a box.
    # To cover (0,0,0) and (0,1,0), it must have i=0, k=0, and j in {0,1}.
    # So U_{0,j} != 0 for j=0,1. V_{j,0} != 0 for j=0,1. W_{0,0} != 0.
    # This tensor also covers (1,0,0) if U_{1,j} != 0? No, i is fixed to 0.
    # So it covers (0,0,0) and (0,1,0) only.
    # Similarly, another tensor can cover (1,0,0) and (1,1,0).
    # So we can do it with 2 tensors?
    # Term 1: i=0, j in {0,1}, k=0.
    # Term 2: i=1, j in {0,1}, k=0.
    # Let's check the values.
    # T_{0,0,0} = 1. T_{0,1,0} = 1.
    # Term 1 contributes U_{0,0} V_{0,0} W_{0,0} to (0,0,0) and U_{0,1} V_{1,0} W_{0,0} to (0,1,0).
    # We need both to be 1.
    # Let W_{0,0}=1. U_{0,0}=1, V_{0,0}=1. U_{0,1}=1, V_{1,0}=1.
    # Term 2 contributes to (1,0,0) and (1,1,0).
    # Let W_{1,0}=1. U_{1,0}=1, V_{0,0}=1. U_{1,1}=1, V_{1,0}=1.
    # Now check cross terms.
    # Does Term 1 contribute to (1,0,0)? U_{1,j} is 0 for Term 1. So no.
    # Does Term 2 contribute to (0,0,0)? U_{0,j} is 0 for Term 2. So no.
    # So this decomposition works!
    # Rank 2 for 2x2x1?
    # Wait, C_{i,k} = sum_r ...
    # C_{0,k} = (A_{0,0} B_{0,k} + A_{0,1} B_{1,k}) * 1.
    # This is correct!
    # C_{1,k} = (A_{1,0} B_{0,k} + A_{1,1} B_{1,k}) * 1.
    # This is correct!
    # So rank of 2x2x1 is 2.
    # My previous analysis was wrong because I thought the cross terms were a problem, but they aren't because the sums are separate for each i.
    
    # So we can reduce the rank of the 2x2x1 blocks from 4 to 2.
    # Previous rank: 4*7 + 2*4 = 28 + 8 = 36.
    # New rank: 4*7 + 2*2 = 28 + 4 = 32.
    # This is an improvement!
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L B_11
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R B_21
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L B_12
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R B_22
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L B_15 (rank 2)
    # C_{0,4} = A_{0,0} B_{0,4} + A_{0,1} B_{1,4}
    # C_{1,4} = A_{1,0} B_{0,4} + A_{1,1} B_{1,4}
    
    # Term 1 for i=0
    U[u_idx(0,0)][col] = 1
    U[u_idx(0,1)][col] = 1
    V[v_idx(0,4)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    # Term 2 for i=1
    U[u_idx(1,0)][col] = 1
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,4)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Block 6: A_R B_25 (rank 2)
    # C_{0,4} += A_{0,2} B_{2,4} + A_{0,3} B_{3,4}
    # C_{1,4} += A_{1,2} B_{2,4} + A_{1,3} B_{3,4}
    
    # Term 3 for i=0
    U[u_idx(0,2)][col] = 1
    U[u_idx(0,3)][col] = 1
    V[v_idx(2,4)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # Term 4 for i=1
    U[u_idx(1,2)][col] = 1
    U[u_idx(1,3)][col] = 1
    V[v_idx(2,4)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
