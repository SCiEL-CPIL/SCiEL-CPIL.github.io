"""Kronecker-factorized rank-20 decomposition of the 2x3 x 3x8 matrix product."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 150 - 15  # 15s safety margin

    n, m, p = DIMENSIONS
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
    R = 20

    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)

    # ---------- Rank-5 scheme for 2x1 @ 1x4 (output C is 2x4) ----------
    # C[i,k] = sum_r U5[i,r] * V5[k,r] * W5[i*2+k, r]
    # Verified rank-5 decomposition (classic construction):
    # Columns 0..2 correspond to rank-3 scheme for 2x1 @ 1x2 on C[:, 0..1]
    # Columns 3..4 correspond to 1x1 @ 1x2 on C[0,2], C[1,2], C[1,3] with overlap
    # We use a standard explicit rank-5 for 2x1 x 1x4:
    # Factor rows:
    #  U5[i,r]: i in {0,1}, r in {0..4}
    #  V5[k,r]: k in {0..3}, r in {0..4}
    #  W5[i*2+k, r]

    U5 = np.array([
        [1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1],
    ], dtype=np.int8)

    V5 = np.array([
        [0, 1, 1, 0, 1],   # k=0
        [0, 0, 0, 0, 0],   # k=1  (handled by row 0 overlap; k=1 contributes via r=0? no)
        [0, 0, 0, 1, 0],   # k=2
        [0, 0, 0, 0, 1],   # k=3
    ], dtype=np.int8)

    W5 = np.array([
        # i=0: k=0..3
        [0, 1, 0, 0, 0],   # (i=0,k=0)
        [0, 1, 0, 0, 0],   # (i=0,k=1)
        [0, 1, 1, 1, 0],   # (i=0,k=2)
        [0, 1, 1, 1, 0],   # (i=0,k=3)
        # i=1: k=0..3
        [0, 0, 1, 1, 0],   # (i=1,k=0)
        [0, 0, 1, 1, 0],   # (i=1,k=1)
        [0, 0, 1, 1, 0],   # (i=1,k=2)
        [0, 0, 1, 1, 0],   # (i=1,k=3)
    ], dtype=np.int8)

    # Verify U5: V5: W5 gives correct 2x1 @ 1x4
    # C[i,k] = a_i * b_k (since A is 2x1, B is 1x4, C[i,k] = a_i * b_k)
    # Product tensor: P[i,k,r] = U5[i,r]*V5[k,r]*W5[i*2+k,r] should = 1 iff... 
    # Actually for matrix multiplication 2x1 @ 1x4: the tensor T[i,k,j] = delta where 
    # the "middle" dimension is size 1. So T[i,k,0] = 1 for all i,k (it's the identity in the middle).
    # Wait - the tensor for multiplying A (2x1) by B (1x4) has dimensions (n,m,p) = (2,1,4)
    # T[i,j,k] = 1 if ... no. The matmul tensor T[i,j,k] is defined as T[i,j,k] = delta (the 
    # structural tensor). For 2x1 @ 1x4: T[i,j,k] = 1 for all (i,j,k) with j=0 only, 
    # i.e. T[i,0,k] = 1 for all i,k.
    # A rank-r decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Here j is only 0, so we need: sum_r U[i,r]*V[0,r]*W[k,r] = 1 for all i in {0,1}, k in {0..3}
    
    # Let me just construct the factors properly:
    # We need sum_r U[i,r]*W[k,r] = 1 for all i,k (with V[0,r] folded in, set V=I or just absorb)
    # Since m=1, V[j,r] is just one row. Let's set V[0,r] = 1 for all r, then need
    # U[i,r]*W[k,r] summed over r = 1.
    # Simple: use r=0 only: U[i,0]=1, W[k,0]=1 => rank 1? No, that gives C[i,k]=a_i*b_k which IS correct.
    # Wait, 2x1 @ 1x4: C[i,k] = a_i * b_k. The tensor is T[i,0,k] = 1. 
    # Rank-1: U[i,0]=1, V[0,0]=1, W[k,0]=1. That works! Rank is 1!
    # 
    # OH WAIT. I confused myself. Let me re-examine.
    # The matmul tensor for n x m @ m x p has T[i,j,k] = 1 if the "path" i->j->k exists.
    # For 2x1 @ 1x4: T[i,j,k] where i in [0,2), j in [0,1), k in [0,4).
    # T[i,j,k] = 1 for all valid (i,j,k) because the only connection is i->0->k for all i,k.
    # So T has 2*1*4 = 8 ones.
    # Rank 1 decomposition: U[i,0]=1, V[0,0]=1, W[k,0]=1. Product: 1*1*1 = 1 for all (i,0,k). CORRECT!
    # 
    # So 2x1 @ 1x4 has rank 1! Not 5. I was wrong about that.
    # 
    # Similarly 2x2 @ 2x2 has rank 7 (Strassen).
    # 2x2 @ 2x4 has rank... let me think.
    
    # Let me restart the analysis with correct ranks.
    # 2x1 @ 1x4: rank 1 (trivial)
    # 2x2 @ 2x2: rank 7 (Strassen)
    # 2x2 @ 2x4: can factor as 2x2 @ 2x2 @ 2x2? No, as (2x2 @ 2x2) then... 
    #   Actually 2x2 @ 2x4 = (A @ B) where A is 2x2, B is 2x4. 
    #   B = [B_left | B_right] each 2x2. 
    #   A @ B = (A @ B_left, A @ B_right). Two independent 2x2@2x2 products => rank 14.
    #   But can we do better? 
    #   Alternatively factor the 4 output columns. 
    #   2x2 @ 2x4: tensor has 2*2*4 = 16 ones. Rank at most 16.
    #   Using the block structure: rank(2x2@2x4) <= rank(2x2@2x2) * rank(2x2@2x2)? No, that's not right.
    #   
    #   Actually, 2x2 @ 2x4: think of it as multiplying a 2x2 matrix by a 2x4 matrix.
    #   We can write: C = A @ B. If we factor B = B1 @ J where B1 is 2x2 and J is 2x4:
    #   Then C = (A @ B1) @ J. 
    #   But (A@B1) is 2x2, then @ J (2x4) gives 2x4.
    #   This is two separate matmuls, not a single tensor factorization directly.
    #   
    #   The tensor for 2x2 @ 2x4: T[i,j,k] = 1, i in {0,1}, j in {0,1}, k in {0..3}.
    #   Total 8 ones (2*2*4 = 16 entries but only 8 are 1? No, all are 1).
    #   Wait no. For matmul n x m @ m x p, T[i,j,k] = 1 for ALL (i,j,k) in the valid range.
    #   So for 2x2 @ 2x4: T has 2*2*4 = 16 ones. Rank <= 16.
    #   
    #   Can we factor this? 
    #   Consider: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    #   If we let J = kron(I_2, C) where C is some 2x2 matrix, then 2x2 @ 2x4 via:
    #   C_out = (A @ B1) @ (B1_inv... no this is getting circular.
    
    # Let me take a step back and think about the overall structure differently.
    
    # The 3x8 part: we have A (2x3) @ B (3x8) = C (2x8).
    # Write A = [A1 (2x2) | A2 (2x1)], B = [B1 (2x8); B2 (1x8)].
    # C = A1 @ B1 + A2 @ B2.
    
    # A2 @ B2: 2x1 @ 1x8. Tensor T2[i,j,k], i in {0,1}, j=0, k in {0..7}.
    # All 2*1*8 = 16 entries are 1. Rank = 1 (as shown above).
    # U2[i,0]=1, V2[0,0]=1, W2[k,0]=1.
    
    # A1 @ B1: 2x2 @ 2x8. Tensor T1[i,j,k], i in {0,1}, j in {0,1}, k in {0..7}.
    # 2*2*8 = 32 entries, all 1.
    # Factor B1 = B1a @ J where B1a is 2x4, J is 4x8 = kron(I_2, C_2x4).
    # Then A1 @ B1 = (A1 @ B1a) @ J.
    # (A1 @ B1a) is 2x4. Call it D = A1 @ B1a. Then C1 = D @ J = D @ kron(I_2, C).
    # C1[i,k] where k = l*2 + t (l in {0,1} selects which 2x4 block, t in {0,1} is position within).
    # C1[i, l*2+t] = sum_d D[i,d] * J[d, l*2+t] = sum_d D[i,d] * I_2[d//2, l] * C[d%2, t]
    #               = sum_s D[i, l*2+s] * C[s, t]  (using d = l*2+s, I_2 picks d//2=l)
    
    # So C1 = [D_left @ C, D_right @ C] where D_left = D[:,0:2], D_right = D[:,2:4].
    # This is two 2x2 @ 2x2 matmuls plus the initial 2x2 @ 2x4 matmul.
    # Total tensor: (2x2 @ 2x4) followed by two (2x2 @ 2x2).
    # Rank of 2x2 @ 2x4: let's figure this out.
    # Rank of 2x2 @ 2x2 = 7 (Strassen).
    # 
    # For 2x2 @ 2x4: T[i,j,k], i,j in {0,1}, k in {0..3}, all = 1.
    # This is equivalent to two 2x2 @ 2x2 matmuls sharing the same input A.
    # T[i,j,k] = T1[i,j,k mod 2] * (1 if k//2 matches...) 
    # Actually, just view it as: for fixed j, the row T[i,j,:] for i in {0,1} is a 2x4 matrix of all 1s.
    # 
    # Let's think of 2x2 @ 2x4 as: A (2x2) times B (2x4) = C (2x4).
    # Split B = [B_a (2x2) | B_b (2x2)]. C = [A@B_a | A@B_b].
    # These are two 2x2@2x2 products. They share the same A input.
    # In tensor decomposition, sharing input means we can potentially share U factors.
    # T[i,j,k] for k in {0,1} is the "left" 2x2@2x2, for k in {2,3} is the "right".
    # 
    # A rank-r decomposition of 2x2@2x2: exists with r=7.
    # For 2x2@2x4: we need T[i,j,k] = sum_r U[i,r]V[j,r]W[k,r] = 1 for all i,j,k.
    # 
    # One approach: use rank 7 for each of the two 2x2@2x2 sub-problems, but since they share
    # U and V, maybe we can do better. Actually no - the W factors differ for the two sub-problems.
    # 
    # Upper bound: rank(2x2 @ 2x4) <= rank(2x2 @ 2x2) + rank(2x2 @ 2x2) = 14 (trivial bound).
    # Better: since it's two 2x2@2x2 sharing U,V, the rank is at most 7*2 = 14.
    # Can we do it in fewer? Probably not significantly.
    
    # OK here's a cleaner approach. Let me just use:
    # 2x2 @ 2x4 has rank at most 14 (two Strassen 2x2@2x2).
    # But actually, 2x2 @ 2x4: the tensor is T[i,j,k] = 1, dimensions (2,2,4).
    # This is a "rectangular" tensor. Known: rank(2x2 @ 2x4) = rank(2x2@2x2) + rank(1x1@1x... ) 
    # No. Let me just verify: can 2x2 @ 2x4 be done in rank < 14?
    # 
    # Actually, I recall that rank(2 x m @ m x n) for fixed 2 is well-studied.
    # rank(2x2@2x2) = 7.
    # rank(2x2@2x3): ? 
    # rank(2x2@2x4): ?
    #
    # For 2x2 @ 2x4, one can show rank <= 7 + 4 = 11 or something? Not sure.
    #
    # Let me just go with a concrete construction and verify.
    # 
    # Strategy: 
    # C = A1 @ B1 + A2 @ B2
    # A2 @ B2: rank 1 (trivial)
    # A1 @ B1: 2x2 @ 2x8. 
    #   B1 (2x8) = B1a (2x4) @ J (4x8) where J = kron(I_2, C_2x2)
    #   Wait, let me pick J = kron(I_2, C) where C is a 2x2 matrix of rank 2.
    #   Then 4x8 = 2x2 kron 2x4? No, I_2 is 2x2, C is 2x2, kron gives 4x4. I need 4x8.
    #   
    # Let me redo: B1 is 2x8. Write B1 = B1a (2x4) @ J (4x8).
    # J should be 4x8. Let J = kron(I_2, J2) where J2 is 2x4. Then J is 4x8. 
    # J[i*2+a, j] = I_2[i,i'] * J2[a,j]... actually kron(I_2, J2)[i*2+a, j*1+b]... 
    # J is 4x8, so J = kron(I_2 (2x2), J2 (2x4)) gives (2*2) x (2*4) = 4x8. Correct!
    # J[row, col]: row = i*2 + a (i in {0,1}, a in {0,1}), col = b*1 + c... 
    # Actually kron(A (p x q), B (r x s)) gives (p*r) x (q*s).
    # kron(I_2 (2x2), J2 (2x4)): (2*2) x (2*4) = 4 x 8. Yes.
    # J[(i*2+a), (b*4+c)]... no. 
    # kron(A,B)[i*p+r, j*q+s] = A[i,j]*B[r,s] where A is p x q, B is r x s.
    # Here A = I_2 (p=2, q=2), B = J2 (r=2, s=4).
    # J[i*2+r, j*4+s] = I_2[i,j] * J2[r,s]
    # J[row, col] where row = i*2+r, col = j*4+s.
    # For fixed i=j: J[i*2+r, i*4+s] = J2[r,s]. For i != j: 0.
    # So J is block-diagonal with J2 in positions (0,0) and (1,1) (each 2x4).
    # 
    # Then (A1 @ B1a) @ J: 
    # D = A1 @ B1a (2x4). 
    # C1 = D @ J (2x8).
    # C1[i, j*4+s] = sum_r D[i, r] * J[r, j*4+s] = sum_r D[i,r] * I_2[r//2, j] * J2[r%2, s]
    # = D[i, j*2] * J2[0,s] + D[i, j*2+1] * J2[1,s]
    # 
    # So C1[i, j*4+s] = D[i, 2j] * J2[0,s] + D[i, 2j+1] * J2[1,s]
    # This is C1[:, j*4 : j*4+4] = [D[i,2j], D[i,2j+1]] @ J2
    # i.e., each 2x4 block of C1 is a 1x2 @ 2x4 product (a linear combination of rows of J2).
    
    # So the full computation is:
    # 1. Compute D = A1 @ B1a (2x2 @ 2x4): rank R1
    # 2. For j=0,1: C1_block_j = row(D, 2j:2j+2) @ J2 (1x2 @ 2x4): each has some rank
    # 3. C2 = A2 @ B2 (2x1 @ 1x8): rank 1
    # 4. C = C1 + C2
    
    # Step 1: 2x2 @ 2x4. Let's just use rank 14 (two 2x2@2x2).
    #   But wait, can I factor it more cleverly?
    #   2x2 @ 2x4: T[i,j,k] = 1, (i,j) in {0,1}^2, k in {0..3}.
    #   Note T[i,j,k] = T[i,j,k mod 2] * 1_{k//2 = j'?} No, there's no such dependence.
    #   Actually all entries are 1, so it's just the "full" tensor of shape (2,2,4).
    #   
    #   rank(2,2,4) full tensor: this is the matmul tensor M_{2,2,4}.
    #   Known: M_{2,2,4} has rank at most 7*2 = 14 (two Strassen).
    #   Is it less? I believe rank(M_{2,2,4}) = 14 actually, but let me not assume.
    #   Let's use 14 to be safe. Actually, let me try to get it lower.
    
    #   Alternative: 2x2 @ 2x4 = 2x2 @ (2x2 @ 2x2)? No, 2x2 @ 2x2 is 2x2, not 2x4.
    #   2x2 @ 2x4: factor the 4 columns as 2 groups of 2.
    #   = [2x2 @ 2x2, 2x2 @ 2x2] => rank 7+7 = 14.
    
    # Step 2: 1x2 @ 2x4: T[i,j,k] with i=0, j in {0,1}, k in {0..3}. All 8 entries = 1.
    #   rank: need sum_r V[j,r] * W[k,r] = 1 for all j in {0,1}, k in {0..3}. (U[0,r] absorbed)
    #   This is the same as 2x4 all-ones. Can be done in rank 1: V[j,0]=1, W[k,0]=1.
    #   Wait no, that gives V[j,0]*W[k,0] = 1. Yes! Rank 1!
    #   But wait, that's for a 1x2 @ 2x4 where the result is 1x4. 
    #   T[j,k] = 1 for all j,k. U[0,r]=1, V[j,r], W[k,r]. sum_r V[j,r]*W[k,r] = 1.
    #   Rank 1: V[j,0]=1, W[k,0]=1. Yes, rank 1.
    
    # So step 2 adds rank 1 for each of the 2 blocks => rank 2 total.
    # But these two blocks use different U factors (D[i,2j] and D[i,2j+1]), so they need
    # separate V,W slots? Actually in the tensor decomposition of the overall product,
    # each "rank term" is one (U,V,W) triple. The two 1x2@2x4 operations are part of
    # the overall tensor and need their own rank terms.
    
    # Let me count total rank:
    # - 2x2 @ 2x4 (step 1): rank 14
    # - Two (1x2 @ 2x4) for J2 application (step 2): rank 1 + 1 = 2  
    # - 2x1 @ 1x8 (step 3): rank 1
    # Total: 14 + 2 + 1 = 17.
    
    # Hmm, but wait. The step 2 operations are "applied to the output" of step 1. 
    # In a tensor decomposition, you can't just chain them - you need to express the
    # entire computation as a single tensor product.
    # 
    # The issue is that C1[i, 2j+s... ] involves products like D[i,2j] * J2[0,s] * J2[1,s],
    # where D itself is a product involving U,V,W from step 1.
    # So the full rank term for C1[i, 2j+s] would be a product of 3 factors from step 1
    # times 2 more factors from step 2, giving a degree-5 product. But tensor decompositions
    # only allow products of 3 factors.
    # 
    # This means I can't just chain the decompositions. I need to think of it differently.
    # 
    # The correct way: the tensor for 2x2 @ 2x8 is T[i,j,k], i,j in {0,1}, k in {0..7}, all = 1.
    # I need a rank-r decomposition: T[i,j,k] = sum_{r} U[i,r] * V[j,r] * W[k,r].
    # 
    # If I factor the k-dimension: k = a*2 + b where a in {0,1} (block) and b in {0,1} (within block).
    # T[i,j,a*2+b] = 1 for all i,j,a,b.
    # 
    # Suppose I use a decomposition where W[k, r] encodes the (a,b) structure.
    # If I had W[a*2+b, r] = W_a[a, r1] * W_b[b, r2] (Kronecker in W), and similarly
    # for U,V... but U only has 2 rows, V only has 2 rows, so their Kronecker structure is limited.
    # 
    # Actually, the matmul tensor M_{2,2,8} can be factored as:
    # M_{2,2,8}[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # If I write k = 2a + b, and set W[2a+b, r] = W_a[a, r] where r indexes only the a-dimension...
    # No, W needs to map all 8 k-values to the same r-index space.
    # 
    # Let me try: use r = (r1, r2, r3) where:
    # - r1 indexes the 2x2@2x2 decomposition of the "a-block" selection
    # - r2 indexes something else
    # 
    # Actually, the cleanest is: M_{2,2,8} = M_{2,2,2} \otimes M_{1,1,4} in some sense?
    # M_{2,2,2}[i,j,a] * M_{1,1,4}[0,0,b] = 1*1 = 1 = M_{2,2,8}[i,j,2a+b].
    # But this is a Kronecker product of tensors: (M_{2,2,2} \otimes M_{1,1,4})
    # The rank of a Kronecker product of tensors is at most the product of ranks.
    # rank(M_{2,2,2}) = 7, rank(M_{1,1,4}) = 1 (since 1x1@1x4 has rank 1).
    # So rank(M_{2,2,8}) <= 7 * 1 = 7!
    
    # Wait, is M_{1,1,4} really rank 1? 
    # M_{1,1,4}[i,j,k] = 1, i=j=0, k in {0..3}. 
    # U[0,r]*V[0,r]*W[k,r] = 1. Set U[0,0]=1, V[0,0]=1, W[k,0]=1. Rank 1. Yes!
    # 
    # And the Kronecker: (A \otimes B)[(i1,i2),(j1,j2),(k1,k2)] = A[i1,j1,k1]*B[i2,j2,k2]
    # For M_{2,2,8}: shape is (2,2,8) = (2,2,2*4) = ((2,2,2),4)... 
    # Actually M_{2,2,8} has shape (2, 2, 8). If I write 8 = 2*4:
    # M_{2,2,8}[i,j,2a+b] = 1 = M_{2,2,2}[i,j,a] * M_{1,1,4}[0,0,b]
    # But M_{2,2,2} \otimes M_{1,1,4} has shape (2*1, 2*1, 2*4) = (2,2,8). 
    # (M_{2,2,2} \otimes M_{1,1,4})[(i1*1+i2, j1*1+j2, k1*4+k2)] = M_{2,2,2}[i1,j1,k1] * M_{1,1,4}[i2,j2,k2]
    # = M_{2,2,2}[i1,j1,k1] (since M_{1,1,4}[0,0,k2] = 1 for all k2).
    # = 1 for all valid indices. 
    # And M_{2,2,8}[i,j,k] = 1 for all. So yes, M_{2,2,8} = M_{2,2,2} \otimes M_{1,1,4}.
    # Rank <= 7 * 1 = 7!
    
    # So rank(2x2 @ 2x8) <= 7! That's the same as 2x2 @ 2x2.
    
    # Let me verify: M_{2,2,8}[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # where U,V come from Strassen 2x2x2 (r in {0..6}), and W[k, r] = W_strassen[k % 2, r] (independent of k // 2).
    # Wait: M_{1,1,4}[0,0,k2] = 1 for all k2, so the "W" from the second factor is just 1 for all k.
    # In the Kronecker product decomposition:
    # U_kron[i, (r1, r2)] = U1[i, r1] * U2[0, r2] = U1[i, r1] * 1 = U1[i, r1]
    # But this means U doesn't depend on r2, which means we can just set r2 = 0 and get:
    # U[i, r1] = U1[i, r1], V[j, r1] = V1[j, r1], W[k, r1] = W1[k%2, r1] (since W2[k2, 0] = 1).
    # Actually W2 has shape (1,1,4), so W2[0, 0, k2] = 1. In the Kronecker, W_kron[k1, k2, r] = W1[k1, r1] * W2[k2, r2].
    # With r2 = 0 only: W_kron[k, r1] = W1[k % 2, r1] * W2[k // 2, 0] = W1[k % 2, r1] * 1.
    # So W[k, r1] = W1[k % 2, r1].
    # 
    # Verification: sum_r U[i,r]*V[j,r]*W[k,r] = sum_r1 U1[i,r1]*V1[j,r1]*W1[k%2, r1] = M_{2,2,2}[i,j,k%2] = 1. 
    # And M_{2,2,8}[i,j,k] = 1. Correct!
    
    # So rank(2x2 @ 2x8) = 7 (using Strassen for the 2x2@2x2 part, and the 1x1@1x4 is free).
    
    # Now for our problem: 2x3 @ 3x8.
    # C[i,k] = sum_j A[i,j] * B[j,k], i in {0,1}, j in {0,1,2}, k in {0..7}.
    # Split j: C[i,k] = (A1 @ B1)[i,k] + A2[i] * B2[k]
    # where A1 = A[:, 0:2] (2x2), B1 = B[0:2, :] (2x8), A2 = A[:, 2] (2x1), B2 = B[2, :] (1x8).
    
    # A1 @ B1: 2x2 @ 2x8, rank 7.
    # A2 * B2: outer product, 2x1 @ 1x8. Rank 1 (as shown).
    # Total rank: 7 + 1 = 8!
    
    # Let me construct this.
    # 
    # Strassen 2x2 @ 2x2:
    # Standard Strassen factors (rank 7):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0+M3-M5-M6
    # C01 = M2+M4
    # C10 = M1+M5
    # C11 = M0-M1+M2+M4
    # 
    # In tensor form: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For 2x2@2x2, the 7 rank terms:
    # r=0 (M0): U = [1,1] (A00,A11), V = [1,1] (B00,B11), W = [1,0,0,1] (C00,C01,C10,C11)
    # r=1 (M1): U = [0,1] (A10,A11), V = [1,0] (B00), W = [0,0,1,0] (C10)
    # r=2 (M2): U = [1,0] (A00), V = [0,1] (B01,B11->B01-B11), W = [0,1,0,0] (C01)
    # r=3 (M3): U = [0,1] (A11), V = [-1,1] (B10-B00), W = [1,0,0,0] (C00)
    # r=4 (M4): U = [1,1] (A00,A01), V = [0,1] (B11), W = [0,1,0,1] (C01,C11)
    # r=5 (M5): U = [-1,1] (A10-A00), V = [1,1] (B00,B01), W = [0,0,1,0] (C10)
    # r=6 (M6): U = [-1,1] (A01-A11), V = [1,1] (B10,B11), W = [1,0,0,0] (C00)
    
    # Wait, I need to be careful with the indexing. In the problem, U has shape (n*m, rank) = (6, rank)
    # where the first index is i*m + j = i*3 + j for n=2, m=3.
    # V has shape (m*p, rank) = (24, rank), index j*p + k = j*8 + k.
    # W has shape (p*n, rank) = (16, rank), index k*n + i = k*2 + i.
    
    # For the 2x2 @ 2x8 part (using j in {0,1}, k in {0..7}):
    # U indices: i*3 + j for i in {0,1}, j in {0,1} -> {0, 1, 3, 4}
    # V indices: j*8 + k for j in {0,1}, k in {0..7} -> {0..15}
    # W indices: k*2 + i for k in {0..7}, i in {0,1} -> {0..15}
    
    # Strassen for 2x2@2x2 applied to the "2x2" part of the 2x8:
    # The 2x8 has B as 2x8. For the Strassen factors, I need V[j, r] for j in {0,1}.
    # Since B1 is 2x8, and I'm applying the rank-7 2x2@2x2 decomposition with W[k, r] = W1[k%2, r]:
    # V[j, r] for the 2x2 part: V[j, r] = V_strassen_2x2_2x2[j, r]
    # But V needs to have entries for all 8 k-values. Since the decomposition is:
    # sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k in the 2x2@2x8 tensor
    # With W[k, r] = W_strassen[k % 2, r], I need:
    # sum_r U[i,r] * V[j,r] * W_strassen[k%2, r] = 1 for all i,j and k.
    # This holds for k%2 = 0 and k%2 = 1 independently, which means:
    # sum_r U[i,r] * V[j,r] * W_strassen[0, r] = 1 and sum_r U[i,r] * V[j,r] * W_strassen[1, r] = 1
    # These are exactly the two conditions of the 2x2@2x2 tensor (for output columns 0 and 1).
    # So I can set V[j, r] = V_strassen[j, r] (same as for 2x2@2x2), and W[k, r] = W_strassen[k%2, r].
    
    # Let me code this up.
    
    # Strassen 2x2@2x2 factors (U is 2x7, V is 2x7, W is 4x7):
    U_s = np.array([
        [ 1,  0,  1,  0,  1, -1, -1],  # i=0: A00, A11 contributions
        [ 1,  1,  0,  1,  1,  1,  1],  # i=1
    ])
    V_s = np.array([
        [ 1,  1,  0, -1,  0,  1,  1],  # j=0: B00, B01 contributions
        [ 1,  0,  1,  1,  1,  1,  1],  # j=1
    ])
    W_s = np.array([
        [ 1,  0,  0,  1,  0,  0,  1],  # k=0 (C00)
        [ 0,  0,  1,  0,  1,  0,  0],  # k=1 (C01)
        [ 0,  1,  0,  0,  0,  1,  0],  # k=2 (C10)
        [ 1,  0,  0,  0,  1,  0,  0],  # k=3 (C11)
    ])
    
    # Verify Strassen: T[i,j,k] = sum_r U_s[i,r]*V_s[j,r]*W_s[k,r] should = 1 for all i,j,k in 2x2x2.
    T_s = U_s[:, :, None] * V_s[None, :, :, None] * W_s[None, None, :, :]
    T_s = np.sum(T_s, axis=2)
    assert np.all(T_s == 1), f"Strassen verification failed: {T_s}"
    
    # For 2x2 @ 2x8:
    # U_28[i, r] = U_s[i, r] for i in {0,1}, r in {0..6}
    # V_28[j, r] = V_s[j, r] for j in {0,1}, r in {0..6}
    # W_28[k, r] = W_s[k % 2, r] for k in {0..7}, r in {0..6}
    
    # Place into the full U, V, W arrays for the 2x3 @ 3x8 problem:
    # U full shape: (6, R), index = i*3 + j
    # V full shape: (24, R), index = j*8 + k
    # W full shape: (16, R), index = k*2 + i
    
    # Part 1: 2x2 @ 2x8 (j in {0,1}, k in {0..7}), rank 7, placed in columns 0..6 of U,V,W
    for i in range(2):
        for r in range(7):
            U[i * 3 + 0, r] = U_s[i, r]
            U[i * 3 + 1, r] = U_s[i, r]
        # Wait, U_s[i, r] is the coefficient for A[i, j] in the r-th Strassen term.
        # In Strassen, U[i, r] represents the linear form in A. For the r-th term:
        # U_s[i, r] is the coefficient of A[i, j] ... no.
        # Actually in the tensor decomposition:
        # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # U[i, r] is associated with input A[i, j]? No!
        # U[i, r] is the r-th factor for the i-index of the first matrix.
        # The first matrix A is n x m = 2 x 2 (for the 2x2 part). 
        # So U[i, r] where i indexes rows of A.
        # V[j, r] where j indexes columns of A (and rows of B).
        # W[k, r] where k indexes columns of B.
        
        # So for the 2x2 @ 2x8:
        # A is 2x2, indexed by (i, j) with i in {0,1}, j in {0,1}.
        # U[i, r] = U_s[i, r] (coefficient for A[i, *] in the i-dimension)
        # V[j, r] = V_s[j, r] (coefficient for A[*, j] / B[j, *] in the j-dimension)
        # W[k, r] = W_s[k % 2, r] (coefficient for B[*, k] in the k-dimension)
    
    # Clear and fill properly:
    U[:, :] = 0
    V[:, :] = 0
    W[:, :] = 0
    
    # Columns 0..6: 2x2 @ 2x8 via Strassen
    for i in range(2):
        for r in range(7):
            U[i * 3 + 0, r] += U_s[i, r]  # j=0
            U[i * 3 + 1, r] += U_s[i, r]  # j=1
    # Wait, this is wrong. U_s[i, r] is the same for all j. But in the tensor,
    # U[i, r] is the r-th "slice" for the i-dimension. It's a vector in r-space for each i.
    # U_s has shape (2, 7): U_s[i, r] for i in {0,1}, r in {0..6}.
    # This is correct: U_s[i, r] is the coefficient in the r-th rank term for the i-th row of A.
    # So in the full U (shape 6 x 20), for the 2x2 part:
    # U_full[i*3 + j, r] should be... 
    # Oh wait. In the problem, U has shape (n*m, rank) where the index is i*m + j.
    # So U[i*3 + j, r] is the entry for (row i, col j) of A in the r-th term.
    # But in the standard tensor decomposition of matmul, U[i, r] is for the i-index (row of A),
    # V[j, r] is for the j-index (col of A = row of B), W[k, r] is for the k-index (col of B).
    # 
    # So U[i*m + j, r] in the problem = U[i, r] from the tensor * delta? No.
    # Actually, the problem's U matrix has dimensions (n*m) x rank, and the entry U[i*m + j, r]
    # is the "U-factor" for the (i,j) element of A in the r-th rank term.
    # 
    # In the standard formulation: the matmul tensor T is a 3-tensor of shape (n, m, p) with
    # T[i, j, k] = 1. A rank-r decomposition is T[i,j,k] = sum_{c=1}^r U[i, c] * V[j, c] * W[k, c].
    # The problem defines U with shape (n*m, rank) where U[i*m + j, c] corresponds to... 
    # Looking at the schoolbook: U[i*m + j, c] = 1 when c corresponds to (i,j,k). 
    # So U[i*m+j, c] is NOT U[i, c] from the tensor. It's a flattened version where
    # the first index of U in the tensor (i) is combined with j.
    # 
    # Re-reading the schoolbook: 
    # U[i*m+j][c] = 1, V[j*p+k][c] = 1, W[k*n+i][c] = 1 for c = (i,j,k).
    # So the "U" in the problem is indexed by (i, j) of the first matrix, "V" by (j, k), "W" by (k, i).
    # This means:
    # U[i*m+j, c] = the c-th factor for A[i, j]
    # V[j*p+k, c] = the c-th factor for B[j, k]  
    # W[k*n+i, c] = the c-th factor for C[k, i] (or equivalently C[i, k] with different indexing)
    # 
    # And the tensor identity is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k] = 1 for the matmul.
    # 
    # So in the standard tensor notation: the first tensor index is (i,j) flattened, the second is (j,k) flattened,
    # the third is (k,i) flattened. And T[(i,j), (j,k), (k,i)] = 1.
    
    # OK so for the 2x2 @ 2x8 part (j in {0,1}):
    # U[i*3+j, r] for the r-th rank term: this is the coefficient for A[i,j] in the r-th term.
    # In the standard 2x2@2x2 Strassen: U_std[i, r] * V_std[j, r] gives the product for A[i,j].
    # So U_full[i*3+j, r] = U_std[i, r] * V_std[j, r]? No!
    # 
    # The standard decomposition: T2x2[i,j,k] = sum_r U2[i,r]*V2[j,r]*W2[k,r]
    # The problem's decomposition: T[i,j,k] = sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c]
    # 
    # These are different! In the standard form, U depends on i, V on j, W on k.
    # In the problem's form, U depends on (i,j), V on (j,k), W on (k,i).
    # 
    # For the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # Product: 1*1*1 = 1. This works because each (i,j,k) triple has its own column c.
    # 
    # For a rank-r decomposition: 
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # 
    # This is a valid tensor decomposition where the three modes are:
    # Mode 1: (i, j) flattened, size n*m
    # Mode 2: (j, k) flattened, size m*p
    # Mode 3: (k, i) flattened, size p*n
    # And T[i,j,k] = 1 is the tensor to decompose.
    # 
    # Note this is DIFFERENT from the standard (n, m, p) tensor decomposition where
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r].
    # 
    # In the standard form, rank(2x2@2x2) = 7. But in the problem's form, the tensor is
    # T[(i,j), (j,k), (k,i)] = 1, which is a (n*m) x (m*p) x (p*n) tensor.
    # For n=m=p=2: 4 x 4 x 4 tensor with T[i,j, j,k, k,i] = 1 (16 ones out of 64).
    # 
    # The rank of this tensor in the problem's formulation may be different from the standard rank!
    # 
    # Let me check: for the schoolbook, rank = n*m*p = 48. 
    # For the standard 2x2@2x2 tensor, rank 7 means: T_std[i,j,k] = sum_{r=1}^7 U[i,r]*V[j,r]*W[k,r].
    # In the problem's form for 2x2@2x2: T[(i,j),(j,k),(k,i)] = sum_c U[i*2+j, c]*V[j*2+k, c]*W[k*2+i, c].
    # 
    # If I take the standard rank-7 decomposition: U_std[i,r]*V_std[j,r]*W_std[k,r] = T[i,j,k].
    # Can I express T[(i,j),(j,k),(k,i)] = 1 as sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c]?
    # 
    # From the standard: T[i,j,k] = sum_r U_std[i,r]*V_std[j,r]*W_std[k,r] = 1.
    # I need: sum_c U_prob[i*2+j, c]*V_prob[j*2+k, c]*W_prob[k*2+i, c] = 1.
    # 
    # If I set c = r and:
    # U_prob[i*2+j, r] = ? * U_std[i, r]
    # V_prob[j*2+k, r] = ? * V_std[j, r]  
    # W_prob[k*2+i, r] = ? * W_std[k, r]
    # Then U_prob * V_prob * W_prob = U_std[i,r]*V_std[j,r]*W_std[k,r] * (scaling factors).
    # If I set U_prob[i*2+j, r] = U_std[i, r], V_prob[j*2+k, r] = V_std[j, r], W_prob[k*2+i, r] = W_std[k, r],
    # then the product is U_std[i,r]*V_std[j,r]*W_std[k,r] and summing over r gives T[i,j,k] = 1. 
    # But this only works when the (i,j), (j,k), (k,i) are consistent (they always are since j,j,k,k,i are shared).
    # 
    # Wait, but U_prob[i*2+j, r] = U_std[i, r] means U_prob doesn't depend on j. And V_prob[j*2+k, r] = V_std[j, r] doesn't depend on k. And W_prob[k*2+i, r] = W_std[k, r] doesn't depend on i.
    # 
    # Let me verify: sum_r U_prob[i*2+j, r] * V_prob[j*2+k, r] * W_prob[k*2+i, r]
    # = sum_r U_std[i,r] * V_std[j,r] * W_std[k,r] = T[i,j,k] = 1. YES!
    # 
    # So the rank is the same! The problem's tensor for matmul has the same rank as the standard tensor.
    # rank(2x2@2x2) = 7 in both formulations.
    
    # Great, so my construction works:
    # For the 2x2 @ 2x8 part:
    # U_prob[i*3+j, r] = U_strassen[i, r] for i in {0,1}, j in {0,1}, r in {0..6}
    # V_prob[j*8+k, r] = V_strassen[j, r] for j in {0,1}, k in {0..7}, r in {0..6}
    # W_prob[k*2+i, r] = W_strassen[k%2, r] for k in {0..7}, i in {0,1}, r in {0..6}
    
    # Verify: sum_{r=0}^6 U_prob[i*3+j, r] * V_prob[j*8+k, r] * W_prob[k*2+i, r]
    # = sum_r U_s[i,r] * V_s[j,r] * W_s[k%2, r]
    # = T_2x2_2x2[i, j, k%2] = 1. And we need this = T[i,j,k] = 1 for the 2x2@2x8. Correct!
    
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U[i * 3 + j, r] = U_s[i, r]
        for k in range(8):
            for r in range(7):
                V[0 * 8 + k, r] = V_s[0, r]  # j=0
                V[1 * 8 + k, r] = V_s[1, r]  # j=1
        for k in range(8):
            for i2 in range(2):
                for r in range(7):
                    W[k * 2 + i2, r] = W_s[k % 2, r]
    
    # Part 2: A2 * B2 = A[:, 2] (2x1) @ B[2, :] (1x8), rank 1, column 7
    # U_prob[i*3+2, 7] = 1 for i in {0,1} (A[i,2] coefficient)
    # V_prob[2*8+k, 7] = 1 for k in {0..7} (B[2,k] coefficient)
    # W_prob[k*2+i, 7] = 1 for all k, i (C[i,k] coefficient)
    
    for i in range(2):
        U[i * 3 + 2, 7] = 1
    for k in range(8):
        V[2 * 8 + k, 7] = 1
    for k in range(8):
        for i in range(2):
            W[k * 2 + i, 7] = 1
    
    # Verify the full decomposition
    n, m, p = DIMENSIONS
    R = 8
    U_int = U[:, :R].astype(int)
    V_int = V[:, :R].astype(int)
    W_int = W[:, :R].astype(int)
    
    # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                if s != 1:
                    ok = False
                    print(f"FAIL: i={i}, j={j}, k={k}, sum={s}")
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fallback to schoolbook
        rank = n * m * p
        U_f = [[0] * rank for _ in range(n * m)]
        V_f = [[0] * rank for _ in range(m * p)]
        W_f = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][column] = 1
                    V_f[j * p + k][column] = 1
                    W_f[k * n + i][column] = 1
                    column += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    # Convert to lists
    return {
        "U": U_int.tolist(),
        "V": V_int.tolist(),
        "W": W_int.tolist(),
    }
# EVOLVE-BLOCK-END
