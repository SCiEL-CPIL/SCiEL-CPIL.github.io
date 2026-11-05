"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Matmul tensor: T[i,j,k] = 1 for all i,j,k
    # Factorized: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i*m + j]
                vj = V[j*p + k]
                wk = W[k*n + i]
                val = sum(ui[c] * vj[c] * wk[c] for c in range(len(ui)))
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_rank(n, m, p, target_rank, time_limit):
    """Try to find a decomposition of the given rank using integer/half-integer entries."""
    import numpy as np
    deadline = time.time() + time_limit
    
    # For matrix multiplication n x m by m x p, the tensor has entries:
    # T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # The schoolbook rank is n*m*p = 36.
    # Known bounds: For 2x3 * 3x6, Strassen-like algorithms may help.
    # Let's try to find lower rank decompositions.
    
    # One approach: use the fact that for small matrices, we can try
    # structured decompositions.
    
    # The matmul tensor for (n,m,p) is a 3-tensor of size n*m*p.
    # We seek rank-r decomposition.
    
    # For r < n*m*p, we need non-trivial structure.
    
    # Known: rank of 2x2*2x2 matmul is 7 (Strassen).
    # For 2x3*3x6: Let's think about block structure.
    
    # A 2x3*3x6 matrix can be seen as multiplying A(2x3) by B(3x6).
    # The result is a 2x6 matrix.
    
    # Let me try to use Strassen's idea recursively or partially.
    
    # Actually, let me try a more direct approach: 
    # For the tensor T of shape (n*m, m*p, p*n) = (6, 18, 12),
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k.
    
    # The "support" structure: T[a,b,c]=1 iff the indices are consistent.
    
    # For rank r, we need r terms. Let's try to find small-r decompositions.
    
    # Heuristic: For n=2, m=3, p=6, a known bound from literature suggests
    # rank is at most some value. Let me try brute-force search over 
    # small structured families.
    
    # Family 1: Block-diagonal Strassen-like.
    # Split the m=3 dimension into m1=1, m2=2 (or m1=2, m2=1).
    # And p=6 into p1=3, p2=3.
    
    # Let's try: decompose into two blocks.
    # Block 1: A[:, :m1] * B[:m1, :] gives A_left (2x1) * B_left (1x6) -> 2x6
    # Block 2: A[:, m1:] * B[m1:, :] gives A_right (2x2) * B_right (2x6) -> 2x6
    
    # For 2x1 * 1x6: this is outer product, rank 1 per output element? 
    # Actually it's just scaling: result[i, k] = A[i, 0] * B[0, k]. Rank = 2*1*1 = 2 for the tensor? No.
    # The tensor for 2x1 * 1x6: T[i,k] = A[i,0]*B[0,k]. This has rank 1 (as a 2-tensor) but as part of 3-tensor...
    
    # Let me reconsider. The tensor for n*m*p matrix mult:
    # T is a tensor of order 3 with dimensions (n, m, p) where T[i,j,k] is 1 
    # and we factorize the bilinear map.
    
    # Actually the matmul tensor M_{n,m,p} has entries indexed by (i,j,k) where
    # M_{i,j,k} = 1, and we want rank r such that
    # M = sum_c u^{(c)} ⊗ v^{(c)} ⊗ w^{(c)}
    # where u has length n*m, v has length m*p, w has length p*n.
    
    # Hmm, let me just try to use a simple heuristic search.
    
    # Try rank from max(n*p, m*p, n*m) up to n*m*p:
    min_rank = max(n*p, m*p, n*m)  # = max(12, 18, 6) = 18
    
    # For each candidate rank, try to see if a decomposition exists.
    # This is hard in general. Let me try a specific structured approach.
    
    # Approach: Use the fact that the matmul tensor has a lot of zeros in a 
    # certain "combinatorial" structure and exploit it.
    
    # Actually, for this specific case (2,3,6), let me try to see if we can
    # achieve rank 18 (the minimum possible given the dimensions).
    
    # The matmul tensor M_{2,3,6}: 
    # u-space: 6 dims (i=0,1; j=0,1,2)
    # v-space: 18 dims (j=0,1,2; k=0..5)
    # w-space: 12 dims (k=0..5; i=0,1)
    
    # T[i,j,k] = u_{i,j} ⊗ v_{j,k} ⊗ w_{k,i}
    
    # For a rank-r decomposition, we need r rank-1 tensors whose sum equals T.
    
    # Let me try: can we do it with r=18?
    # Lower bound: The tensor is nonzero in all 2*3*6=36 entries of the (i,j,k) tensor.
    # But the rank could still be less than 36.
    
    # For the 2x2*2x2 case, rank is 7 < 8.
    
    # Let me try a concrete construction for (2,3,6):
    # 
    # Use the following idea: 
    # For the (i,j,k) entries, group them by (i,k) pair. For each (i,k),
    # there are m=3 values of j. The slice T[:, :, (i,k)] is a vector of length 
    # n*m * m*p restricted to specific entries.
    
    # Actually, let me just try to find a rank-18 or rank-20 decomposition 
    # through a structured ansatz.
    
    # Structured ansatz: Use the m=3 dimension cleverly.
    # 
    # Let me define for each (i, k): the "fiber" is over j=0,1,2.
    # T[i,j,k] = 1 for all valid (i,j,k).
    
    # A rank-1 term looks like: f(i,j) * g(j,k) * h(k,i)
    # and we need sum_c f_c(i,j) * g_c(j,k) * h_c(k,i) = 1 for all (i,j,k).
    
    # This is equivalent to: for each (i,k), sum_c f_c(i,j) * h_c(k,i) = 1/g_c(j,k)... no.
    # 
    # sum_c f_c(i,j) * g_c(j,k) * h_c(k,i) = 1  for all i,j,k.
    
    # Let me fix (i,k). Then: sum_c f_c(i,j) * h_c(k,i) * g_c(j,k) = 1 for all j.
    # 
    # Define for each (i,k) and j: this must equal 1.
    
    # This is a system of 36 equations (for 36 (i,j,k) triples) in r*36 variables 
    # (each rank-1 term has up to 6*18*12 = 1296 parameters but structured).
    
    # This is getting complex. Let me try a simpler approach: just return the 
    # schoolbook and verify it works, or try a slightly better construction.
    
    # Known: For (2,3,6), I'll try rank 24 by using Strassen on the 2x3 -> 2x3 part.
    
    # Actually, let me try a completely different approach. Let me see if I can
    # find a rank-27 decomposition or similar.
    
    # For now, let me implement a small search: try ranks from min_rank to 36,
    # and for each, try a few structured random initializations and see if we 
    # can solve.
    
    # Given time constraints, let me just try the schoolbook rank and verify.
    # But wait - maybe there's a known better rank for (2,3,6).
    
    # From the literature: R(2,3,6) <= ? 
    # A(2,3) * A(3,6) -> the rank is bounded by various methods.
    
    # Let me just try to construct one with rank 24 or so.
    
    # Simple approach: treat the m=3 as m1=1 + m2=2.
    # For the 2x1 * 1x6 part: we need 2*1*6 = 12 terms? No, it's simpler.
    # A(2x1) * B(1x6): result C(i,k) = A(i,0)*B(0,k). 
    # This bilinear map has tensor rank 1! Because it's just A(i,0)*B(0,k).
    # So u = e_i ⊗ e_0, v = e_0 ⊗ e_k, w = e_k ⊗ e_i. 
    # Wait, that's not right. Let me be more careful.
    
    # For A(2x1) * B(1x6): C[i,k] = sum_{j=0}^{0} A[i,j]*B[j,k] = A[i,0]*B[0,k].
    # The tensor is T[i, j=0, k] = 1 (only j=0 exists).
    # So T has only 2*1*6 = 12 nonzero entries (one per (i,k) with j=0).
    # Actually T[i,0,k] = 1 for i=0,1 and k=0..5. That's 12 entries.
    # Rank of this 3-tensor (dimensions 2, 1, 6): since the middle dimension is 1,
    # T[i,0,k] = a_i * b_k where a_i=1, b_k=1. So it factors as (1,1) ⊗ (1) ⊗ (1,1,1,1,1,1)? 
    # No: T[i,0,k] = u_i * v_0 * w_k. We need u_i*v_0*w_k = 1 for all i,k.
    # If v_0 = 1, then u_i*w_k = 1 for all i,k. This has rank 1!
    # u = (1,1), v = (1), w = (1,1,1,1,1,1). So rank 1 for the (2,1,6) block.
    
    # For the 2x2 * 2x6 part (m2=2, p2=6): 
    # This is a 2x2 * 2x6 matmul. 
    # T[i,j,k] for i=0,1, j=0,1, k=0..5.
    # Can we use Strassen-like on the 2x2 part?
    # For 2x2 * 2x2, Strassen gives rank 7. For 2x2 * 2x6:
    # The 2x2*2x2 part uses 7 multiplications. For each of the 6 output columns in the k=0..5 range...
    # Hmm, 2x2 * 2x6: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # This is like 2 independent 2x1 * 1x6 matmuls? No.
    # T[i,j,k] = 1 for all i,j,k in the sub-block.
    # 
    # Let me think: 2x2 * 2x6 tensor. Dimensions: u has 4, v has 12, w has 12.
    # T is nonzero for all (i,j,k) in the 2*2*6 = 24 combinations.
    # 
    # One approach: for each pair (j,k), we have contributions from i=0,1.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    
    # For fixed (j,k): T[0,j,k] = sum_c U[0,j,c] V[j,k,c] W[k,0,c] = 1
    #               T[1,j,k] = sum_c U[1,j,c] V[j,k,c] W[k,1,c] = 1
    
    # This is getting complex. Let me just try a direct small-rank search.
    
    # Given the complexity, let me try rank 18 first (the dimension-based lower bound).
    # 
    # Actually, I realize I should just try a few specific ranks and use a 
    # simple verification. Let me start with the schoolbook (rank 36) and 
    # try to improve.
    
    # For the (2,3,6) case, let me try to use a decomposition based on:
    # Split p=6 into two groups of 3: p1=3, p2=3.
    # 
    # Block 1: 2x3 * 3x3 -> 2x3
    # Block 2: 2x3 * 3x3 -> 2x3
    # 
    # Each 2x3*3x3 matmul: what's its rank?
    # For 2x3*3x3: tensor dimensions (6, 9, 9) in (u,v,w) space.
    # T has 18 nonzero entries (2*3*3).
    # Lower bound: max(6, 9, 9) = 9.
    # 
    # Hmm, this doesn't obviously help.
    
    # Let me try yet another approach. For the specific dimensions (2,3,6):
    # 
    # I'll try to find a rank-r decomposition by treating it as a system of 
    # polynomial equations and using a small search.
    
    # Actually, the most practical thing: let me just verify the schoolbook 
    # and return it. The score will be 36/36 = 1.0. To improve, I need rank < 36.
    
    # Let me try a specific known construction.
    # 
    # For multiplying 2x3 by 3x6:
    # Write A = [A1 | A2] where A1 is 2x1, A2 is 2x2.
    # Write B = [B1; B2] where B1 is 1x6, B2 is 2x6.
    # C = A1*B1 + A2*B2.
    
    # A1*B1: 2x1 * 1x6 -> rank 1 tensor (as shown above).
    #   U: u_{i,0} for i=0,1 (the A indices), 0 for i=2..5 (but we don't have those)
    #   Actually in the u-space (i*m+j = i*3+j), for the A1 block (j=0):
    #   u_{3i+0} = 1 for i=0,1; all other u = 0.
    #   v-space (j*p+k = j*6+k), for B1 (j=0): v_{0*6+k} = v_k = 1 for k=0..5, rest 0.
    #   w-space (k*n+i = k*2+i), for the result: w_{2k+i} = 1 for all k,i.
    #   Check: sum = 1*1*1 = 1. Good. This uses 1 rank term.
    
    # A2*B2: 2x2 * 2x6.
    #   u-space: for j=1,2 (the A2 part): u_{3i+1} and u_{3i+2}.
    #   v-space: for j=1,2 (the B2 part): v_{6+ k} and v_{12+k}.
    #   w-space: same as before: w_{2k+i}.
    #   
    # For 2x2 * 2x6: T[i, j', k] = 1 where i=0,1, j'=0,1 (local), k=0..5.
    #   In global indices: i=0,1, j=1,2, k=0..5.
    #   
    # Can I decompose 2x2 * 2x6 with low rank?
    #   The tensor is T[i,j',k] = 1 for i=0,1, j'=0,1, k=0..5.
    #   This is a 2x2x6 tensor (all ones).
    #   Its tensor rank: for an all-ones 2x2x6 tensor, rank = ?
    #   All-ones tensor of size a x b x c has tensor rank max(a,b,c) if we allow real entries? 
    #   No! The rank of the all-ones tensor of order 3 and size a×b×c is max(a,b,c) only if... 
    #   Actually, the tensor rank of the all-ones 2×2×2 tensor is 1? No!
    #   All-ones 2×2×2: T[i,j,k]=1. Can we write as sum of rank-1 tensors?
    #   T = u⊗v⊗w + ... The all-ones 2×2×2 has rank 1 iff it's separable, which it's not 
    #   in the 3-tensor sense? Wait, (1,1)⊗(1,1)⊗(1,1) = all ones! So rank 1!
    #   
    #   YES! The all-ones 2×2×6 tensor = (1,1,1) ⊗ (1,1) ⊗ (1,1,1,1,1,1) which is rank 1!
    #   Wait, that's u⊗v⊗w where u has length 2, v has length 2, w has length 6.
    #   u⊗v⊗w[i,j,k] = u_i * v_j * w_k = 1*1*1 = 1. Yes! Rank 1!
    
    # So A2*B2 (which is 2x2*2x6, all-ones tensor) has tensor rank 1!
    
    # Therefore: total rank = 1 (for A1*B1) + 1 (for A2*B2) = 2??
    # That can't be right. Let me check.
    
    # Wait, the issue is that the u, v, w spaces are shared. Let me be more careful.
    
    # For A1*B1 (j=0 part):
    #   u: u_{3i+0} = 1 for i=0,1 (positions 0, 3); u_{3i+1}=u_{3i+2}=0 for i=0,1 (positions 1,2,4,5)
    #   v: v_{0*6+k} = 1 for k=0..5 (positions 0..5); v_{j*6+k}=0 for j=1,2 (positions 6..17)
    #   w: w_{2k+i} = 1 for all k=0..5, i=0,1 (all 12 positions)
    #   Product: for (i,j=0,k): u_{3i}*v_k*w_{2k+i} = 1*1*1 = 1. ✓
    #   For (i,j=1,k) or (i,j=2,k): u_{3i+1}=0 or u_{3i+2}=0, so contribution = 0. ✓
    
    # For A2*B2 (j=1,2 part):
    #   u: u_{3i+1} = 1 and u_{3i+2} = 1 for i=0,1 (positions 1,2,4,5); u_{3i+0}=0 for i=0,1
    #   v: v_{(1)*6+k} = 1 and v_{(2)*6+k} = 1 for k=0..5 (positions 6..17); v_{0*6+k}=0
    #   w: w_{2k+i} = 1 for all k,i
    #   Product for (i,j=1,k): u_{3i+1} * v_{6+k} * w_{2k+i} = 1*1*1 = 1 ✓
    #   Product for (i,j=2,k): u_{3i+2} * v_{12+k} * w_{2k+i} = 1*1*1 = 1 ✓
    #   But wait: for (i,j=1,k), we get contribution 1. For (i,j=2,k), also 1.
    #   But in the original tensor, T[i,1,k]=1 and T[i,2,k]=1. So total from this term: 
    #   T[i,1,k] gets 1, T[i,2,k] gets 1. ✓
    #   But for (i,j=0,k): u_{3i+0}=0, so contribution = 0. ✓
    
    # BUT WAIT: The v-space term for A2*B2 has v_{6+k}=1 AND v_{12+k}=1 simultaneously.
    # So for a given (i,k) with j=1: v_{1*6+k}=v_{6+k}=1, contribution = u_{3i+1}*1*w_{2k+i}=1.
    # For j=2: v_{2*6+k}=v_{12+k}=1, contribution = u_{3i+2}*1*w_{2k+i}=1.
    # 
    # But these are different (i,j,k) triples, so they're fine. Each triple gets exactly 1.
    
    # So we have a rank-2 decomposition?! Let me double-check all 36 triples:
    # (i=0,j=0,k=0..5): A1*B1 term gives 1, A2*B2 gives 0. Total=1. ✓
    # (i=1,j=0,k=0..5): same. ✓
    # (i=0,j=1,k=0..5): A1*B1 gives 0, A2*B2 gives 1. Total=1. ✓
    # (i=0,j=2,k=0..5): A1*B1 gives 0, A2*B2 gives 1. Total=1. ✓
    # (i=1,j=1,k=0..5): same. ✓
    # (i=1,j=2,k=0..5): same. ✓
    
    # Wait, this seems too good to be true. Let me re-examine.
    # 
    # The problem: in the A2*B2 term, I have v_{6+k}=1 AND v_{12+k}=1 for the SAME rank term.
    # This means the v-vector has 12 nonzero entries (positions 6..17).
    # The u-vector has 4 nonzero entries (positions 1,2,4,5).
    # 
    # For the triple (i=0, j=1, k=0): 
    #   u_{0*3+1} = u_1 = 1, v_{1*6+0} = v_6 = 1, w_{0*2+0} = w_0 = 1. Product = 1. ✓
    # For the triple (i=0, j=2, k=0):
    #   u_{0*3+2} = u_2 = 1, v_{2*6+0} = v_12 = 1, w_{0*2+0} = w_0 = 1. Product = 1. ✓
    # 
    # These are DIFFERENT entries of the tensor, so there's no double-counting.
    # Each entry (i,j,k) gets contribution from term 1 (if j=0) or term 2 (if j=1 or j=2).
    # 
    # So the rank is indeed 2?? That seems impossibly low for a matrix multiplication.
    
    # OH WAIT. I think I'm making an error. Let me reconsider what the tensor represents.
    # 
    # The matmul tensor T_{n,m,p} is NOT the all-ones tensor. Let me re-read the problem.
    # 
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 6 matrix."
    # 
    # The matmul tensor T has entries T[i,j,k] where i indexes rows of A (and C), 
    # j indexes columns of A / rows of B, k indexes columns of B (and C).
    # T[i,j,k] = 1 for all valid (i,j,k).
    # 
    # And we factorize as: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has dimensions (n*m) x rank = 6 x rank,
    #       V has dimensions (m*p) x rank = 18 x rank,  
    #       W has dimensions (p*n) x rank = 12 x rank.
    # 
    # So T is indeed the all-ones 2×3×6 tensor (indexed by i∈{0,1}, j∈{0,1,2}, k∈{0,...,5}).
    # 
    # And I showed that the all-ones 2×3×6 tensor = (1,1)⊗(1,1,1)⊗(1,1,1,1,1,1) has rank 1!
    # 
    # So the matmul tensor for 2×3*3×6 has tensor rank 1???
    # 
    # That can't be right because then we could do any matrix multiplication with 1 scalar multiplication,
    # which is clearly false.
    
    # Let me re-examine. The issue might be in how the factorization works.
    # 
    # The standard matmul tensor factorization:
    # C = A*B where C[i,k] = sum_j A[i,j] B[j,k].
    # 
    # The bilinear form: C[i,k] = sum_j A[i,j] B[j,k].
    # This is a 3-linear form? No, it's bilinear in A and B.
    # 
    # The tensor T is defined by: T(A,B) = C where C[i,k] = <A⊗B, T_{i,k}>.
    # Or equivalently, T is a tensor in (A-space)^* ⊗ (B-space)^* ⊗ (C-space)
    # where T_{(i,j),(j,k),(i,k)} = 1... 
    
    # Hmm, I think the correct formulation is:
    # The matmul map is a bilinear map: Mat_{n×m} × Mat_{m×p} -> Mat_{n×p}.
    # It can be represented as a tensor T ∈ (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ R^{n*p}
    # such that for A ∈ R^{n×m}, B ∈ R^{m×p}:
    #   (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #   = sum_{a=0}^{n*m-1} sum_{b=0}^{m*p-1} T[a,b, (i,k)] * A_flat[a] * B_flat[b]
    # 
    # where a = i*m+j, b = j*p+k, and (i,k) indexed by... 
    # 
    # The output C[i,k] is indexed by pair (i,k). If we flatten C to C_flat[c] where c = k*n+i (as in the problem), then:
    #   C_flat[c] = sum_{a,b} T[a,b,c] * A_flat[a] * B_flat[b]
    #   where c = k*n + i, a = i*m + j, b = j*p + k.
    #   T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for consistent (i,j,k), else 0.
    
    # So T is NOT the all-ones tensor! It's a sparse tensor!
    # T[a,b,c] = 1 only when the indices (i,j,k) extracted from a, b, c are consistent.
    # 
    # For a = i*m+j, b = j*p+k, c = k*n+i:
    #   i = a div m = a // m, j_local = a mod m = a % m
    #   j_from_b = b div p = b // p, k_from_b = b mod p = b % p
    #   k_from_c = c div n = c // n, i_from_c = c mod n = c % n
    #   
    #   T[a,b,c] = 1 iff (a%m == b//p) and (b%p == c//n) and (c%n == a//m)
    #   i.e., j_from_a = j_from_b, k_from_b = k_from_c, i_from_c = i_from_a.
    
    # So T is a very sparse tensor with exactly n*m*p = 36 nonzero entries (one for each (i,j,k) triple).
    # 
    # This is very different from the all-ones tensor! The all-ones 2×3×6 tensor has 
    # rank 1, but this sparse tensor has much higher rank.
    
    # OK so I was completely wrong. Let me redo this.
    
    # The matmul tensor T ∈ R^{6 × 18 × 12} (or rather (6,18,12) with the W indexing being (p,n) = (6,2))
    # has T[a,b,c] = 1 iff the index consistency conditions hold.
    
    # For each valid (i,j,k): a = 3i+j, b = 6j+k, c = 2k+i.
    # T[3i+j, 6j+k, 2k+i] = 1.
    
    # We want: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is the tensor rank factorization. The rank is the minimum r such that
    # we can find U (6×r), V (18×r), W (12×r) with T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
    
    # The schoolbook uses r=36 (one term per (i,j,k)).
    # Can we do better?
    
    # For the 2×2×2 matmul: T[2i+j, 2j+k, 2k+i] = 1. This has rank 7 (Strassen).
    # For 2×3×6: let's think about lower bounds.
    
    # Lower bound 1: The tensor has 36 nonzero entries, but that doesn't directly give a rank lower bound.
    # Lower bound 2: Slice arguments. Fix c=0 (k=0,i=0): T[a,b,0] = 1 iff 3i+j=a, 6j+k=b, 2*0+0=0 → k=0, i=0.
    #   So T[3*0+j, 6j+0, 0] = T[j, 6j, 0] = 1 for j=0,1,2.
    #   The slice T[:,:,0] has entries (j, 6j) for j=0,1,2: (0,0), (1,6), (2,12).
    #   This slice as a matrix (6×18) has rank 3 (three 1's in different rows/columns).
    #   Tensor rank >= max slice rank >= 3. Not very tight.
    
    # For Strassen on 2×2: the tensor has 8 nonzero entries and rank 7.
    # The "improvement" comes from sharing structure.
    
    # For 2×3×6: One approach is to embed Strassen.
    # 
    # Let me try: use Strassen for the 2×2 part of m=3.
    # Split j: j1=j%2 (j=0,2 map to local 0,1) and j2=j//2 (j=0,1 map to 0, j=2 maps to 1).
    # Hmm, that's messy.
    
    # Alternative: Let me just try to verify the schoolbook and see if I can find
    # a lower-rank solution through a small search.
    
    # Given time constraints, let me implement the schoolbook (rank 36) and also 
    # try a couple of structured lower-rank guesses.
    
    # Actually, let me try rank 32. For the 2×2×2 sub-problems embedded in 2×3×6:
    # 
    # Think of it as: for each pair of "strassen blocks".
    # 
    # This is getting very involved. Let me just return the schoolbook for now.
    # The key insight from my analysis: the tensor is NOT all-ones, it's sparse.
    
    return None  # Signal that this attempt didn't find a solution


def construct():
    # Schoolbook decomposition of rank n*m*p
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
