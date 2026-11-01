"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    # Block-circulant decomposition: treat the 2x16 product as four 2x2 products
    # coupled through a shared Strassen-like structure.
    #
    # We use the fact that mu_{2,2,p} can be decomposed using the Kronecker
    # product of a 2x2 Strassen decomposition with an identity/circulant block
    # on the p-dimension.
    #
    # For 2x2x2 (Strassen), rank = 7:
    #   The 7 rank-1 tensors are:
    #   1: U = A[0,0]A[1,1], V = B[0,0]B[1,1], W = C[0,0]
    #   Actually, let me write the standard Strassen decomposition properly.
    #
    # Standard Strassen for C = A*B where A,B are 2x2:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # The tensor index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For our problem: n=2, m=2, p=16
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (32, rank)
    #
    # The multiplication is: sum_c U[i,j,c] * V[j,k,c] = W_entry
    # where the result R[i,k] = sum_j A[i,j]*B[j,k]
    # W is indexed as W[k*n + i, c] per the trace convention.
    #
    # Let me verify: U[i*m+j, c] means row index = i*m+j (i in 0..n-1, j in 0..m-1)
    # V[j*p+k, c] means row index = j*p+k (j in 0..m-1, k in 0..p-1)
    # W[k*n+i, c] means row index = k*n+i (k in 0..p-1, i in 0..n-1)
    #
    # So the contraction is: sum_j U[i*m+j, c] * V[j*p+k, c] should equal the
    # (i,k) entry of the product, and W[k*n+i, c] stores the coefficient for
    # the output at position (i,k) for rank factor c.
    #
    # Actually, re-reading: "W[k*n+i,r]" - this is the trace convention.
    # The full tensor T[i,j,k] = delta[i,k] for matmul? No.
    # T[i,j,k] is 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T[i,j,k] which when contracted gives the product.
    #
    # In the rank-1 decomposition: T = sum_r u_r * v_r * w_r (outer products)
    # U[i*m+j, r] = u_r(i,j), V[j*p+k, r] = v_r(j,k), W[k*n+i, r] = w_r(k,i)
    #
    # Wait, the indices don't quite match. Let me re-read.
    # U has shape (n*m, rank): indexed by (i,j) pair -> row i*m+j
    # V has shape (m*p, rank): indexed by (j,k) pair -> row j*p+k  
    # W has shape (p*n, rank): indexed by (k,i) pair -> row k*n+i
    #
    # The tensor T[i,j,k] for matrix multiplication: T[i,j,k] = 1 for all i,j,k?
    # No! The matrix multiplication tensor has T[i,j,k] = 1 when the (i,k) entry
    # of C depends on A[i,j] and B[j,k]. Since C[i,k] = sum_j A[i,j]*B[j,k],
    # the tensor entry T[i,j,k] = 1 for all valid i,j,k.
    #
    # Wait no. The tensor is defined such that the bilinear form
    # sum_{i,j,k} T[i,j,k] a[i,j] b[j,k] gives the product.
    # Actually the standard convention: the multiplication tensor mu_{n,m,p}
    # is in A^* x B^* x C^* where A is n x m, B is m x p, C is n x p.
    # mu = sum_{i,j,k} e_i^A e_j^B ... 
    # 
    # Let me just use the direct construction. The tensor T has entries
    # T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # This is because C[i,k] = sum_j A[i,j] B[j,k], and each term contributes.
    #
    # No wait, that's wrong too. The tensor is:
    # mu = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{i,j}^A * E_{j,k}^B * E_{i,k}^C
    # where E_{i,j}^A is the basis element of A^* etc.
    #
    # So T[i,j,k] in the flattened form: the entry at position (i,j) in U-space,
    # (j,k) in V-space, (k,i) in W-space... 
    #
    # Actually I think the correct interpretation is:
    # The tensor has one index for each matrix: i (row of A / row of C),
    # j (col of A / row of B), k (col of B / col of C).
    # T[i,j,k] = 1 for all i,j,k.
    #
    # Then a rank-1 decomposition gives:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] = 1 for all i,j,k.
    # Wait, that's not right either. Let me think again.
    #
    # The multiplication tensor mu_{n,m,p} is an element of R^{n x m} * R^{m x p} * R^{n x p}
    # (the dual space, i.e., multilinear forms on A, B, C).
    # mu(A,B,C) = sum_{i,j,k} A[i,j] B[j,k] C[i,k] = trace(A^T * B * C^T)... 
    # Hmm, actually mu(A,B) = C = A*B, so the tensor is:
    # mu = sum_{i,j,k} e_i^C e_j^B e_k^A (as a trilinear form, mu(a,b,c) = sum a[k] b[j] c[i] * delta)
    #
    # I'm overcomplicating this. Let me just verify the schoolbook:
    # In the schoolbook, for each (i,j,k) triple:
    #   U[i*m+j][col] = 1  (i.e., U[i,j] = 1)
    #   V[j*p+k][col] = 1  (i.e., V[j,k] = 1)
    #   W[k*n+i][col] = 1  (i.e., W[k,i] = 1)
    #
    # And the contraction sum_j U[i,j,c] V[j,k,c] for fixed i,k,c gives
    # sum_j delta[i,j]_something... 
    #
    # I think the check is: for each rank factor c, the contribution to C[i,k] is
    # sum_j U[i*m+j, c] * V[j*p+k, c], and this should sum over c to give W's
    # coefficient... 
    #
    # Actually, I believe the correct check is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (since the multiplication tensor has all ones).
    #
    # Wait, that's a 3-way contraction, but rank-1 tensors factor as
    # u * v * w, so sum_c u_c * v_c * w_c = T.
    #
    # Yes! So the condition is:
    # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    #   sum_{c=0}^{rank-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # (assuming T[i,j,k] = 1 for the multiplication tensor)
    #
    # Now, for Strassen on 2x2x2 (n=m=p=2):
    # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k in {0,1}.
    #
    # Let me construct the 2x2x2 Strassen decomposition with this convention.
    # The 7 rank-1 terms correspond to the 7 Strassen products.
    # 
    # For each Strassen term r, we need U_r[i,j], V_r[j,k], W_r[k,i] such that
    # sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 for all i,j,k.
    #
    # This is a decomposition of the all-ones tensor of size 2x2x2.
    #
    # The all-ones tensor J_{2,2,2} can be decomposed as sum of 7 rank-1 tensors.
    #
    # One way: use the fact that J = (sum_i e_i)(sum_j e_j)(sum_k e_k) but that's rank 1.
    # No, J_{2,2,2} has entries all 1, and it's NOT rank 1.
    #
    # Actually, the all-ones tensor IS rank 1: J[i,j,k] = 1 = (1)(1)(1).
    # So the multiplication tensor is NOT the all-ones tensor!
    #
    # Let me reconsider. The multiplication tensor mu_{n,m,p}:
    # mu = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} |i><j| ⊗ |j><k| ⊗ |i><k|
    # No, that's not right either.
    #
    # The standard definition: mu_{n,m,p} ∈ (R^{n×m})^* ⊗ (R^{m×p})^* ⊗ (R^{n×p})^*
    # mu = sum_{i,j,k} E_{ij}^{(n,m)*} ⊗ E_{jk}^{(m,p)*} ⊗ E_{ik}^{(n,p)*}
    #
    # So mu(A,B,C) = sum_{i,j,k} A_{ij} B_{jk} C_{ik}
    #
    # And indeed, if C = A*B, then mu(A,B,A*B) = sum_{i,j,k} A_{ij} B_{jk} (AB)_{ik}
    # = sum_{i,j,k,l} A_{ij} B_{jk} A_{il} B_{lk}... this is getting circular.
    #
    # The key point: mu(A,B,C) = sum_{i,j,k} A_{ij} B_{jk} C_{ik}
    # For the decomposition: mu = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r ∈ (R^{n×m})^*, V_r ∈ (R^{m×p})^*, W_r ∈ (R^{n×p})^*
    #
    # In coordinates: U_r is indexed by (i,j), V_r by (j,k), W_r by (i,k)
    # (note: W is indexed by (i,k) not (k,i) in the standard convention)
    #
    # But the problem says W[k*n+i, r], so W is indexed by (k,i) where the
    # row index is k*n+i. This means W_r[k,i] in the flattened form.
    #
    # So the condition is:
    # sum_c U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c] = 1 for all i,j,k
    #
    # Wait, but mu(A,B,C) = sum_{i,j,k} A_{ij} B_{jk} C_{ik}, so the tensor
    # entry at (i,j) for A, (j,k) for B, (i,k) for C is 1.
    #
    # With W indexed as W[k*n+i], the (i,k) entry of C^* is at position k*n+i.
    # So W_r[k*n+i, c] corresponds to the (i,k) component... no, position k*n+i
    # means k is the "row" and i is the "column" in the flattened p×n matrix.
    #
    # I think there might be an index swap. Let me just go with the schoolbook
    # verification: in the schoolbook, U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # for the term corresponding to (i,j,k). Then:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 (only one c contributes)
    # This equals 1 for all i,j,k, which matches mu having entry 1 at (i,j,k)
    # when W is indexed by (k,i) corresponding to C's (i,k) entry.
    #
    # OK so the condition is confirmed: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    # for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    #
    # Now for Strassen 2x2x2: I need 7 rank-1 tensors.
    # Let me use a known decomposition. The all-1 tensor of size 2x2x2...
    # wait, mu_{2,2,2}[i,j,k] = 1 for all i,j,k? Yes, because every (i,j,k) triple
    # contributes to the multiplication.
    #
    # So I need to decompose the 2x2x2 all-ones tensor into 7 rank-1 tensors.
    #
    # This is equivalent to finding 7 triples of 2x2 matrices (U_r, V_r, W_r)
    # such that sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 for all i,j,k.
    #
    # Let me try a systematic approach. I'll use the fact that the all-ones
    # 2x2x2 tensor can be written as:
    # J = 1⊗1⊗1 where 1 is the all-ones 2-vector.
    # But that's rank 1! So J_{2,2,2} IS rank 1.
    #
    # Wait, that can't be right because Strassen says rank is 7.
    #
    # Ah, I see my confusion. The multiplication tensor is NOT the all-ones tensor
    # in the sense I'm thinking. Let me re-examine.
    #
    # mu_{n,m,p}(A,B,C) = sum_{i,j,k} A_{ij} B_{jk} C_{ik}
    #
    # The tensor mu as an element of the tensor product space:
    # mu = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{ij} ⊗ E_{jk} ⊗ E_{ik}
    #
    # where E_{ij} is the (i,j) basis vector in R^{n×m}, etc.
    #
    # So mu[i,j,k] in the sense of "the coefficient of E_{ij} ⊗ E_{jk} ⊗ E_{ik}"
    # is 1 for all i,j,k. So yes, mu IS the all-ones tensor in this basis.
    #
    # And the all-ones tensor J_{n,m,p}[i,j,k] = 1 for all i,j,k has rank 1
    # (it's the outer product of all-ones vectors).
    #
    # But Strassen says rank(mu_{2,2,2}) = 7, not 1. So something is wrong.
    #
    # OH WAIT. I think the issue is that the tensor lives in a different space.
    # The multiplication tensor is in (R^{n×m})^* ⊗ (R^{m×p})^* ⊗ (R^{n×p})^*,
    # and the "rank" refers to the tensor rank in THIS space, where the three
    # factors have different dimensions and the index j is shared between the
    # first two factors and the index i is shared between the first and third,
    # etc.
    #
    # No, tensor rank is just tensor rank. The all-ones tensor in R^{4} ⊗ R^{4} ⊗ R^{4}
    # (since n×m = 4, m×p = 4, n×p = 4 for 2×2×2) is indeed rank 1.
    #
    # I think I'm confusing the multiplication tensor with something else.
    # Let me look at this differently.
    #
    # The matrix multiplication tensor T ∈ R^{nm × mp × np} is defined by:
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k for some i,j,k
    # Otherwise 0.
    #
    # For n=m=p=2: a = i*2+j, b = j*2+k, c = i*2+k
    # (i,j,k) ∈ {0,1}^3
    # (0,0,0): a=0, b=0, c=0 → T[0,0,0]=1
    # (0,0,1): a=0, b=1, c=1 → T[0,1,1]=1
    # (0,1,0): a=1, b=0, c=2 → T[1,0,2]=1
    # (0,1,1): a=1, b=1, c=3 → T[1,1,3]=1
    # (1,0,0): a=2, b=0, c=0 → T[2,0,0]=1
    # (1,0,1): a=2, b=1, c=1 → T[2,1,1]=1
    # (1,1,0): a=3, b=2, c=2 → T[3,2,2]=1
    # (1,1,1): a=3, b=3, c=3 → T[3,3,3]=1
    #
    # So T is a 4×4×4 tensor with 8 non-zero entries (all 1s), NOT all 64 entries.
    #
    # THIS is the correct tensor! It's sparse, not all-ones.
    #
    # Now the rank of this tensor is what we want to minimize.
    # The schoolbook gives rank 8 (one for each (i,j,k) triple).
    # Strassen shows rank ≤ 7 for 2×2×2.
    #
    # Great, so for our problem with n=2, m=2, p=16:
    # T ∈ R^{4 × 32 × 32}
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i∈{0,1}, j∈{0,1}, k∈{0,...,15}
    #
    # That's 2*2*16 = 64 non-zero entries.
    #
    # Now, the structure: for fixed i,j, the entries in the k-direction are
    # T[i*m+j, j*p+k, k*n+i] for k=0..15.
    #
    # This is a tensor with a lot of structure. The (i,j) index determines
    # which "block" of the k-index we're looking at.
    #
    # For a given (i,j), the slice T[i*m+j, :, :] has non-zeros at positions
    # (j*p+k, k*n+i) for k=0..15.
    #
    # This looks like it could benefit from a decomposition that exploits
    # the regular structure in k.
    #
    # Let me think about this using the block-circulant idea from the directive.
    # The p=16 dimension can be seen as 4 blocks of size 4, or 8 blocks of size 2,
    # or 16 blocks of size 1.
    #
    # Actually, the directive says "four independent 2x2 matrix multiplications
    # coupled via a shared rank-8 structure". So p=16 = 4 * 4, and each block
    # of 4 columns corresponds to... hmm, 4 columns of B, which with m=2 rows
    # gives a 2×4 block. That's not 2×2.
    #
    # Maybe p=16 = 4 * 4 where each 4 corresponds to 2×2 (since m=2, each 2
    # columns of B form a 2×2). So 16 columns = 8 blocks of 2 columns each.
    # Or 16/2 = 8 independent 2×2 multiplications.
    #
    # Actually, B is 2×16. We can think of it as 8 blocks of 2×2: B[:, 0:2],
    # B[:, 2:4], ..., B[:, 14:16]. Each 2×2 block multiplies with A (2×2) to
    # give a 2×2 result. So C = A * B is 2×16, which is 8 blocks of 2×2.
    #
    # So the problem decomposes into 8 independent 2×2×2 multiplications!
    # Each has rank at most 7 (Strassen), so total rank ≤ 8*7 = 56.
    #
    # But can we do better by sharing structure? The key insight is that A is
    # shared across all 8 blocks. If we use the same Strassen decomposition for
    # each block, we get rank 56. But if we can share the A-side computation...
    #
    # Actually, in the tensor decomposition, each rank-1 term has a U factor
    # (depending on A), V factor (depending on B), and W factor (depending on C).
    # If we use Strassen for each of the 8 blocks, the U factors for different
    # blocks are the same (since A is the same), but the V and W factors differ
    # (since different blocks of B and C).
    #
    # However, in a rank-1 decomposition, each term is a full outer product,
    # so we can't really "share" U across terms without increasing the effective
    # rank.
    #
    # Let me just go with 8 independent Strassen decompositions, giving rank 56.
    # This is already much better than the schoolbook rank of 64.
    #
    # Actually wait - can we do even better? What if we use a decomposition
    # that exploits the full 2×16 structure more cleverly?
    #
    # Let me try a different approach: use the Kronecker product structure.
    # The tensor T for 2×2×16 can be written as T ⊗ I_8 in some sense,
    # where T is the 2×2×2 tensor and I_8 is an identity-like structure.
    #
    # If mu_{2,2,2} has rank 7, and the 16-dimensional k-index can be split
    # into 8 independent 2-dimensional subspaces, then mu_{2,2,16} might have
    # rank 7*8 = 56.
    #
    # But maybe we can do better. What if we use a decomposition of the
    # 2×2×2 tensor of rank 7, and then "tile" it 8 times? That gives 56.
    #
    # Alternatively, is there a way to get rank less than 56?
    #
    # The lower bound for tensor rank of mu_{n,m,p} is max(nm, mp, np) = max(4, 32, 32) = 32.
    # So the absolute minimum rank is 32.
    #
    # Can we achieve rank 32? That would require a very clever decomposition.
    #
    # For now, let me implement the 8×Strassen approach with rank 56, which is
    # a solid improvement over 64. If I can find a better structure, I'll use it.
    #
    # Actually, let me reconsider. The directive mentions "shared rank-8 structure"
    # and "Kronecker product properties with a 4x4 identity". 
    #
    # What if we decompose the 16-dimension as 4×4, and use a 4×4 circulant
    # or identity structure? The 2×2×(4×4) tensor might decompose more efficiently.
    #
    # Let me try: p = 16 = 4 * 4. Split k into (k1, k2) where k = k1*4 + k2,
    # k1 ∈ {0,1,2,3}, k2 ∈ {0,1,2,3}.
    #
    # The tensor entry T[i,j,k1,k2] (with k = k1*4+k2) is 1 when:
    # row_U = i*2+j, row_V = j*16 + k1*4 + k2, row_W = (k1*4+k2)*2 + i
    #
    # Hmm, this doesn't immediately suggest a Kronecker structure.
    #
    # Let me just go with the 8×Strassen decomposition. It's clean and gives
    # rank 56.
    
    n, m, p = DIMENSIONS
    # Split p=16 into 8 blocks of 2
    block_size = 2
    num_blocks = p // block_size  # 8
    
    # Strassen decomposition for 2x2x2
    # 7 rank-1 terms. Each term r has U_r (2x2), V_r (2x2), W_r (2x2)
    # such that sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = T_{2,2,2}[i,j,k]
    # where T_{2,2,2}[i,j,k] = 1 if the (i,j) of A, (j,k) of B, (k,i) of C
    # (note: C index is (k,i) in our W convention, which is (row=k, col=i) in
    # the p*n = 2*2 matrix, so W_r[k,i])
    
    # For 2x2x2, the tensor has entries:
    # T[a,b,c] where a=i*2+j, b=j*2+k, c=k*2+i
    # (i,j,k) in {0,1}^3
    # Non-zero entries:
    # (0,0,0), (0,1,1), (1,0,2), (1,1,3), (2,0,0), (2,1,1), (3,2,2), (3,3,3)
    
    # Let me construct the Strassen decomposition explicitly.
    # I'll use the standard Strassen formulas and translate to this tensor form.
    
    # Strassen terms (U, V, W) as 2x2 matrices:
    # Term 0: U = [[1,0],[1,0]], V = [[1,1],[0,0]], W = [[1,1],[0,0]]
    #   Check: sum over j of U[i,j]*V[j,k]*W[k,i]
    #   i=0,j=0,k=0: 1*1*1 = 1
    #   i=0,j=0,k=1: 1*1*0 = 0  
    #   i=0,j=1,k=0: 0*0*1 = 0
    #   i=0,j=1,k=1: 0*0*0 = 0
    #   i=1,j=0,k=0: 1*1*0 = 0
    #   i=1,j=0,k=1: 1*1*0 = 0
    #   i=1,j=1,k=0: 0*0*0 = 0
    #   i=1,j=1,k=1: 0*0*0 = 0
    #   Sum for (i,k) = (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0
    #   This is the (0,0) entry of the product.
    
    # This is getting complicated. Let me use a known rank-7 decomposition.
    
    # I'll use the following approach: construct the 8 non-zero tensor entries
    # and find a rank-7 decomposition using linear algebra.
    
    # Actually, let me just use the known Strassen decomposition in the
    # standard form and verify it works.
    
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00  
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In tensor form, each Mr corresponds to a rank-1 tensor:
    # Mr = (A-side vector) ⊗ (B-side vector) ⊗ (C-side contribution)
    
    # For M1 = (A00+A11)(B00+B11):
    # A-side: [1, 0, 0, 1] (coefficients of A00, A01, A10, A11)
    # B-side: [1, 1, 0, 0] (coefficients of B00, B01, B10, B11)
    # C-side: contributes to C00 with +1, so W has +1 at position (k=0, i=0)
    #   In our W indexing: W[k*n+i] = W[0*2+0] = W[0]
    
    # But wait, in the tensor decomposition, each rank-1 term contributes
    # to ALL (i,j,k) entries, not just one C entry. The C-side of a rank-1
    # term is a single scalar that multiplies U[i,j]*V[j,k].
    
    # Let me re-derive. A rank-1 term is:
    # u ⊗ v ⊗ w where u ∈ R^{nm}, v ∈ R^{mp}, w ∈ R^{np}
    # Contribution to T[a,b,c] is u[a]*v[b]*w[c]
    # 
    # We need: sum_r u_r[a] * v_r[b] * w_r[c] = T[a,b,c]
    #
    # For the Strassen decomposition of 2×2×2:
    # The tensor T has 8 non-zero entries (all 1s) at:
    # (0,0,0), (0,1,1), (1,0,2), (1,1,3), (2,0,0), (2,1,1), (3,2,2), (3,3,3)
    #
    # I'll construct the decomposition by solving a linear system.
    # But that might be fragile. Let me use a known explicit decomposition.
    
    # Here's a rank-7 decomposition of the 2×2×2 multiplication tensor:
    # Using the approach from the literature.
    
    # Let me just verify the schoolbook for 2×2×2 has rank 8, and then
    # try to find a rank-7 decomposition by hand.
    
    # Actually, for the purpose of this task, let me just use 8 independent
    # schoolbook decompositions for the 8 blocks of 2×2, giving rank 8*8=64.
    # That's the same as the original. Not helpful.
    
    # Let me use Strassen. I'll hardcode the 7 rank-1 terms for 2×2×2.
    
    # After careful derivation, here's a valid rank-7 decomposition:
    # (U_r, V_r, W_r) for r = 0..6, each a 4-vector (flattened 2×2 matrix)
    
    # U vectors (flattened, row-major: [U00, U01, U10, U11]):
    U_strassen = [
        [1, 0, 0, 1],  # r=0
        [0, 0, 1, 1],  # r=1
        [1, 0, 0, 0],  # r=2
        [0, 0, 0, 1],  # r=3
        [1, 1, 0, 0],  # r=4
        [0, 0, 1, -1], # r=5
        [0, 1, 0, -1], # r=6
    ]
    
    # V vectors (flattened, row-major: [V00, V01, V10, V11]):
    V_strassen = [
        [1, 1, 0, 0],  # r=0
        [1, 0, 0, 0],  # r=1
        [0, 1, 0, -1], # r=2
        [0, -1, 1, 0], # r=3
        [0, 0, 0, 1],  # r=4
        [1, 1, 0, 0],  # r=5
        [0, 0, 1, 1],  # r=6
    ]
    
    # W vectors (flattened, row-major for p*n = 2*2: [W00, W01, W10, W11]
    # where W[k*n+i] = W[k*2+i], so position 0=(k=0,i=0), 1=(k=0,i=1),
    # position 2=(k=1,i=0), 3=(k=1,i=1)):
    W_strassen = [
        [1, 0, 0, 0],  # r=0: contributes to C[0,0]
        [0, 0, 0, 1],  # r=1: contributes to C[1,1]
        [0, 1, 0, 0],  # r=2: contributes to C[0,1]
        [0, 0, 1, 0],  # r=3: contributes to C[1,0]
        [0, 1, 0, 1],  # r=4: contributes to C[0,1] and C[1,1]
        [0, 0, 1, 1],  # r=5: contributes to C[1,0] and C[1,1]
        [1, 0, 0, 0],  # r=6: contributes to C[0,0]
    ]
    
    # Let me verify this decomposition by checking all 8 tensor entries.
    # T[a,b,c] should equal sum_r U_r[a]*V_r[b]*W_r[c]
    
    # Non-zero entries of T (for 2x2x2):
    # (i,j,k) -> (a,b,c) = (i*2+j, j*2+k, k*2+i)
    entries = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i*2+j
                b = j*2+k
                c = k*2+i
                entries.append((a,b,c))
    
    # Verify
    for (a,b,c) in entries:
        total = sum(U_strassen[r][a] * V_strassen[r][b] * W_strassen[r][c] 
                    for r in range(7))
        if total != 1:
            # This decomposition is wrong, let me try a different one
            pass
    
    # Since I'm not confident in the hardcoded values, let me use a
    # computational approach: solve for the rank-7 decomposition.
    
    # Actually, let me just use the schoolbook for each 2×2 block.
    # This gives rank 8 per block, 8 blocks, total rank 64.
    # Same as the original. Not an improvement.
    
    # Let me try a different strategy: use the fact that the tensor has
    # a specific structure and find a low-rank decomposition computationally.
    
    # The tensor T ∈ R^{4 × 32 × 32} has 64 non-zero entries.
    # Let me try to find a rank-r decomposition by solving:
    # T[a,b,c] = sum_{r=1}^{r} U[a,r] * V[b,r] * W[c,r]
    
    # For a given r, this is a system of 64 equations (one per non-zero entry)
    # with 4*r + 32*r + 32*r = 68*r unknowns.
    # For r=32: 64 equations, 2176 unknowns (underdetermined)
    # For r=56: 64 equations, 3808 unknowns
    
    # The issue is that we need the equations to be satisfied, but there are
    # also the zero entries that must remain zero. There are 4*32*32 - 64 = 
    # 4032 zero entries, giving 4032 more equations. Total: 4096 equations.
    
    # This is a hard non-linear system. Let me just go with a known good
    # decomposition.
    
    # I'll use 8 independent Strassen decompositions. Even if my hardcoded
    # Strassen is wrong, I can verify and fix it.
    
    # Let me use a verified Strassen decomposition. I'll construct it
    # programmatically by finding a rank-7 decomposition of the 2×2×2 tensor.
    
    import numpy as np
    
    def find_2x2x2_decomposition():
        """Find a rank-r decomposition of the 2x2x2 multiplication tensor."""
        # Build the 2x2x2 tensor
        T = np.zeros((4, 4, 4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i*2+j
                    b = j*2+k
                    c = k*2+i
                    T[a,b,c] = 1.0
        
        # Try rank 7
        # Use random initialization and gradient-free approach
        # Or use the known Strassen decomposition
        
        # Known Strassen decomposition (verified):
        # I'll use the formulation from the Wikipedia article on Strassen's algorithm
        # and translate it to tensor form.
        
        # The 7 intermediate products:
        # M1 = (A00+A11)(B00+B11) -> U1=[1,0,1,0]... 
        # Wait, let me be very careful.
        
        # A is 2x2: A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
        # Flattened (row-major): index 0=A00, 1=A01, 2=A10, 3=A11
        # B is 2x2: B[0,0]=B00, B[0,1]=B01, B[1,0]=B10, B[1,1]=B11
        # Flattened: index 0=B00, 1=B01, 2=B10, 3=B11
        # C is 2x2: C[0,0]=C00, C[0,1]=C01, C[1,0]=C10, C[1,1]=C11
        # W is indexed by k*n+i where n=2, so:
        #   W[0] = W[k=0,i=0] -> C[0,0]
        #   W[1] = W[k=0,i=1] -> C[1,0]
        #   W[2] = W[k=1,i=0] -> C[0,1]
        #   W[3] = W[k=1,i=1] -> C[1,1]
        
        # Wait, I need to double-check the W indexing.
        # W has shape (p*n, rank) = (2*2, rank) = (4, rank)
        # W[k*n+i, r] where k is the column index of C, i is the row index of C.
        # So W[k*2+i] is the coefficient for C[i,k] in rank factor r.
        # Position 0: k=0,i=0 -> C[0,0]
        # Position 1: k=0,i=1 -> C[1,0]
        # Position 2: k=1,i=0 -> C[0,1]
        # Position 3: k=1,i=1 -> C[1,1]
        
        # Now the tensor condition:
        # sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k
        
        # Let me define the 7 Strassen terms:
        # Each term r has U_r (length 4), V_r (length 4), W_r (length 4)
        
        # Term 0: M1 = (A00+A11)(B00+B11), contributes to C00 and C11
        # U0 = [1, 0, 0, 1]  (A00 + A11)
        # V0 = [1, 0, 0, 1]  (B00 + B11)
        # W0 = [1, 0, 0, 1]  (C00 + C11)
        
        # Term 1: M2 = (A10+A11)B00, contributes to C10
        # U1 = [0, 0, 1, 1]  (A10 + A11)
        # V1 = [1, 0, 0, 0]  (B00)
        # W1 = [0, 1, 0, 0]  (C10)
        
        # Term 2: M3 = A00(B01-B11), contributes to C01
        # U2 = [1, 0, 0, 0]  (A00)
        # V2 = [0, 1, 0, -1]  (B01 - B11)
        # W2 = [0, 0, 1, 0]  (C01)
        
        # Term 3: M4 = A11(B10-B00), contributes to C00 and C10
        # U3 = [0, 0, 0, 1]  (A11)
        # V3 = [0, 0, 1, -1]  (B10 - B00)
        # W3 = [1, 1, 0, 0]  (C00 + C10)
        
        # Term 4: M5 = (A00+A01)B11, contributes to C01 and C11
        # U4 = [1, 1, 0, 0]  (A00 + A01)
        # V4 = [0, 0, 0, 1]  (B11)
        # W4 = [0, 0, 1, 1]  (C01 + C11)
        
        # Term 5: M6 = (A10-A00)(B00+B01), contributes to C11
        # U5 = [0, 0, 1, -1]  (A10 - A00)
        # V5 = [1, 1, 0, 0]  (B00 + B01)
        # W5 = [0, 0, 0, 1]  (C11)
        
        # Term 6: M7 = (A01-A11)(B10+B11), contributes to C00
        # U6 = [0, 1, 0, -1]  (A01 - A11)
        # V6 = [0, 0, 1, 1]  (B10 + B11)
        # W6 = [1, 0, 0, 0]  (C00)
        
        U_s = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1],
        ]
        V_s = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        W_s = [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i*2+j
                    b = j*2+k
                    c = k*2+i
                    total = sum(U_s[r][a] * V_s[r][b] * W_s[r][c] for r in range(7))
                    if total != 1:
                        return None  # decomposition failed
        
        return U_s, V_s, W_s
    
    result = find_2x2x2_decomposition()
    if result is None:
        # Fallback: use schoolbook for the whole thing
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
    
    U_s, V_s, W_s = result
    
    # Now build the 2x2x16 decomposition using 8 independent Strassen blocks.
    # Each block handles 2 columns of B (and thus 2 columns of C).
    # Block t handles columns 2t and 2t+1 of B and C.
    
    num_blocks = p // 2  # 8
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        col_offset = 7 * t
        for r in range(7):
            # U factor: same for all blocks (A is shared)
            for a in range(4):
                U[a][col_offset + r] = U_s[r][a]
            
            # V factor: B[j, 2t + k] for k=0,1
            # V[j*p + k, c] where k is the global column index
            # For block t, local columns are 2t and 2t+1
            # V_r[j*2 + k_local] where k_local = 0,1 corresponds to
            # global column 2t + k_local
            for j in range(2):
                for k_local in range(2):
                    b = j * p + (2 * t + k_local)
                    # V_s[r] is flattened 2x2: index j*2+k_local
                    V[b][col_offset + r] = V_s[r][j * 2 + k_local]
            
            # W factor: C[i, 2t + k] for k=0,1
            # W[k*n + i, c] where k is the global column index
            # For block t, local columns are 2t and 2t+1
            # W_r[k_local*n + i] where k_local = 0,1
            # Global position: (2t + k_local)*n + i
            for i in range(2):
                for k_local in range(2):
                    c = (2 * t + k_local) * n + i
                    # W_s[r] is flattened with index k*2+i (since n=2)
                    W[c][col_offset + r] = W_s[r][k_local * n + i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
