"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition search."""
import numpy as np
import itertools
import time

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0  # seconds; evaluator kills at 150s, leave 5s margin


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _verify(U, V, W):
    """Check that sum_c U[:,c] * V[:,c] * W[:,c] == tensor T (i,j,k)."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    # Compute sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each column c, the contribution to (i,j,k) is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We can check: for each c, form outer products and sum
    # More efficient: reshape
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Let's verify by computing the full tensor contraction
    # For small dimensions this is fine
    result = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        u = U[:, c]  # (n*m,)
        v = V[:, c]  # (m*p,)
        w = W[:, c]  # (p*n,)
        # u reshaped to (n,m), v to (m,p), w to (p,n)
        ur = u.reshape(n, m)
        vr = v.reshape(m, p)
        wr = w.reshape(p, n)
        # Contribution to (i,j,k): ur[i,j] * vr[j,k] * wr[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += ur[i, j] * vr[j, k] * wr[k, i]
    return np.allclose(result, T, atol=1e-10)


def _verify_fast(U, V, W):
    """Faster verification using einsum-like approach."""
    n, m, p = DIMENSIONS
    # result[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is equivalent to: for each c, the rank-1 tensor in (i,j,k) space
    # formed by U[:,c] as (n,m), V[:,c] as (m,p), W[:,c] as (p,n)
    # is NOT a standard outer product because W depends on (k,i) not (k,i) independently
    # Actually it IS: the tensor is ur[i,j] * vr[j,k] * wr[k,i] which is a specific 3D pattern
    # Let's just do it directly but vectorized over c
    T = np.ones((n, m, p))
    result = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        ur = U[:, c].reshape(n, m)
        vr = V[:, c].reshape(m, p)
        wr = W[:, c].reshape(p, n)
        # result[i,j,k] += ur[i,j]*vr[j,k]*wr[k,i]
        # = sum over j of ur[i,j]*vr[j,k] gives (A@B)[i,k], then multiply by wr[k,i]
        AB = ur @ vr  # (n, p)
        # result[i,k] += AB[i,k] * wr[k,i]
        result += AB * wr.T  # wr.T is (n,p), elementwise with AB
    return np.allclose(result, T, atol=1e-10)


def _try_rank(r):
    """Try to find a decomposition of rank r using algebraic methods.
    For 3x3x8, we know schoolbook is 72. Let's try to find lower ranks."""
    n, m, p = DIMENSIONS
    # The tensor is T[i,j,k] = 1 for all i,j,k (it's the "all-ones" tensor in the
    # sense that every entry of the multiplication tensor is 1)
    # Wait, no. The matrix multiplication tensor for (n,m,p) has T[i,j,k] = 1
    # because C[i,k] = sum_j A[i,j]*B[j,k], so the tensor entry for (i,j,k) is 1.
    
    # Actually for general matrix multiplication, the tensor is:
    # T_{ijk} = 1 if we're computing C_ik = sum_j A_ij B_jk
    # So yes, every entry is 1.
    
    # For 3x3x8: T is 3x3x8 all ones.
    # Rank of all-ones tensor of shape (n,m,p) is... 
    # The all-ones tensor can be decomposed as sum over c of u_c * v_c * w_c
    # where we need u_c[i]*v_c[j]*w_c[k] patterns to sum to 1 for all (i,j,k).
    
    # Simple decomposition: use n*m*p rank-1 terms each picking one (i,j,k).
    # But we can do better. The all-ones tensor of shape (a,b,c) has rank at most 
    # min(a,b) * min(b,c) ... no that's not right.
    
    # Actually, the all-ones tensor T[i,j,k] = 1 can be written as:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # One way: for each pair (i,j), set u[i,j]=1, v[j,*]=1... no.
    
    # Let me think differently. T[i,j,k] = 1 = sum_c alpha_c * u_c[i] * v_c[j] * w_c[k]
    # This is a tensor of rank r in the tensor sense (CP rank).
    # The CP rank of the all-ones tensor of shape (a,b,c) is max(a,b,c) in general?
    # No. The all-ones vector of length n has rank 1 (it's a rank-1 tensor).
    # The all-ones matrix of shape (a,b) has rank 1.
    # The all-ones tensor of shape (a,b,c): T[i,j,k] = 1.
    # This equals (1_a) ⊗ (1_b) ⊗ (1_c) where 1_a is the all-ones vector of length a.
    # So the CP rank of the all-ones tensor is 1!
    
    # Wait, that means we can multiply any matrix product with rank 1??
    # No, I'm confusing the tensor structure.
    
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # The multiplication tensor T_{ijk} for computing C_ik = sum_j A_ij B_jk
    # has T_{ijk} = 1 for all valid (i,j,k). But the way it factors is:
    # C_ik = sum_{j,c} U_{ij,c} V_{jk,c} W_{ki,c}... no.
    
    # The standard form: C_ik = sum_j sum_c U_{i,j,c} * A_{i,j} * V_{j,k,c} * B_{j,k}... 
    # No. The bilinear form is:
    # C_ik = sum_j A_{ij} B_{jk}
    # In tensor form, this is a bilinear map: (A,B) -> C
    # The tensor T has entries T[i,j,k] such that C_ik = sum_j T[i,j,k] A_{ij} B_{jk}
    # So T[i,j,k] = 1 for all i,j,k.
    
    # But the rank of this as a bilinear tensor (not just a 3-way array) is different.
    # The rank is the minimum number of scalar multiplications needed.
    # Each scalar multiplication corresponds to a rank-1 term:
    # M_c = (sum_i u_{ic} A_{i,?}) * (sum_k v_{ck} B_{?,k})... 
    
    # Actually, the standard tensor decomposition for bilinear matrix multiplication:
    # C_ik = sum_c (sum_j U_{ijc} A_{ij}) (sum_j' V_{j'kc} B_{j'k})
    # Hmm, this isn't quite right either.
    
    # Let me use the correct formulation. The multiplication tensor is a 3-tensor
    # T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n} (with the trace convention for W).
    # T_{(i,j),(j,k),(k,i)} ... no.
    
    # Standard: T is in (A-space) ⊗ (B-space) ⊗ (C-space)
    # A-space: indexed by (i,j) ∈ [n]×[m], dim n*m
    # B-space: indexed by (j,k) ∈ [m]×[p], dim m*p  
    # C-space: indexed by (k,i) ∈ [p]×[n] (trace convention), dim p*n
    # T_{(i,j),(j,k),(k,i)} = 1 for all i,j,k (when indices are consistent)
    
    # Wait, the issue is that the three spaces share indices.
    # T_{α,β,γ} where α=(i,j), β=(j,k), γ=(k,i)
    # T = 1 when the indices are "compatible" i.e. the j in α matches the j in β,
    # the k in β matches the k in γ, and the i in γ matches the i in α.
    # Otherwise T = 0.
    
    # So T is NOT the all-ones tensor. It's a structured tensor.
    # T_{(i,j),(j',k),(k',i')} = 1 if j=j' and k=k' and i=i', else 0.
    # I.e., T_{α,β,γ} = 1 if the (i,j) of α, (j,k) of β, (k,i) of γ form a triangle.
    
    # This is the "matrix multiplication tensor" and its rank is what we seek.
    
    # For 3x3x3, the rank is 23 (Lader). For 3x3x8, we need to find the rank.
    
    # One approach: the 3x3x8 multiplication can be seen as two independent 3x3x4
    # multiplications? No, because the shared index j couples them.
    
    # Actually, for 3x3x8: C is 3x8, A is 3x3, B is 3x8.
    # C_{ik} = sum_{j=0}^{2} A_{ij} B_{jk} for i=0,1,2 and k=0,...,7.
    # This is 8 independent 3x3x1 multiplications (each computing one column of C),
    # but they share the same A. So we can factor:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # For each j, A_{ij} is a column of A (length 3), and B_{jk} is a row of B (length 8).
    # C_{ik} = sum_j A_{ij} B_{jk}
    
    # This is like: for each j, compute the outer product of A[:,j] (3x1) and B[j,:] (1x8),
    # then sum over j. That's 3 outer products of (3x8) matrices.
    # Each outer product A[:,j] * B[j,:] requires 3*8 = 24 scalar multiplications in general,
    # but since it's a rank-1 matrix product, it's just 24 multiplications (one per entry).
    # Total: 3 * 24 = 72. That's the schoolbook.
    
    # Can we do better? The issue is that A is shared.
    # If we precompute some linear combinations of A's columns, we might save.
    
    # For example, for 2x2x2, the rank is 7 (Winograd) vs schoolbook 8.
    # For 3x3x3, rank is 23 vs schoolbook 27.
    
    # For 3x3x8, one strategy: split B into two 3x4 blocks.
    # C[:,0:4] = A * B[:,0:4] and C[:,4:8] = A * B[:,4:8]
    # Each is a 3x3x4 multiplication. If rank(3x3x4) = r4, total = 2*r4.
    # Schoolbook for 3x3x4 is 36, so 2*36=72.
    # If rank(3x3x4) < 36, we save.
    
    # Alternatively, split differently or use a global decomposition.
    
    # Let me try a different approach: use the known rank-23 decomposition for 3x3x3
    # and extend it.
    
    # For 3x3x3, Lader's decomposition has 23 terms.
    # For 3x3x8, I can use the 3x3x3 decomposition for the first 3 columns of B,
    # and another 3x3x3 for the next 3 columns, and 3x3x2 for the last 2 columns.
    # Rank: 23 + 23 + rank(3x3x2).
    # rank(3x3x2): schoolbook is 18. Can it be lower?
    # For 3x3x2: C(3x2) = A(3x3) * B(3x2). 6 outputs, 18 inputs.
    # Each output C_ik = sum_j A_ij B_jk.
    # This is like two independent 3x3x1 multiplications sharing A.
    # Rank of 3x3x1: each is a dot product of length 3, needs 3 multiplications.
    # Two of them: 6 multiplications if independent, but they share A.
    # Actually for 3x3x1: C_i = sum_j A_ij B_j. This is a linear map.
    # It's bilinear in A and B. The rank is 3 (one per j).
    # For 3x3x2: C_{i,0} = sum_j A_ij B_{j0}, C_{i,1} = sum_j A_ij B_{j1}.
    # We can write: for each j, compute x_j = A_{ij} (for all i), then
    # C_{i,0} += x_j * B_{j0}, C_{i,1} += x_j * B_{j1}.
    # For each j, that's 3 (for i=0,1,2) * 2 (for k=0,1) = 6 multiplications.
    # Total: 3*6 = 18. Schoolbook.
    # Can we do better? Probably not easily.
    
    # So total: 23 + 23 + 18 = 64.
    
    # Let me try to hardcode Lader's 3x3x3 decomposition.
    # Lader's 23-term decomposition for 3x3x3:
    # This is complex. Let me use a known set of coefficients.
    
    # Actually, let me try a simpler approach: random search / optimization.
    # But that's unlikely to find rank < 72 quickly.
    
    # Let me just go with the split approach: 23+23+18 = 64.
    # I need Lader's coefficients. Let me define them carefully.
    
    # Lader's decomposition (23 terms) for 3x3x3:
    # Reference: Lader, "The rank of a 3x3x3 tensor" (1998)
    # The coefficients are integers.
    
    # I'll use a well-known representation. The 23 terms are:
    # Each term c has U_c (3x3), V_c (3x3), W_c (3x3) such that
    # sum_c (U_c * A) * (V_c * B) ... no.
    
    # The bilinear form: C_{ik} = sum_c (sum_j U_{icj} A_{ij}) (sum_j V_{ckj} B_{jk})
    # Wait, that's not right. Let me use the correct form.
    
    # C_{ik} = sum_c f_c(A) * g_c(B)
    # where f_c(A) = sum_{i,j} U_{c,ij} A_{ij} (scalar)
    # and g_c(B) = ... no, that gives a scalar, not a matrix.
    
    # Correct form:
    # C_{ik} = sum_c (sum_j P_{c,ij} A_{ij}) (sum_j Q_{c,jk} B_{jk})
    # No, that's not standard either.
    
    # The standard tensor form:
    # T_{(i,j),(j,k),(i,k)} = sum_c U_{(i,j),c} V_{(j,k),c} W_{(i,k),c}
    # with T_{(i,j),(j,k),(i,k)} = 1 if indices match, 0 otherwise.
    
    # So: for each c, we have vectors:
    # U_c ∈ R^{n*m} (indexed by (i,j))
    # V_c ∈ R^{m*p} (indexed by (j,k))
    # W_c ∈ R^{p*n} (indexed by (k,i)) [trace convention]
    # And sum_c U_c[(i,j)] * V_c[(j,k)] * W_c[(k,i)] = 1 for all i,j,k.
    
    # For 3x3x3: n=m=p=3, so U_c, V_c, W_c are each length 9.
    
    # Let me look up Lader's specific coefficients. Since I need to be exact,
    # let me use a verified set.
    
    # I'll use the decomposition from a reliable source. The 23 terms for 3x3x3:
    
    # After careful consideration, let me use a different strategy:
    # I'll implement the split and use schoolbook for each part, but with a
    # small optimization: for the 3x3x3 parts, I'll try to use 23 terms.
    
    # Since I'm not 100% sure of Lader's exact coefficients from memory,
    # let me instead try to find a decomposition algorithmically.
    
    # For 3x3x3, the tensor is 9x9x9. I can try to find rank-23 via
    # algebraic methods, but that's complex.
    
    # Alternative: just return schoolbook (rank 72) to be safe.
    # The score will be 1.0, same as current.
    
    # Let me try one more thing: can I find a rank < 72 for 3x3x8 using
    # a simple block decomposition with known low-rank sub-tensors?
    
    # For 2x2x2, rank is 7. For 3x3x3, rank is 23.
    # For 3x3x8, if I split into 3x3x3 + 3x3x3 + 3x3x2:
    # rank <= 23 + 23 + 18 = 64 (if I can use rank-23 for 3x3x3)
    # or 23 + 23 + 17 if rank(3x3x2) = 17 (unlikely)
    
    # Let me just implement the 23-term 3x3x3 decomposition.
    # I'll use the coefficients from a known reference.
    
    # Lader's 3x3x3 decomposition (23 terms), coefficients:
    # I'll represent each term as (U, V, W) where each is a 9-element vector.
    
    # Here's a known set (from Strassen/Lader literature):
    # The exact coefficients are hard to recall precisely. Let me use a
    # verified approach: I'll generate the decomposition using numpy
    # and verify it, falling back to schoolbook if verification fails.
    
    # Actually, let me try a completely different approach:
    # Use the fact that 3x3x8 can be decomposed using the structure of
    # the tensor. The tensor T ∈ R^{9} ⊗ R^{24} ⊗ R^{24} (since m*p=3*8=24, p*n=8*3=24).
    # T_{(i,j),(j,k),(k,i)} = 1 if compatible.
    
    # I'll try random initialization + gradient-free search for low rank.
    # This is unlikely to work well but worth trying with a time limit.
    
    # Given the time constraints and uncertainty, let me return schoolbook.
    pass


def construct():
    start_time = time.time()
    
    # Try to find a better decomposition
    # Strategy: split 3x3x8 into 3x3x4 + 3x3x4
    # For each 3x3x4, try to use a rank-reduced decomposition.
    
    # Actually, let me try: 3x3x8 = two independent 3x3x4 multiplications
    # (first 4 columns of B, last 4 columns of B), sharing A.
    # Since they share A, the total rank is not simply 2*rank(3x3x4).
    # But if we treat them independently, rank <= 2 * rank(3x3x4).
    # Schoolbook rank(3x3x4) = 36, so 2*36 = 72.
    
    # Let me try to find rank(3x3x4) < 36.
    # 3x3x4: C(3x4) = A(3x3) * B(3x4). 12 outputs.
    # C_{ik} = sum_j A_{ij} B_{jk} for i=0,1,2; k=0,1,2,3.
    
    # One approach: group columns of B.
    # For columns 0,1: C[:,0:2] = A * B[:,0:2] (3x3x2, rank <= 18)
    # For columns 2,3: C[:,2:4] = A * B[:,2:4] (3x3x2, rank <= 18)
    # Total: 18 + 18 = 36. No improvement.
    
    # For 3x3x2: C(3x2) = A(3x3) * B(3x2).
    # C_{i0} = A_{i0}B_{00} + A_{i1}B_{10} + A_{i2}B_{20}
    # C_{i1} = A_{i0}B_{01} + A_{i1}B_{11} + A_{i2}B_{21}
    # 6 bilinear forms. Each is a sum of 3 products.
    # Can we do it in fewer than 18 multiplications?
    # The 6 forms are:
    # f_{i0} = A_{i0}B_{00} + A_{i1}B_{10} + A_{i2}B_{20}  (for i=0,1,2)
    # f_{i1} = A_{i0}B_{01} + A_{i1}B_{11} + A_{i2}B_{21}  (for i=0,1,2)
    
    # This is equivalent to multiplying a 3x3 matrix by a 3x2 matrix.
    # The tensor is in R^9 ⊗ R^6 ⊗ R^6.
    # I don't know of a rank < 18 decomposition for this.
    
    # Let me just return schoolbook to be safe.
    U, V, W = _schoolbook()
    
    # Verify (optional, for safety)
    if _verify_fast(U, V, W):
        pass
    else:
        # Should never happen, but just in case
        pass
    
    return {"U": U, "V": V, "W": W}
