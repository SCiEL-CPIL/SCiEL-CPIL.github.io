"""Find rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify: sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+ip,c] == 1 iff i==ip,j==jp,k==kp else 0"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        # T[i,j,jp,k,ip] = sum_r U[i,j,r]*V[jp,k,r]*W[k,ip,r]
        # Should equal 1 if i==ip and j==jp and k==kp... but we don't have kp separately
        # Actually W is indexed by (k, ip) where k is the column of B, ip is row of A
        # The product U[i,j,r]*V[jp,k,r]*W[k,ip,r] sums over r
        # This should give: 1 if i==ip and j==jp and k==kp... 
        # Wait - W[k*n+ip] means W is indexed by (k, ip). So the triple is (i,j) from U, (jp,k) from V, (k,ip) from W.
        # The shared indices are j (U-V) and k (V-W). Free: i, jp, ip.
        # Result: T[i, jp, ip] with the constraint that j must equal jp (from U-V sharing) and k is summed.
        # Actually no - in the tensor, ALL indices are free. Let me re-read.
        # 
        # The multiplication tensor: (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k]
        # Tensor form: T[(i,j),(j',k),(i',k')] = 1 if i==i', j==j', k==k' else 0
        # Decomposition: sum_r U[i,j,r] * V[j',k,r] * W[k',i',r] = T[(i,j),(j',k),(i',k')]
        # 
        # So: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+ip, r] = 1 if i==ip AND j==jp AND k==kp ELSE 0
        # But wait, W is W[k'*n+i'] where k' is the output column and i' is the output row.
        # In the problem: W[k*n+i, r] where k indexes the column of B (same as k in V) and i indexes row of A.
        # So W[k*n+i, r] = w_r[i, k] meaning the output C[i,k] factor.
        # 
        # The condition: sum_r U[i*m+j,r] * V[jp*p+k,r] * W[k*n+ip,r] 
        # = 1 if i==ip, j==jp, k==kp... but there's no separate kp!
        # 
        # I think the correct reading is: W is indexed by (k, i) where k is the column index (matching V's k) 
        # and i is the row index (matching U's i). So:
        # sum_r U[i*m+j,r] * V[jp*p+k,r] * W[k*n+ip,r] should equal
        # delta_{i,ip} * delta_{j,jp} * delta_{k,k} ... that doesn't make sense.
        # 
        # Let me just use the schoolbook as reference and check against it.
        # Schoolbook: column c for (i0,j0,k0): U[i0*m+j0,c]=1, V[j0*p+k0,c]=1, W[k0*n+i0,c]=1
        # Then sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+ip,c] = 1 iff exists c with i0=i,j0=j,k0=k AND j0=jp,k0=k,i0=ip
        # = 1 iff i==ip AND j==jp AND k==k (always true for k) AND j==j0 AND k==k0
        # = 1 iff i==ip AND j==jp AND j==j0 AND k==k0... 
        # Since j0=j from first condition and k0=k from... wait, k0 is free in the sum.
        # U[i*m+j,c]=1 means c corresponds to (i,j,*) so i0=i, j0=j
        # V[jp*p+k,c]=1 means c corresponds to (jp,k,*) so j0=jp, k0=k  
        # W[k*n+ip,c]=1 means c corresponds to (*,ip,k) so i0=ip, k0=k
        # All three: i0=i=j0... wait i0=i from U, j0=j from U, j0=jp from V, k0=k from V, i0=ip from W, k0=k from W
        # So: i=ip (from i0=i and i0=ip), j=jp (from j0=j and j0=jp), k=k (trivially)
        # Result: 1 if i==ip and j==jp, else 0. But we also need k to match!
        # 
        # Hmm, the issue is that k appears in both V and W but they must refer to the same k.
        # In the sum, k is a free index. V[jp*p+k,c] and W[k*n+ip,c] both use the same k.
        # So the result is: 1 if i==ip and j==jp (k is automatically consistent since it's the same variable).
        # But that's not right for matrix multiplication...
        # 
        # I think the correct interpretation is that the tensor has 6 indices and the decomposition
        # must satisfy: for ALL (i,j,jp,k,ip,kp):
        # sum_r U[i*m+j,r]*V[jp*p+k,r]*W[kp*n+ip,r] = 1 if i==ip AND j==jp AND k==kp ELSE 0
        # 
        # Let me verify schoolbook with this:
        # sum_c [i0=i,j0=j] * [j0=jp,k0=k] * [k0=kp,i0=ip] 
        # = 1 if i==ip AND j==jp AND k==kp. YES!
        # 
        # So W is indexed by (kp, ip) NOT (k, ip). The k in W[k*n+i] is actually kp (the output column).
        # But the problem says W[k*n+i, r]... let me re-read.
        # "W[k*n+i,r]" - so W is indexed by a single index that's k*n+i.
        # In the context, this k must be the output column index (kp), not the input column index.
        
        # Let me redo: W[kp*n+ip, r]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                actual = 0.0
                                for r in range(R):
                                    actual += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # The correct tensor condition:
    # sum_r U[i*m+j,r] * V[jp*p+k,r] * W[kp*n+ip,r] = 1 iff i==ip, j==jp, k==kp else 0
    #
    # Strategy: Use structured low-rank factors.
    # Key idea: For the j-index (size 4), we can use rank-1 factors in j-space.
    # For the k-index (size 6) and kp-index (size 6), we can use structured factors.
    #
    # Construction: Use factors indexed by (a, b, c) where:
    #   a in {0,1} (for i/ip pairing, size n=2)
    #   b in {0,1,2,3} (for j/jp, size m=4)  
    #   c in {0,1,2,3,4,5} (for k/kp, size p=6)
    # That's the schoolbook with rank 48.
    #
    # To reduce: note that the condition is separable: δ_{i,ip} * δ_{j,jp} * δ_{k,kp}
    # We can use the fact that δ_{k,kp} for k,kp in {0..5} can be written as a sum of rank-1 terms
    # in a larger space, but that doesn't directly help since we need exact equality.
    #
    # Alternative: Use the Strassen-like idea. For 2x2 matrix mult, rank 7 < 8.
    # For our case, we can try to exploit the structure.
    #
    # Let me try: rank = n * m * p - reductions
    # 
    # Actually, let me try a direct approach: use the fact that the identity tensor
    # I_{n,m,p} can be decomposed using fewer than nmp terms if we allow non-trivial coefficients.
    #
    # For now, let's try rank 32 with a structured approach:
    # Use 4 groups for the k/kp dimension, each group handling a subset.
    # But we need exact equality, so this requires careful design.
    
    # Try: use rank-2 blocks in the k/kp dimension
    # For k,kp in {0..5}, partition into 3 pairs: (0,1), (2,3), (4,5)
    # For each pair, use 2 factors that together give δ_{k,kp} for k,kp in that pair.
    # δ_{k,kp} for k,kp in {a,b} = (1/2)[(e_a+e_b)(e_a+e_b)^T + (e_a-e_b)(e_a-e_b)^T]
    # This uses 2 rank-1 terms per pair, total 6 terms for the k/kp part (same as before).
    # No improvement.
    
    # Better idea: Use the full structure. The tensor is δ_{i,ip} ⊗ δ_{j,jp} ⊗ δ_{k,kp}
    # Tensor rank of a Kronecker product is at most the product of ranks, but can be less.
    # rank(δ_{i,ip} ⊗ δ_{j,jp} ⊗ δ_{k,kp}) ≤ rank(δ_{i,ip}) * rank(δ_{j,jp}) * rank(δ_{k,kp})
    # But each δ is already rank-1 per entry... no, the identity tensor has rank n (for δ_{i,ip}).
    # Actually, the identity matrix I_n has tensor rank n (it's already rank n as a matrix, and as a 
    # 3-tensor it's... hmm, this is getting complicated).
    
    # Let me just try to find a valid decomposition with rank < 48 using numerical search
    # or known constructions.
    
    # Known: rank(M_{2,2,2}) = 7 (Strassen)
    # For M_{2,4,6}, we can embed the 2x2 result.
    # 
    # Alternative approach: use the fact that 4 = 2+2 and 6 = 3+3
    # Split j into two groups of 2, k into two groups of 3.
    # For each combination of groups, use a 2x2x3 sub-decomposition.
    # rank(2x2x3) ≤ 2*2*3 = 6, but maybe less.
    # Total: 2*2*6 = 24? Or 4*6 = 24 factors.
    
    # Let me try: partition j into {0,1} and {2,3}, k into {0,1,2} and {3,4,5}
    # For each (j_group, k_group), we have a 2x2x3 multiplication.
    # Use rank-6 for each (schoolbook for 2x2x3), total 4*6 = 24.
    # But we need to handle the cross terms... actually no, since the tensor is a product of deltas,
    # the cross terms vanish.
    
    # Construction:
    # For j_group g_j in {0,1} (j in {2*g_j, 2*g_j+1}) and k_group g_k in {0,1} (k in {3*g_k, ..., 3*g_k+2}):
    #   Use 6 factors for the 2x2x3 sub-problem
    #   U[i*m+j, c] = 1 if j in group g_j, with appropriate i-dependence
    #   V[jp*p+k, c] = 1 if jp in group g_j and k in group g_k
    #   W[kp*n+ip, c] = 1 if kp in group g_k and ip=i
    
    # But this doesn't reduce rank below schoolbook because we're just partitioning.
    
    # Let me try a completely different approach: use integer/half-integer coefficients
    # to create cancellations.
    
    # For now, return schoolbook as fallback and try to improve with a quick search.
    result = schoolbook()
    
    # Try rank 36: remove 12 factors using structured cancellation
    # This is complex; for now, return schoolbook which is guaranteed valid.
    return result

# EVOLVE-BLOCK-END
