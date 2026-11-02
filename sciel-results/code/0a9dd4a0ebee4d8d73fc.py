import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor entries
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    # Convert to numpy arrays for fast verification
    entries_arr = np.array(tensor_entries, dtype=np.int32)
    A_idx = entries_arr[:, 0]
    B_idx = entries_arr[:, 1]
    C_idx = entries_arr[:, 2]
    
    # Precompute constraint structure for ALS
    a_to_bc = {}
    b_to_ac = {}
    c_to_ab = {}
    for (a, b, c) in tensor_entries:
        a_to_bc.setdefault(a, []).append((b, c))
        b_to_ac.setdefault(b, []).append((a, c))
        c_to_ab.setdefault(c, []).append((a, b))
    
    # Fast vectorized verification
    def verify_fast(U, V, W):
        """Vectorized verification of tensor decomposition."""
        # For each entry (a,b,c), check sum_r U[a,r]*V[b,r]*W[c,r] == 1
        U_sel = U[A_idx, :]  # shape (num_entries, rank)
        V_sel = V[B_idx, :]
        W_sel = W[C_idx, :]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.all(np.abs(vals - 1.0) < 1e-10)
    
    # Build the rank-39 Strassen decomposition
    rank39 = 5 * 7 + 1 * 4
    U39 = np.zeros((n * m, rank39))
    V39 = np.zeros((m * p, rank39))
    W39 = np.zeros((p * n, rank39))
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U39[0, r] = 1; U39[3, r] = 1
        V39[v_e, r] = 1; V39[v_h, r] = 1
        W39[w_c00, r] = 1; W39[w_c11, r] = 1
        r += 1
        
        # m2 = (c+d)e
        U39[2, r] = 1; U39[3, r] = 1
        V39[v_e, r] = 1
        W39[w_c10, r] = 1; W39[w_c11, r] = -1
        r += 1
        
        # m3 = a(f-h)
        U39[0, r] = 1
        V39[v_f, r] = 1; V39[v_h, r] = -1
        W39[w_c01, r] = 1; W39[w_c11, r] = 1
        r += 1
        
        # m4 = d(g-e)
        U39[3, r] = 1
        V39[v_g, r] = 1; V39[v_e, r] = -1
        W39[w_c00, r] = 1; W39[w_c10, r] = 1
        r += 1
        
        # m5 = (a+b)h
        U39[0, r] = 1; U39[1, r] = 1
        V39[v_h, r] = 1
        W39[w_c00, r] = -1; W39[w_c01, r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U39[2, r] = 1; U39[0, r] = -1
        V39[v_e, r] = 1; V39[v_f, r] = 1
        W39[w_c11, r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U39[1, r] = 1; U39[3, r] = -1
        V39[v_g, r] = 1; V39[v_h, r] = 1
        W39[w_c00, r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # ae
    U39[0, r] = 1
    V39[v_e2, r] = 1
    W39[w_c02, r] = 1
    r += 1
    
    # bg
    U39[1, r] = 1
    V39[v_g2, r] = 1
    W39[w_c02, r] = 1
    r += 1
    
    # ce
    U39[2, r] = 1
    V39[v_e2, r] = 1
    W39[w_c12, r] = 1
    r += 1
    
    # dg
    U39[3, r] = 1
    V39[v_g2, r] = 1
    W39[w_c12, r] = 1
    r += 1
    
    # Verify the rank-39 decomposition
    if not verify_fast(U39, V39, W39):
        return {"U": U39.tolist(), "V": V39.tolist(), "W": W39.tolist()}
    
    # Try to find a lower rank decomposition
    # Strategy: try different structural decompositions
    # For 2x2 x 2xP matrix multiplication, the tensor has structure:
    # For each k in 0..P-1, we need to compute 4 products
    # Strassen handles 2 k's with 7 multiplications instead of 8
    # So for P=11: 5 pairs of k's with Strassen (7 each) + 1 single k (4) = 39
    # Alternative: could we mix different strategies?
    
    # Try: what if we use a different grouping?
    # The key insight is that Strassen's 7-multiplication identity for 2x2
    # is essentially optimal for that sub-tensor. The question is whether
    # we can share multiplications across different k-pairs.
    
    # Let's try a direct search for rank-38 using a smarter approach
    # We'll use the structure: the tensor is a direct sum of 6 sub-tensors
    # (5 Strassen 2x2x2 blocks + 1 schoolbook 2x2x1 block)
    # The rank of a direct sum is at most the sum of ranks, and 
    # the question is whether we can do better.
    
    # For the 2x2x2 sub-tensor (Strassen), rank is 7 (optimal)
    # For the 2x2x1 sub-tensor, rank is 4 (optimal, it's just 4 independent products)
    
    # Actually, let me think about this differently. The full tensor T for
    # 2x2 x 2x11 multiplication has 4*2*11 = 88 nonzero entries.
    # The tensor is: T[i,j,k; j',k'; k'',i'] where the only nonzero
    # entries are when j=j' and k''=i'... wait let me re-derive.
    
    # Actually the tensor entries are:
    # T[a,b,c] = 1 iff a = i*m+j, b = j'*p+k, c = k'*n+i'
    # with the constraint that this corresponds to (A*B)[i,k'] = sum_j A[i,j]*B[j,k']
    # So T is nonzero when: a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # This means: i = a//m, j = a%m, j' = b//p, k = b%p, k' = c//n, i' = c%n
    # And we need j = j' and i = i'
    
    # So the tensor decomposes as a direct sum over (i,k) pairs? No...
    # Let me think again. The multiplication tensor for (n x m) * (m x p) -> (n x p)
    # has entries T[i,k',j] = 1 for all valid i, j, k'.
    # In the flattened form: T[a,b,c] where a=i*m+j, b=j*p+k', c=k'*n+i
    # The constraint is: from a we get (i,j), from b we get (j',k'), from c we get (k'',i')
    # And we need j=j' and i=i' and k'=k''
    
    # So T[a,b,c] = 1 iff (a%m == b//p) and (a//m == c%n) and (b%p == c//n)
    
    # This means the tensor is NOT a direct sum in the simple sense.
    # The k index couples the b and c indices.
    
    # However, for fixed i, the sub-tensor involving row i of A and column i of C
    # has structure: sum_j A[i,j] * B[j,k] for each k.
    # This is like a 1 x m times m x p = 1 x p product, which has rank p.
    # But we can share across different i's using Strassen-like tricks.
    
    # Let me try a more systematic search. Since we have time, let's try
    # rank 38 with a structured approach.
    
    # Key observation: the Strassen decomposition for each pair of k's uses
    # 7 multiplications. The 7 Strassen products are:
    # m1 = (a+d)(e+h)  -> contributes to C[0,0] and C[1,1]
    # m2 = (c+d)e      -> contributes to C[1,0] and C[1,1]
    # m3 = a(f-h)      -> contributes to C[0,1] and C[1,1]
    # m4 = d(g-e)      -> contributes to C[0,0] and C[1,0]
    # m5 = (a+b)h      -> contributes to C[0,0] and C[0,1]
    # m6 = (c-a)(e+f)  -> contributes to C[1,1]
    # m7 = (b-d)(g+h)  -> contributes to C[0,0]
    
    # Where for a 2x2 x 2x2 block with A = [[a,b],[c,d]] and B = [[e,f],[g,h]]:
    # C[0,0] = m1 - m2 - m4 + m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m3 + m6
    
    # The question is: can we share any of these 7 multiplications across
    # different k-pairs? The answer is NO in general, because the A matrix
    # is the same for all k's, but the B matrix changes.
    
    # However, we could potentially use a different decomposition that
    # groups k's differently. For example, instead of pairing (0,1), (2,3), etc.,
    # we could try other groupings.
    
    # But actually, for 2x2 x 2xP, the optimal rank is known to be:
    # If P is even: 7*(P/2)
    # If P is odd: 7*((P-1)/2) + 4
    # For P=11: 7*5 + 4 = 39
    
    # This is because each pair of k's requires at least 7 multiplications
    # (Strassen's bound for 2x2 matrix multiplication) and a single k
    # requires at least 4 multiplications.
    
    # So rank 39 is likely optimal! But let's verify by trying to find
    # a rank-38 decomposition just to be sure.
    
    # Try a quick randomized search for rank 38
    def try_rank_fast(r, num_trials=10):
        """Try to find rank-r decomposition using fast ALS."""
        if r >= rank39:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(1000 + r * 100 + trial)
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Normalize
            for mat in [U, V, W]:
                norms = np.linalg.norm(mat, axis=0, keepdims=True)
                norms[norms < 1e-10] = 1
                mat /= norms
            
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for a in range(n * m):
                    bc_list = a_to_bc.get(a, [])
                    if not bc_list:
                        continue
                    A_mat = np.zeros((len(bc_list), r))
                    for idx, (b, c) in enumerate(bc_list):
                        A_mat[idx, :] = V[b, :] * W[c, :]
                    try:
                        U[a, :] = np.linalg.lstsq(A_mat, np.ones(len(bc_list)), rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(m * p):
                    ac_list = b_to_ac.get(b, [])
                    if not ac_list:
                        continue
                    A_mat = np.zeros((len(ac_list), r))
                    for idx, (a, c) in enumerate(ac_list):
                        A_mat[idx, :] = U[a, :] * W[c, :]
                    try:
                        V[b, :] = np.linalg.lstsq(A_mat, np.ones(len(ac_list)), rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c in range(p * n):
                    ab_list = c_to_ab.get(c, [])
                    if not ab_list:
                        continue
                    A_mat = np.zeros((len(ab_list), r))
                    for idx, (a, b) in enumerate(ab_list):
                        A_mat[idx, :] = U[a, :] * V[b, :]
                    try:
                        W[c, :] = np.linalg.lstsq(A_mat, np.ones(len(ab_list)), rcond=None)[0]
                    except:
                        pass
                
                # Check convergence with fast verification
                if verify_fast(U, V, W):
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify_fast(U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
        
        return None
    
    # Try rank 38 (most likely to succeed if possible)
    result = try_rank_fast(38, num_trials=5)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the known rank-39 decomposition
    return {"U": U39.tolist(), "V": V39.tolist(), "W": W39.tolist()}
