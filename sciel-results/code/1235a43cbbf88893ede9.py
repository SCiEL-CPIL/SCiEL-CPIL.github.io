import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_dec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _build_rank_39():
    """
    Build the known rank 39 decomposition.
    5 Strassen blocks (2x2x2) + 1 Schoolbook block (2x2x1).
    """
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1

        # m1 = (a00+a11)(b00+b11)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (a10+a11)b00
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a00(b01-b11)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = a11(b10-b00)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a00+a01)b11
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (a10-a00)(b00+b01)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (a01-a11)(b10+b11)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    # Last column k=10: Schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1

    U[0, r] += 1
    V[v_e2, r] += 1
    W[w_c02, r] += 1
    r += 1

    U[1, r] += 1
    V[v_g2, r] += 1
    W[w_c02, r] += 1
    r += 1

    U[2, r] += 1
    V[v_e2, r] += 1
    W[w_c12, r] += 1
    r += 1

    U[3, r] += 1
    V[v_g2, r] += 1
    W[w_c12, r] += 1
    r += 1

    return U, V, W

def _try_rank_38():
    """
    Attempt rank-38 via splitting: find a bilinear form shared between
    the two row-maps of A (row 0 and row 1).
    
    The tensor decomposes as T = T_row0 ⊗ A[0,:] + T_row1 ⊗ A[1,:]
    where T_row0[i,k] = δ_{i,0}·δ_{j,0} (from A[0,0]) + δ_{i,1}·δ_{j,0} (from A[1,0])
    and T_row1[i,k] = δ_{i,0}·δ_{j,1} (from A[0,1]) + δ_{i,1}·δ_{j,1} (from A[1,1])
    
    Actually, more precisely:
    T[a, b, c] = Σ_i,j,k A[i,j] B[j,k] C[k,i]
    
    Split by A's rows:
    T = Σ_i A[i,0] * T_i0 + Σ_i A[i,1] * T_i1
    where T_i0 has entries T_i0[a,b,c] = δ_{a,i} * (terms with j=0)
    
    For rank-38, we need the two "halves" to share one bilinear form.
    Try: find (u, v, w) such that adding it to one half and removing an
    equivalent contribution from the other half still gives valid decomposition.
    
    Practical approach: start from rank-39, try to merge two tensors
    into one by finding a linear combination that still spans.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 100
    
    # Get the rank-39 decomposition
    U39, V39, W39 = _build_rank_39()
    r39 = U39.shape[1]
    
    # The rank-39 has 5 Strassen blocks (7 each = 35) + 4 schoolbook = 39
    # Try to find if any two rank-1 tensors can be merged.
    # Two rank-1 tensors (u1,v1,w1) and (u2,v2,w2) can be merged into one
    # (u,v,w) if there exist scalars α, β ≠ 0 such that:
    # u = α*u1 + β*u2, v = α*v1 + β*v2, w = α*w1 + β*w2
    # and the resulting tensor still decomposes correctly.
    # 
    # This is generally not possible, but we can try a different approach:
    # Use the fact that the last 4 tensors are schoolbook for column k=10.
    # Maybe we can use a Strassen-like trick on a 2x2x1 subproblem.
    
    # Alternative: try to find a rank-38 by solving a linear system.
    # The tensor T has 2*2*11 = 44 non-zero entries (one per (i,j,k)).
    # A rank-r decomposition gives r*44 equations with r*44 unknowns (entries of U,V,W).
    # For r=38, we have 38*44 = 1672 unknowns and 44 equations... that's underdetermined.
    # But the structure is multiplicative, so it's nonlinear.
    
    # Better approach: use the splitting idea.
    # The 44 tensor entries split into:
    # - Entries with i=0 (A[0,0] and A[0,1] contribute): 22 entries
    # - Entries with i=1 (A[1,0] and A[1,1] contribute): 22 entries
    # 
    # Each "half" needs rank at most 22 (schoolbook).
    # If we can find a rank-1 tensor that lies in both halves' span, 
    # we can save 1 rank.
    
    # Let's define the two sub-tensors:
    # T0: entries where output row is 0, i.e., T[0, b, c] for all b, c
    #   T0[b, c] = sum over j,k: A[0,j] B[j,k] C[k,0] where b=j*p+k, c=k*n+0
    #   = A[0,0]*B[0,k]*C[k,0] + A[0,1]*B[1,k]*C[k,0]
    #   Non-zero when b=j*p+k and c=k (since n=2, c=k*2+0=2k)
    #   So T0[j*p+k, 2k] = A[0,j] * B[j,k] * C[k,0]
    #
    # T1: entries where output row is 1, i.e., T[1, b, c] for all b, c
    #   T1[b, c] = A[1,0]*B[0,k]*C[k,1] + A[1,1]*B[1,k]*C[k,1]
    #   T1[j*p+k, 2k+1] = A[1,j] * B[j,k] * C[k,1]
    
    # For a rank-1 tensor (u, v, w) to be "compatible" with T0:
    # u[i*m+j] * v[j*p+k] * w[k*n+0] must match T0's pattern
    # T0 is zero unless i=0, so u[0*m+j] = u[j] can be non-zero, u[2+j]=u[1+j] must be 0
    # Wait, that's not right. u is indexed by (i,j) pairs flattened.
    # u[i*m+j] for i=0: u[0], u[1] (j=0,1)
    # u[i*m+j] for i=1: u[2], u[3] (j=0,1)
    # 
    # For T0 (output row 0), the tensor entry T[0, j*p+k, 2k] involves A[0,j].
    # In the decomposition, T_dec[0, b, c] = sum_r U[0, r] * V[b, r] * W[c, r]
    # This must equal sum_j A[0,j] B[j,k] C[k,0] where b=j*p+k, c=2k
    # 
    # Similarly T_dec[1, b, c] = sum_r U[1, r] * V[b, r] * W[c, r] = sum_j A[1,j] B[j,k] C[k,1]
    
    # The key insight: if we can find a rank-1 component (u,v,w) where
    # u[2] = u[3] = 0 (only contributes to row 0 of output) AND
    # u[0] = u[1] = 0 (only contributes to row 1 of output)
    # that's impossible for a single tensor.
    
    # Instead, let's try a more direct search:
    # Start with rank-39 and try to find a linear dependence among the 39 rank-1 tensors
    # that allows merging.
    
    # Actually, let's try a completely different approach:
    # For 2x2x11, the lower bound is max(2*2, 2*11, 2*11) = 22 (trivial)
    # But Strassen gives 7 for 2x2x2, so for 5 blocks: 5*7 + 4 = 39.
    # Can we do better on the last block? 2x2x1 has rank 4 (schoolbook is optimal for 2x2x1? 
    # Actually rank of 2x2x1 matmul is 4 since it's just two dot products of length 2).
    # Wait, 2x2x1: A is 2x2, B is 2x1, C is 1x2. Output is 2x2.
    # T[i,j,k=0] = sum_l A[i,l] B[l,0] C[0,j]
    # This is rank 4 (schoolbook) and I believe that's optimal.
    
    # What about combining the last column with the 5th Strassen block?
    # The 5th block handles columns k=8,9. The schoolbook handles k=10.
    # Can we make a 2x2x3 block with better than 7+4=11?
    # 2x2x3 matmul: known rank is 11 (same as 7+4).
    
    # Let me try yet another approach: randomized search for rank 38.
    # Fix U and V to have sparse structure, solve for W.
    
    # Actually, the most promising approach given the directive:
    # Use the splitting method. The two "halves" (row 0 and row 1 of output)
    # each need rank 22. If they share a rank-1 tensor, total is 22+22-1 = 43.
    # That's worse than 39. So splitting at the output row level doesn't help.
    
    # Better split: by the input column j (which column of B).
    # j=0: T[i, 0*p+k, k*n+i] = A[i,0] B[0,k] C[k,i] for i=0,1, k=0..10
    # j=1: T[i, 1*p+k, k*n+i] = A[i,1] B[1,k] C[k,i] for i=0,1, k=0..10
    # Each half has 22 non-zero entries.
    # Half j=0: rank at most 22 (schoolbook), potentially less.
    # Half j=1: same.
    # If they share rank-1 tensors, we save.
    
    # The j=0 half: T0[i, k, k*n+i] = A[i,0] B[0,k] C[k,i]
    # This is a 2x11x2 tensor (i, k, output-index = k*n+i)
    # With A[i,0] factored out: T0 = A[:,0] outer (B[0,:] outer C[:,i] pattern)
    # The "shape" tensor S0[i,k,l] where l=k*n+i: S0[i,k,k*n+i] = 1
    # This is the 2x11x2 "diagonal" tensor. Its rank?
    # S0[i,k,k*n+i] = 1 means for each (i,k), there's one non-zero at position (i, k, k*2+i)
    # This is 22 non-zero entries. The rank is at most 22.
    # Can it be less? The entries are "independent" in some sense.
    # For fixed i, the entries are at (i, k, 2k+i) for k=0..10: 11 entries.
    # These form a "diagonal" in the (k, 2k+i) plane.
    # The rank of a single diagonal of length 11 is 11 (they're linearly independent).
    # Two such diagonals (for i=0 and i=1) might share structure.
    # For i=0: positions (0, k, 2k) for k=0..10
    # For i=1: positions (1, k, 2k+1) for k=0..10
    # These are disjoint in the first index, so the combined rank is 11+11=22.
    # So the j=0 half has rank exactly 22.
    
    # Similarly j=1 half has rank 22.
    # Total without sharing: 44. With Strassen on pairs of columns: 39.
    # For rank 38, we'd need the two halves to share at least 4 rank-1 tensors
    # (22+22-4=40, still not 38) or find a different structure.
    
    # I think rank 38 may not be achievable with this approach.
    # Let me instead focus on optimizing the existing rank-39 solution
    # and ensuring it's correct, while trying a few more ideas.
    
    # Idea: can we use a 2x2x2 Strassen block that spans columns 9,10 
    # instead of 8,9, and handle column 8 differently?
    # Column 8 alone needs rank 4. Columns 9,10 with Strassen need rank 7.
    # Total: 35 (first 4 blocks) + 4 + 7 = 46. Worse.
    
    # What about mixing: use Strassen on (7,8,9,10) as a 2x2x4 block?
    # 2x2x4 matmul rank: unknown, but at most 7+4=11.
    # Probably 11 is optimal.
    
    # Let's just verify rank 39 is correct and return it.
    # But first, let me try one more thing: can we reduce the schoolbook
    # part from 4 to 3 by sharing with a Strassen tensor?
    
    # The schoolbook tensors for k=10 are:
    # s1: U[0], V[10], W[20]  (A[0,0]*B[0,10]*C[10,0])
    # s2: U[1], V[10], W[21]  (A[0,1]*B[1,10]*C[10,0])  
    # Wait, let me recheck the indexing.
    # U is indexed by (i,j) -> i*m+j = i*2+j
    # V is indexed by (j,k) -> j*p+k = j*11+k
    # W is indexed by (k,i) -> k*n+i = k*2+i
    
    # For the last schoolbook block (k=10):
    # T[i, j*11+10, 20+i] = A[i,j] * B[j,10] * C[10,i]
    # i=0,j=0: T[0, 10, 20] = A[0,0]*B[0,10]*C[10,0]
    # i=0,j=1: T[1, 10, 20] = A[0,1]*B[1,10]*C[10,0]
    # i=1,j=0: T[2, 10, 21] = A[1,0]*B[0,10]*C[10,1]
    # i=1,j=1: T[3, 10, 21] = A[1,1]*B[1,10]*C[10,1]
    
    # In the current decomposition, these are 4 separate rank-1 tensors:
    # (U=e0, V=e10, W=e20), (U=e1, V=e21, W=e20), (U=e2, V=e10, W=e21), (U=e3, V=e21, W=e21)
    # Wait, V index for (j,k): j=0,k=10 -> 0*11+10=10; j=1,k=10 -> 1*11+10=21
    # W index for (k,i): k=10,i=0 -> 10*2+0=20; k=10,i=1 -> 10*2+1=21
    
    # Can we merge any of these with a Strassen tensor from the 5th block?
    # The 5th block (k=8,9) has tensors involving V indices 8,9,19,20 and W indices 16,17,18,19.
    # W index 20 is used in both the 5th block (for k=10? no, k=8,9 give W indices 16-19) 
    # and the schoolbook (W index 20 for k=10,i=0).
    # Actually W index 20 = 10*2+0, which is k=10, i=0. Not in the 5th block.
    
    # I don't see an obvious way to merge. Let's just return rank 39.
    
    if time.time() > deadline:
        return None
    
    return None  # No improvement found

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Build the known rank 39 solution
    U, V, W = _build_rank_39()
    
    # 2. Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 44
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # 3. Try rank 38 (quick attempt)
    if time.time() < deadline - 5:
        r38 = _try_rank_38()
        if r38 is not None:
            U38, V38, W38 = r38
            if _verify(U38, V38, W38, n, m, p):
                return {"U": U38.tolist(), "V": V38.tolist(), "W": W38.tolist()}
    
    # Return the verified rank 39 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
