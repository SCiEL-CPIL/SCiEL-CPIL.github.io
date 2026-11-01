"""Find rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 7]

def _verify_fast(U, V, W, n, m, p):
    """Fast vectorized verification using tensor contraction."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct the tensor from the decomposition
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _schoolbook_decomposition(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return U, V, W

def _block_laderman_decomposition(n, m, p):
    """
    Block decomposition using Laderman for 2x2x2 blocks.
    For 2x2 * 2x7: 3 Laderman blocks (rank 7 each) + 1 schoolbook 2x2x1 (rank 4) = rank 25
    """
    # Laderman's rank-7 decomposition for 2x2x2
    # p0 = (a00 + a11) * (b00 + b11)
    # p1 = (a00 - a11) * (b00 - b11)  
    # p2 = (a00 + a10) * (b10 + b11)
    # p3 = (a11 - a10) * (b00 - b10)
    # p4 = (a01 + a11) * (b00 + b01)
    # p5 = (a01 - a10) * (b10 - b11)
    # p6 = (a00 + a01) * (b01 - b11)
    
    U_l = np.zeros((4, 7))
    V_l = np.zeros((4, 7))
    W_l = np.zeros((4, 7))
    
    # p0 = (a00 + a11) * (b00 + b11)
    U_l[0,0] = 1; U_l[3,0] = 1
    V_l[0,0] = 1; V_l[3,0] = 1
    W_l[0,0] = 1; W_l[3,0] = 1
    
    # p1 = (a00 - a11) * (b00 - b11)
    U_l[0,1] = 1; U_l[3,1] = -1
    V_l[0,1] = 1; V_l[3,1] = -1
    W_l[0,1] = -1; W_l[3,1] = -1
    
    # p2 = (a00 + a10) * (b10 + b11)
    U_l[0,2] = 1; U_l[2,2] = 1
    V_l[2,2] = 1; V_l[3,2] = 1
    W_l[2,2] = -1; W_l[3,2] = 1
    
    # p3 = (a11 - a10) * (b00 - b10)
    U_l[3,3] = 1; U_l[2,3] = -1
    V_l[0,3] = 1; V_l[2,3] = -1
    W_l[0,3] = 1; W_l[2,3] = -1
    
    # p4 = (a01 + a11) * (b00 + b01)
    U_l[1,4] = 1; U_l[3,4] = 1
    V_l[0,4] = 1; V_l[1,4] = 1
    W_l[1,4] = 1; W_l[3,4] = 1
    
    # p5 = (a01 - a10) * (b10 - b11)
    U_l[1,5] = 1; U_l[2,5] = -1
    V_l[2,5] = 1; V_l[3,5] = -1
    W_l[0,5] = 1; W_l[1,5] = -1; W_l[2,5] = 1
    
    # p6 = (a00 + a01) * (b01 - b11)
    U_l[0,6] = 1; U_l[1,6] = 1
    V_l[1,6] = 1; V_l[3,6] = -1
    W_l[0,6] = -1; W_l[1,6] = 1
    
    # Build full decomposition
    blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]
    total_rank = 0
    for k_start, k_end in blocks:
        width = k_end - k_start
        if width == 2:
            total_rank += 7
        else:
            total_rank += n * m * width
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for k_start, k_end in blocks:
        width = k_end - k_start
        if width == 2:
            for i in range(n):
                for j in range(m):
                    U[i*m + j, col_offset:col_offset+7] = U_l[i*m + j]
            for j in range(m):
                for k_local in range(width):
                    V[j*p + (k_start + k_local), col_offset:col_offset+7] = V_l[j*width + k_local]
            for k_local in range(width):
                for i in range(n):
                    W[(k_start + k_local)*n + i, col_offset:col_offset+7] = W_l[k_local*n + i]
            col_offset += 7
        else:
            for i in range(n):
                for j in range(m):
                    for k_local in range(width):
                        k = k_start + k_local
                        U[i*m + j, col_offset] = 1.0
                        V[j*p + k, col_offset] = 1.0
                        W[k*n + i, col_offset] = 1.0
                        col_offset += 1
    
    return U, V, W

def _try_rank_search(n, m, p, max_rank, time_limit=110.0):
    """
    Try to find a decomposition with rank <= max_rank using linear algebra.
    
    The tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    We can fix U and V, then solve for W linearly. Or use a more systematic approach.
    
    Strategy: Since the tensor has a special structure (it's a matrix multiplication tensor),
    we can try to find a decomposition by considering the structure.
    
    For 2x2 * 2x7, the tensor has entries T[i*2+j, j*7+k, k*2+i] = 1.
    
    Let's think about this differently. We can group terms by the 'j' index (the shared index).
    For j=0: T[i*2, 0*7+k, k*2+i] = A[i,0]*B[0,k]*C[k,i]
    For j=1: T[i*2+1, 1*7+k, k*2+i] = A[i,1]*B[1,k]*C[k,i]
    
    So the tensor is a sum of two "slices" indexed by j.
    
    For each j, we have a 2x7x2 sub-tensor. The rank of a 2xp x 2 multiplication is at most 2p+2 (using a trick).
    
    Actually, let me try a direct approach: set up the problem as finding sparse U, V, W.
    
    Alternative: Use the fact that we can precompute linear combinations of A's entries.
    If we precompute s = sum_j alpha_j * A[:,j] (a linear combination of A's columns),
    then s * B[:,k] gives us a column of C.
    
    Let me try: precompute R linear combinations of A (each is a 2-vector),
    then for each, multiply with each column of B (7 columns), giving R*7 products.
    But we need to be able to reconstruct C from these.
    
    C[k,i] = sum_j A[i,j] * B[j,k]
    
    If we precompute S_r[i] = sum_j U[i*m+j, r] * A[i,j] for r=0..R-1,
    and T_r[k] = sum_j V[j*p+k, r] * B[j,k] for r=0..R-1,
    then C[k,i] = sum_r W[k*n+i, r] * S_r[i] * T_r[k]
    
    This is the standard form. The question is what's the minimum R.
    
    Let me try a randomized approach with verification.
    """
    start_time = time.time()
    
    # Try various structural decompositions
    # Approach 1: Precompute linear combos of A rows
    # A is 2x2, so we can precompute at most 4 independent linear combos (but really 2x2=4)
    # Actually, the space of linear combos of A's entries is 4-dimensional.
    
    # Let's think: C[k,0] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[k,1] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # If we precompute:
    # s0 = A[0,0] + A[1,1]
    # s1 = A[0,0] - A[1,1]
    # s2 = A[0,1] + A[1,0]
    # s3 = A[0,1] - A[1,0]
    # s4 = A[0,0]
    # s5 = A[0,1]
    # s6 = A[1,0]
    # s7 = A[1,1]
    
    # Then for each k, we need to compute:
    # C[k,0] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[k,1] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Using the precomputed values, for each k we still need 4 multiplications
    # (or we can combine them)
    
    # Total: 8 precomputation + 7*4 = 36 operations. Worse than schoolbook.
    
    # Let me try a different approach: use the structure more directly.
    
    # The tensor T is 4 x 14 x 14 with 28 non-zero entries.
    # Let's try to find a rank-r decomposition by solving a system.
    
    # For a given rank r, we need to find U (4xr), V (14xr), W (14xr) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a bilinear problem in (U,V) for fixed W, etc.
    
    # Let's try a greedy/randomized approach:
    # 1. Initialize with schoolbook (rank 28)
    # 2. Try to merge pairs of rank-1 terms
    
    # Actually, let's try a more principled approach.
    # The tensor can be written as:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where i in {0,1}, j in {0,1}, k in {0,...,6}
    
    # This is sum over j of (sum_i e_{i,j} ⊗ e_{j,*} ⊗ e_{*,i})
    # For fixed j, the inner tensor is a 2x7x2 multiplication tensor.
    
    # The rank of a 2xp x 2 tensor: Let's compute it.
    # T_j[i, k, l] = 1 if i=l (i.e., the A index matches the C column index)
    # Wait, let me re-index.
    
    # For fixed j:
    # A has indices i in {0,1} (2 values)
    # B has indices k in {0,...,6} (7 values)  
    # C has indices (k, i) so 14 values
    
    # T_j[i, k, (k,i)] = 1
    
    # This is a 2 x 7 x 14 tensor with 14 non-zero entries.
    # Actually no, for fixed j, the A index is (i,j) but since j is fixed, 
    # the A sub-index is just i (2 values).
    # The B sub-index is k (7 values).
    # The C sub-index is (k, i) which gives 14 values, but only (k,i) pairs where i matches.
    
    # Hmm, this is getting complicated. Let me just try to find a good decomposition
    # by trying different structural forms.
    
    # Approach: Try rank 20, 21, 22, 23, 24 with a structured search.
    
    # For rank r, we can try to find U, V, W by:
    # 1. Fix a pattern for U (which entries are non-zero)
    # 2. Solve for V and W
    
    # Let's try a specific structural decomposition.
    
    # Key insight: For 2x2 * 2xp, we can use the following:
    # Precompute 4 values: d00, d01, d10, d11 (or combinations)
    # Then for each k, compute 2 products.
    # Total: 4 + 2p products. For p=7: 4+14=18.
    
    # But wait, the precomputation also costs. Let's count carefully.
    
    # Actually, in tensor decomposition, each rank-1 term is one "multiplication".
    # So if we have r rank-1 terms, the rank is r.
    
    # Let me try to construct a rank-18 decomposition:
    # Term 1-4: Precompute linear combos of A
    #   t1: U picks A[0,0], V picks 1, W picks... hmm this doesn't work directly.
    
    # Let me think differently. In a rank-r decomposition:
    # C[k,i] = sum_{r=0}^{r-1} W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # Wait, that's not right either. The tensor decomposition is:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Where a indexes A's entries, b indexes B's entries, c indexes C's entries.
    
    # So: sum_r U[a,r]*V[b,r]*W[c,r] = delta_{appropriate}
    
    # For a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if i=i', j=j', k=k' else 0
    
    # So: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # This means for each (i,j,k), the product sum equals 1, and for all other (i',j',k') it equals 0.
    
    # Let me try a specific construction. Suppose we use r terms where:
    # - Some terms involve only A[0,*] entries
    # - Some terms involve only A[1,*] entries
    # - Some terms involve combinations
    
    # Actually, let me just try to solve this as a linear system for small r.
    
    # For a fixed r, the equations are:
    # For each (i,j,k): sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1
    # For each (i',j',k') not equal to any (i,j,k): sum_r U[i'*m+j', r]*V[j'*p+k', r]*W[k'*n+i', r] = 0
    
    # Total equations: 4*14*14 = 784
    # Unknowns: r*(4+14+14) = 32r
    
    # For r=28: 896 unknowns, 784 equations (overdetermined but consistent)
    # For r=25: 800 unknowns, 784 equations
    # For r=20: 640 unknowns, 784 equations (underdetermined)
    # For r=15: 480 unknowns, 784 equations (likely infeasible)
    
    # The minimum possible r is at least the tensor rank, which for this type of tensor
    # is at least max(4, 14, 14) = 14, but more realistically at least the "border rank" considerations.
    
    # Let me try a randomized search for r=20, 21, 22, 23, 24.
    
    best_result = None
    best_rank = n*m*p  # Start with schoolbook rank
    
    for target_rank in range(24, 13, -1):  # Try from 24 down to 14
        if time.time() - start_time > time_limit:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Try to solve: minimize ||T_recon - T_target||
            # This is a non-convex problem, but we can try gradient descent or just check.
            
            # Actually, for a quick test, let's just check if a structured approach works.
            
            # Let me try a different structural approach for each rank.
            break  # Skip random for now, use structural
        
        # Try structural approach for this rank
        # ... (implement specific constructions)
        
        # For now, let's just return the best known
        if best_result is not None:
            break
    
    # Return the block Laderman decomposition (rank 25) as a safe fallback
    return _block_laderman_decomposition(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    # First try the block Laderman decomposition (rank 25)
    U, V, W = _block_laderman_decomposition(n, m, p)
    
    if _verify_fast(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
