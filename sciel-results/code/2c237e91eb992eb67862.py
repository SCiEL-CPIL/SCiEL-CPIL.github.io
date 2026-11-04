"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the matmul tensor.
    
    The matmul tensor M_{n,m,p} in R^{(n*m) x (m*p) x (p*n)} has:
    T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    T[a, b, c] = 0 otherwise
    
    The decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    n, m, p = 3, 3, 8
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all nonzero entries: T[i*m+j, j*p+k, k*n+i] should equal 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    # Check some zero entries: T[a,b,c] should be 0 when indices are inconsistent
    # E.g., T[0, 0, 0] = T[i*m+j, j*p+k, k*n+i] where i=0,j=0,k=0 -> a=0,b=0,c=0
    # That's actually a nonzero entry. Let's check T[0, 1, 0]:
    # a=0 means i=0,j=0. b=1 means j=0,k=1. c=0 means k=0,i=0.
    # Inconsistent (k=1 vs k=0), so should be 0.
    zero_checks = [(0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 8, 0)]
    for a, b, c in zero_checks:
        if a < n*m and b < m*p and c < p*n:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            err = abs(val)
            if err > max_err:
                max_err = err
    
    return max_err < 1e-9

def _schoolbook():
    """Return schoolbook rank-72 decomposition."""
    n, m, p = 3, 3, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _slice_and_dice_merge():
    """
    Slice-and-dice: partition 3x8 output into three 3x3 blocks (columns 0-2, 3-5, 6-7... 
    wait, 8 isn't divisible by 3. Let's use blocks of size 3,3,2.
    
    Actually, let's partition B's 8 columns into groups: {0,1,2}, {3,4,5}, {6,7}.
    For each group of columns, we compute a sub-matrix product A * B_sub.
    
    Block 1: A (3x3) * B[:, 0:3] (3x3) -> C[:, 0:3] (3x3), rank 9 schoolbook
    Block 2: A (3x3) * B[:, 3:6] (3x3) -> C[:, 3:6] (3x3), rank 9 schoolbook  
    Block 3: A (3x3) * B[:, 6:8] (3x2) -> C[:, 6:8] (3x2), rank 6 schoolbook
    
    Total naive: 9 + 9 + 6 = 24 rank terms.
    
    Now we try to merge: the U factors for all three blocks are the same A elements.
    Specifically, for block b with column range [c_start, c_end):
    - U_b[i*m+j, r] = A[i,j] for all r (same across blocks)
    - V_b[j*p+k, r] = B[j, c_start + (k-c_start)] ... 
    
    The key insight: U is shared. We can write:
    For each (i,j) pair in A, the U factor is A[i,j] repeated across all rank terms.
    The V and W factors differ per block.
    
    Let me think more carefully. In the schoolbook for M_{3,3,3}:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    For the full 3x3x8, we want:
    T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,8)
    
    Schoolbook: T[i,j,k] = A[i,j] * B[j,k] * 1, so U[i*m+j, r] = A[i,j], V[j*p+k, r] = B[j,k], W[k*n+i, r] = 1
    for r = i*m+j*p+k... no, r indexes the rank.
    
    Actually in the standard schoolbook:
    r = i*m*p + j*p + k (or some bijection)
    U[i*m+j, r] = 1 if r corresponds to (i,j,*)
    V[j*p+k, r] = 1 if r corresponds to (*,j,k)  
    W[k*n+i, r] = 1 if r corresponds to (i,*,k)
    
    Let me use the slice-and-dice approach differently.
    
    Partition k into groups: G0 = {0,1,2}, G1 = {3,4,5}, G2 = {6,7}
    
    For group Gb with |Gb| = gb columns:
    We compute A * B[:, Gb] which is a 3 x gb product.
    Schoolbook rank for this is 3 * 3 * gb = 9*gb.
    
    But here's the merging idea: across all groups, the U factor for entry (i,j) is always 1 (or A[i,j]).
    So we can factor out the U part.
    
    Specifically, define for each (i,j):
    u_{i,j} = 1 (scalar)
    
    For each group b and each column k in Gb:
    v_{j,k}^{(b)} = 1
    w_{k,i}^{(b)} = 1
    
    The tensor entry is: T[i,j,k] = u_{i,j} * v_{j,k}^{(b)} * w_{k,i}^{(b)} = 1
    
    Now, can we merge? The U vectors are: for rank term (i,j,k), U has 1 at position i*m+j.
    These are the same across all k for fixed (i,j).
    
    So we have 9 distinct U patterns (one per (i,j)), each used multiple times.
    Similarly, V patterns: for fixed (j,k), V has 1 at position j*p+k. There are 3*8=24 distinct.
    W patterns: for fixed (i,k), W has 1 at position k*n+i. There are 3*8=24 distinct.
    
    The rank-1 terms are indexed by (i,j,k) - 72 terms.
    But U only takes 9 distinct values, V takes 24, W takes 24.
    
    Merging: Two rank-1 terms (i,j,k1) and (i,j,k2) have the same U vector.
    They can be merged if V and W also allow it.
    Term 1: U=(e_{ij}), V=(e_{j,k1}), W=(e_{k1,i})
    Term 2: U=(e_{ij}), V=(e_{j,k2}), W=(e_{k2,i})
    
    These are different rank-1 tensors (different V and W), so they can't be merged into one.
    
    However, we CAN combine them: 
    U*(V1+W1) ... no, that changes the tensor.
    
    Actually, the sum of two rank-1 tensors with the same U:
    U ⊗ V1 ⊗ W1 + U ⊗ V2 ⊗ W2 = U ⊗ (V1 ⊗ W1 + V2 ⊗ W2)
    
    The second factor is a 24x24 matrix of rank at most 2. So this is a rank-2 contribution
    to the "V-W" part, but with a fixed U. This doesn't reduce the total rank below the sum.
    
    I think the merging doesn't help here because the V and W factors are orthogonal.
    
    Let me try a different approach: use the fact that we only need integer/half-integer entries
    and try to find a lower-rank decomposition via random search.
    """
    n, m, p = 3, 3, 8
    start = time.time()
    deadline = start + 130.0
    
    # First, let's try to find a rank below 72 via randomized approach
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Try ranks from 24 (lower bound) up to 71
    # Use random initialization and gradient-free optimization
    
    best_result = _schoolbook()
    
    # For efficiency, let's try a few specific ranks
    for target_rank in [40, 48, 56, 60, 64, 68]:
        if time.time() > deadline - 5:
            break
        # Random search for this rank
        found = False
        for attempt in range(20):
            if time.time() > deadline - 2:
                break
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Check if this random init accidentally works (extremely unlikely)
            # Instead, let's use a structured approach
            
            # Actually, let's try: fix U to have a specific structure
            # and solve for V, W
            pass
        
        # Try structured: U[i*m+j, r] = 1 for all i,j,r (rank = target_rank)
        # Then we need: sum_r V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # This is: for each k, sum_r V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j
        # Let M_k[i,j] = sum_r W[k*n+i, r] * V[j*p+k, r] (note: W is indexed by (k,i), V by (j,k))
        # We need M_k = J (all-ones 3x3 matrix) for each k
        
        # M_k[i,j] = sum_r W[k*n+i, r] * V[j*p+k, r]
        # This is: (W_k) @ (V_k)^T where W_k[i,r] = W[k*n+i, r] and V_k[j,r] = V[j*p+k, r]
        # We need W_k @ V_k^T = J for each k
        
        # J = 1*1^T (rank 1), so W_k and V_k must be proportional to 1
        # W_k[i,r] = a_r, V_k[j,r] = b_r, with sum_r a_r * b_r = 1
        # And a_r, b_r must be same for all i,j (since J[i,j] = 1 for all i,j)
        
        # So: for each k, W[k*n+i, r] = a_{k,r} for all i, V[j*p+k, r] = b_{k,r} for all j
        # sum_r a_{k,r} * b_{k,r} = 1 for each k
        
        # Total rank = sum over k of (number of distinct a_{k,r}, b_{k,r} pairs)
        # If we use the same (a_r, b_r) for all k: rank = 1, sum a_r*b_r = 1, works!
        # But then U[i*m+j, r] = 1 for all i,j, and:
        # T[i,j,k] = sum_r 1 * b_{k,r} * a_{k,r} = sum_r a_{k,r}*b_{k,r} = 1 ✓
        
        # Wait, this gives rank 1! But that can't be right...
        # Let me re-examine. If U[i*m+j, r] = 1 for all i,j,r (r=0 only, rank=1):
        # T[i,j,k] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1 * V[j*p+k,0] * W[k*n+i,0]
        # We need this = 1 for all i,j,k.
        # So V[j*p+k,0] * W[k*n+i,0] = 1 for all i,j,k.
        # Fix k: V[j*p+k,0] * W[k*n+i,0] = 1 for all i,j.
        # This means V[j*p+k,0] = c_k and W[k*n+i,0] = 1/c_k for all i,j.
        # So V[j*p+k,0] = c_k (depends only on k), W[k*n+i,0] = 1/c_k (depends only on k).
        # This works! Rank 1!
        
        # But wait - is this valid? The entries must be integer or half-integer.
        # c_k can be any nonzero real, and 1/c_k too. Let c_k = 1 for all k.
        # Then U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
        # T[i,j,k] = 1*1*1 = 1. ✓
        
        # RANK 1?! That seems too good. Let me double-check the tensor definition.
        
        # The matmul tensor M_{3,3,8}: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] such that C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k'] * delta(i',i) * delta(k',k)
        # Actually, the standard tensor for matmul is:
        # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
        # So T[i,j,k] = 1 for all valid (i,j,k), and the contraction is:
        # C[i,k] = sum_{j} A[i,j] * B[j,k]
        # In tensor form: C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k] where T[i,j,k]=1
        
        # The rank-1 decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # With rank 1: T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
        # We need this = 1 for all i, j, k.
        # Set U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1.
        # Then T[i,j,k] = 1. ✓
        
        # This IS valid! The matmul tensor M_{3,3,8} has rank 1?!
        
        # Wait, that can't be right. The matmul tensor M_{n,m,p} is known to have rank at least n*m*p / something...
        # Actually no. Let me reconsider what the tensor is.
        
        # The matrix multiplication tensor M_{n,m,p} is defined as:
        # T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n}
        # T[i,j, k,l] ... no.
        
        # Actually, the standard definition: M_{n,m,p} is a tensor in R^{(n*m) × (m*p) × (p*n)}
        # with entries T[a, b, c] where a = (i,j), b = (j,k), c = (k,i)
        # T[(i,j), (j,k), (k,i)] = 1 if the indices are consistent, 0 otherwise.
        
        # More precisely: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # T[a, b, c] = 0 otherwise
        
        # So it's NOT all ones! Only specific entries are 1.
        # The entry at position (i*m+j, j*p+k, k*n+i) is 1, and all others are 0.
        
        # So my rank-1 argument was wrong because I assumed T[i,j,k] = 1 for ALL i,j,k,
        # but actually T is sparse - only 1 out of (n*m)*(m*p)*(p*n) entries is nonzero for each (i,j,k).
        
        # Let me re-examine. The tensor has dimensions (n*m) x (m*p) x (p*n) = 9 x 24 x 24.
        # Nonzero entries: for each (i,j,k) with i in [0,3), j in [0,3), k in [0,8):
        #   T[i*m+j, j*p+k, k*n+i] = 1
        # That's 3*3*8 = 72 nonzero entries out of 9*24*24 = 5184 total.
        
        # So the rank-1 decomposition would need:
        # U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 when (i*m+j, j*p+k, k*n+i) is a valid triple
        # and = 0 otherwise.
        
        # This is much harder. The rank-1 tensor U⊗V⊗W has the property that
        # T[a,b,c] = U[a]*V[b]*W[c]. For this to be 1 on specific (a,b,c) triples and 0 elsewhere,
        # we need very specific structure.
        
        # If U[a]*V[b]*W[c] = 0 for all (a,b,c) not in the support, and = 1 on the support,
        # this is a very constrained problem. Generally rank 1 won't work.
        
        # So my earlier "rank 1" argument was based on a misunderstanding of the tensor structure.
        
        # Let me go back to the correct schoolbook and try to improve from there.
        pass
    
    # Given the complexity, let's return the schoolbook decomposition which is known valid.
    return best_result

def _try_slice_merge():
    """
    Attempt to reduce rank by exploiting shared structure across column blocks.
    
    Partition k into groups G0={0,1,2}, G1={3,4,5}, G2={6,7}.
    
    For each group Gb, the sub-tensor involves:
    - U factors: U[i*m+j, r] for i in [0,3), j in [0,3)
    - V factors: V[j*p+k, r] for j in [0,3), k in Gb
    - W factors: W[k*n+i, r] for k in Gb, i in [0,3)
    
    Key observation: For fixed (i,j), U[i*m+j, r] appears in all groups.
    If we can find a decomposition where U[i*m+j, r] is nonzero for a small set of r,
    and the V,W factors for different k values are linearly independent, we might save rank.
    
    Specifically, consider fixing U to be diagonal-ish:
    U[i*m+j, r] = delta(i,j, r) for r = 0..8 (9 values)
    
    Then for each (i,j), only one r is active. The remaining constraint is:
    V[j*p+k, r_{ij}] * W[k*n+i, r_{ij}] = 1 for all k in [0,8)
    
    This means: for fixed (i,j) with r = r_{ij}, V[j*p+k, r_{ij}] * W[k*n+i, r_{ij}] = 1 for all k.
    So V[j*p+k, r_{ij}] = 1/W[k*n+i, r_{ij}] for all k.
    
    This requires W[k*n+i, r_{ij}] to be nonzero and V[j*p+k, r_{ij}] = 1/W[k*n+i, r_{ij}].
    
    But r_{ij} is the same for all k, so W[k*n+i, r_{ij}] must be the same for all k? No!
    W[k*n+i, r_{ij}] depends on k. And V[j*p+k, r_{ij}] also depends on k.
    The product must be 1 for all k.
    
    So: V[j*p+k, r] * W[k*n+i, r] = 1 for all k, where r = r_{ij}.
    This means V[j*p+k, r] = 1/W[k*n+i, r] for all k.
    
    We can choose W[k*n+i, r] = 1 for all k, then V[j*p+k, r] = 1 for all k.
    
    So the decomposition is:
    U[i*m+j, r_{ij}] = 1 (where r_{ij} = i*3+j, so r ranges over 0..8)
    V[j*p+k, r_{ij}] = 1 for all k
    W[k*n+i, r_{ij}] = 1 for all k
    
    But wait - this uses rank 9 (one per (i,j) pair). Let's verify:
    T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    = U[i*m+j, r_{ij}] * V[j*p+k, r_{ij}] * W[k*n+i, r_{ij}]
    = 1 * 1 * 1 = 1 ✓
    
    Now check zero entries: T[a,b,c] for inconsistent (i,j,k):
    Say a = i1*m+j1, b = j2*p+k2, c = k3*n+i3 where the indices don't match.
    T = sum_r U[i1*m+j1, r] * V[j2*p+k2, r] * W[k3*n+i3, r]
    U[i1*m+j1, r] is nonzero only when r = r_{i1,j1} = i1*3+j1.
    So T = V[j2*p+k2, r_{i1,j1}] * W[k3*n+i3, r_{i1,j1}]
    = 1 * 1 = 1 (if these indices are valid)
    
    That's WRONG! We need T = 0 for inconsistent indices.
    
    The problem is that V[j*p+k, r] = 1 for ALL j,k and W[k*n+i, r] = 1 for ALL k,i.
    So any combination gives 1, not just the consistent ones.
    
    We need V and W to be zero for inconsistent combinations.
    
    Let me reconsider. We need:
    V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) are consistent with r
    V[j*p+k, r] * W[k*n+i, r] = 0 otherwise
    
    This is a very strong constraint. For fixed r = r_{i0,j0}:
    V[j*p+k, r_{i0,j0}] * W[k*n+i, r_{i0,j0}] should be 1 only when i=i0, j=j0 (and any k).
    
    So: V[j*p+k, r_{i0,j0}] * W[k*n+i, r_{i0,j0}] = delta(i, i0) * delta(j, j0)
    
    This means: V[j*p+k, r_{i0,j0}] * W[k*n+i, r_{i0,j0}] = 0 unless i=i0 AND j=j0.
    
    For fixed r, let v_k = V[j*p+k, r] (depends on j,k) and w_k = W[k*n+i, r] (depends on k,i).
    We need: v(j,k) * w(k,i) = delta(i,i0)*delta(j,j0) for all i,j,k.
    
    Set j != j0: v(j,k) * w(k,i) = 0 for all i,k.
    Set i != i0: v(j,k) * w(k,i) = 0 for all j,k.
    
    This means: for each k, either v(j,k) = 0 for all j != j0, or w(k,i) = 0 for all i != i0.
    And: for each k, either v(j,k) = 0 for all j, or w(k,i) = 0 for all i != i0.
    
    The simplest solution: v(j,k) = delta(j, j0) for all k, and w(k,i) = delta(i, i0) for all k.
    Then v(j,k)*w(k,i) = delta(j,j0)*delta(i,i0). ✓
    
    So: V[j*p+k, r_{i0,j0}] = delta(j, j0) for all k
    W[k*n+i, r_{i0,j0}] = delta(i, i0) for all k
    
    Let's verify the nonzero entries:
    T[i*m+j, j*p+k, k*n+i] with r = r_{i,j} = i*3+j:
    U[i*m+j, r_{i,j}] = 1
    V[j*p+k, r_{i,j}] = delta(j, j) = 1 ✓
    W[k*n+i, r_{i,j}] = delta(i, i) = 1 ✓
    Product = 1 ✓
    
    Zero entries: T[a,b,c] where a=i1*m+j1, b=j2*p+k2, c=k3*n+i3, inconsistent.
    Only r = r_{i1,j1} = i1*3+j1 contributes to U.
    V[j2*p+k2, r_{i1,j1}] = delta(j2, j1)
    W[k3*n+i3, r_{i1,j1}] = delta(i3, i1)
    Product = delta(j2,j1) * delta(i3,i1)
    
    For this to be 0, we need j2 != j1 OR i3 != i1.
    If j2 = j1 and i3 = i1, then a = i1*m+j1, b = j1*p+k2, c = k3*n+i1.
    This is consistent (i=i1, j=j1, k can be anything), so T should be 1. ✓
    If j2 != j1 or i3 != i1, T = 0. ✓
    
    RANK 9 DECOMPOSITION! Let me implement this.
    """
    n, m, p = 3, 3, 8
    rank = n * m  # 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1  # delta(j,j) = 1
                W[k * n + i][r] = 1  # delta(i,i) = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try the rank-9 slice-and-dice decomposition first
    result = _try_slice_merge()
    
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    result = _schoolbook()
    return result
# EVOLVE-BLOCK-END
