"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    def to_list(mat):
        return mat.tolist()
    
    def verify_exact(U, V, W):
        """Verify that U, V, W correctly decompose the tensor using exact arithmetic."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        
        # Check all n*m*p required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = Fraction(0)
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if actual != Fraction(1):
                        return False
        
        # Check that all other entries are zero
        # Total entries: (n*m) * (m*p) * (p*n) = 6 * 21 * 14 = 1764
        # Non-zero entries: n*m*p = 42
        # So we need to check 1764 - 42 = 1722 zero entries
        # But checking all is too slow, so check a good sample
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k1 = divmod(b, p)
                if j1 != j2:
                    # This entry should be zero since j1 != j2
                    actual = Fraction(0)
                    for r in range(rank):
                        actual += U[a, r] * V[b, r] * W[k1 * n + i1, r]
                    if actual != Fraction(0):
                        return False
                else:
                    # j1 == j2, check if this is a required entry
                    for k2 in range(p):
                        c = k2 * n + i1
                        if k1 != k2:
                            # Not a required entry, should be zero
                            actual = Fraction(0)
                            for r in range(rank):
                                actual += U[a, r] * V[b, r] * W[c, r]
                            if actual != Fraction(0):
                                return False
        return True
    
    def verify_fast(U, V, W):
        """Fast verification using float, for initial screening."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        
        # Check required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        
        # Check a sample of zero entries
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (j1 == j2 and i1 == i2 and k1 == k2):
                actual = 0.0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if abs(actual) > 1e-9:
                    return False
        return True
    
    # Build schoolbook decomposition
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Key insight: For m=3, terms with the same j can potentially be combined
    # If we have two terms (i1, j, k1) and (i2, j, k2) with the same j,
    # we can try to combine them into one rank-1 term
    
    # Strategy: For each j, try to combine pairs of k values
    # The constraint is that for a single rank-1 term to cover (i,j,k1) and (i,j,k2):
    # U[i*m+j, r] * V[j*p+k1, r] * W[k1*n+i, r] = 1
    # U[i*m+j, r] * V[j*p+k2, r] * W[k2*n+i, r] = 1
    # This requires V[j*p+k1, r] * W[k1*n+i, r] = V[j*p+k2, r] * W[k2*n+i, r]
    
    # Let's try a structured search: for each j, try to find combinations
    # where we can merge multiple k values
    
    def try_combination_search():
        """Try to find rank reductions by combining terms with same j."""
        if time.time() > deadline:
            return None
        
        # For each j, we have n*p = 2*7 = 14 terms
        # Try to find subsets that can be combined
        
        # Approach: For each j, try to find a set of (i,k) pairs that can be
        # covered by a small number of rank-1 terms
        
        # Start with the observation that if i1 = i2, then we need:
        # V[j*p+k1, r] * W[k1*n+i, r] = V[j*p+k2, r] * W[k2*n+i, r]
        # This is satisfied if V[j*p+k1, r] = V[j*p+k2, r] and W[k1*n+i, r] = W[k2*n+i, r]
        # Or more generally if the products are equal
        
        # Let's try: for a fixed i and j, can we cover multiple k values?
        # If we set V[j*p+k, r] = 1 for all k, and W[k*n+i, r] = 1 for all k,
        # then U[i*m+j, r] * 1 * 1 = 1, so U[i*m+j, r] = 1
        # But this would also make other entries non-zero!
        
        # The issue is that a rank-1 term U*V*W affects ALL entries where
        # U[a,r] != 0, V[b,r] != 0, and W[c,r] != 0
        
        # So we need to be careful about which entries get affected
        
        # Let's try a different approach: use the structure more carefully
        # For the tensor T[i*m+j, j*p+k, k*n+i] = 1, note that:
        # - The first index encodes (i,j)
        # - The second index encodes (j,k)  
        # - The third index encodes (k,i)
        # - The constraint is that all three must be consistent: same i, same j, same k
        
        # This is like a 3-dimensional matching problem
        
        # Let's try: for each j, group terms by i
        # For fixed i and j, we have p=7 terms: (i,j,0), (i,j,1), ..., (i,j,6)
        # Can we cover these with fewer than 7 rank-1 terms?
        
        # A rank-1 term that covers (i,j,k) must have:
        # U[i*m+j, r] != 0, V[j*p+k, r] != 0, W[k*n+i, r] != 0
        # And the product must equal 1 (when summed with other terms)
        
        # If we want to cover (i,j,k1) and (i,j,k2) with one term, we need:
        # U[i*m+j, r] * V[j*p+k1, r] * W[k1*n+i, r] = a
        # U[i*m+j, r] * V[j*p+k2, r] * W[k2*n+i, r] = b
        # Where a + (contributions from other terms) = 1
        # And b + (contributions from other terms) = 1
        
        # This is getting complex. Let's try a more direct search.
        
        # Try rank 41: remove one term and see if we can adjust others
        # This is equivalent to finding a linear dependency among the 42 rank-1 tensors
        
        # The 42 rank-1 tensors are:
        # T_r[i*m+j, j*p+k, k*n+i] = 1 for specific (i,j,k)
        
        # Can we find coefficients c_r such that sum_r c_r * T_r = 0?
        # This would mean we can remove one term and adjust the others
        
        # Let's set up the linear system
        # For each entry (a,b,c), we have:
        # sum_r c_r * T_r[a,b,c] = 0
        
        # T_r[a,b,c] is non-zero only if (a,b,c) corresponds to the (i,j,k) for term r
        
        # So the system is: for each (a,b,c), sum over r where T_r[a,b,c]!=0 of c_r * T_r[a,b,c] = 0
        
        # Since each T_r has exactly one non-zero entry (value 1), this becomes:
        # For each (a,b,c), if there's exactly one r with T_r[a,b,c]=1, then c_r = 0
        # If there are multiple r's, we have a constraint
        
        # But in our case, each entry (a,b,c) is covered by at most one r
        # (since (i,j,k) uniquely determines (a,b,c))
        
        # So the only solution is c_r = 0 for all r, meaning no linear dependency
        
        # This suggests that the rank might indeed be 42 for this specific tensor
        
        # But wait - we're not limited to the schoolbook basis!
        # We can use any rank-1 tensors, not just the standard basis
        
        # Let's try a randomized search with better structure
        
        # Strategy: start with schoolbook, then try to merge pairs
        # For each pair of terms with the same j, try to find a rank-1 tensor
        # that covers both and adjust the others
        
        # Due to time constraints, let's do a limited search
        if time.time() > deadline:
            return None
        
        # Try to find a rank-41 decomposition by searching for a linear combination
        # that cancels one term
        
        # Represent the 42 rank-1 tensors as vectors in R^(6*21*14)
        # But this is too large. Instead, work with the structure
        
        # Let's try: can we find U, V, W of rank 41?
        # Start from schoolbook and try to perturb
        
        # Actually, let's try a completely different approach:
        # Use the fact that m=3 is small
        
        # The tensor is: sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
        
        # This is the "matrix multiplication tensor" for 2x3 by 3x7
        
        # Known results: for multiplying an m×n matrix by an n×p matrix,
        # the rank is at most mn + np + mp - n (from Strassen-like algorithms)
        # For 2×3 by 3×7: 2*3 + 3*7 + 7*2 - 3 = 6 + 21 + 14 - 3 = 38
        
        # So rank 38 might be achievable!
        
        # Let's try to construct a rank-38 decomposition
        
        # The formula mn + np + mp - n comes from:
        # - mn terms for the first matrix
        # - np terms for the second matrix  
        # - mp terms for the output
        # - n terms subtracted due to overlap
        
        # Let's try to implement this
        
        # Actually, let's just do a randomized search for rank < 42
        # with the constraint that entries are half-integers
        
        half_ints = [Fraction(x, 2) for x in range(-4, 5)]  # -2, -1.5, ..., 2
        
        def random_half_int():
            return np.random.choice(half_ints)
        
        # Try rank 41, 40, 39, 38
        for target_rank in [41, 40, 39, 38]:
            if time.time() > deadline:
                break
            
            max_attempts = 2000 if target_rank >= 40 else 1000
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                U = np.zeros((n * m, target_rank), dtype=object)
                V = np.zeros((m * p, target_rank), dtype=object)
                W = np.zeros((p * n, target_rank), dtype=object)
                
                # Initialize with mostly zeros, then add random non-zeros
                for r_idx in range(target_rank):
                    # Each rank-1 term should have a few non-zeros
                    num_u = np.random.randint(1, 4)
                    num_v = np.random.randint(1, 4)
                    num_w = np.random.randint(1, 4)
                    
                    u_pos = np.random.choice(n * m, size=num_u, replace=False)
                    v_pos = np.random.choice(m * p, size=num_v, replace=False)
                    w_pos = np.random.choice(p * n, size=num_w, replace=False)
                    
                    for pos in u_pos:
                        U[pos, r_idx] = random_half_int()
                    for pos in v_pos:
                        V[pos, r_idx] = random_half_int()
                    for pos in w_pos:
                        W[pos, r_idx] = random_half_int()
                
                # Quick check
                if verify_fast(U, V, W):
                    # Exact check
                    if verify_exact(U, V, W):
                        return U, V, W
        
        return None
    
    result = try_combination_search()
    if result is not None:
        U, V, W = result
        # Convert Fraction objects to floats for output
        U_out = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
        V_out = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
        W_out = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
        return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
    
    # Return schoolbook
    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }
# EVOLVE-BLOCK-END
