import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    rank_school = n * m * p  # 48
    
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Generate all possible rank-1 tensor support patterns
    # Each pattern is (i, j, k) representing the indices
    all_patterns = list(product(range(n), range(m), range(p)))
    
    # For each target rank, try to find a valid assignment
    # We'll use a more structured approach: try to assign signs to a subset of patterns
    
    def try_rank(target_rank, patterns_subset):
        """
        Given a subset of patterns (each is (i,j,k)), try to assign signs
        such that the tensor decomposition works.
        """
        if len(patterns_subset) < target_rank:
            return None
        
        # Select target_rank patterns from the subset
        if len(patterns_subset) > target_rank:
            # Try different subsets of size target_rank
            from itertools import combinations
            for combo in combinations(patterns_subset, target_rank):
                result = _try_assignment(list(combo))
                if result is not None:
                    return result
        else:
            result = _try_assignment(list(patterns_subset))
            if result is not None:
                return result
        
        return None
    
    def _try_assignment(patterns):
        """
        Try to assign ±1 signs to each pattern so that the tensor equation holds.
        This is a system of linear equations over the reals with sign constraints.
        """
        r = len(patterns)
        
        # Build the constraint matrix
        # For each (i,j,k), we need sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # where U[i,j,r] is nonzero only if pattern r has (i,j)
        # Let's set up: for each rank r, pattern is (i_r, j_r, k_r)
        # U[i_r, j_r, r] = a_r, V[j_r, k_r, r] = b_r, W[k_r, i_r, r] = c_r
        # with a_r, b_r, c_r ∈ {±1}
        # Constraint: for each (i,j,k), sum over r where (i_r,j_r,k_r)=(i,j,k) of a_r*b_r*c_r = 1
        # But if no pattern has (i,j,k), then sum = 0 ≠ 1, so we need at least one pattern per (i,j,k)
        
        # Check coverage
        from collections import defaultdict
        pattern_counts = defaultdict(list)
        for idx, (i, j, k) in enumerate(patterns):
            pattern_counts[(i, j, k)].append(idx)
        
        # Every (i,j,k) must be covered
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in pattern_counts:
                        return None
        
        # For each (i,j,k), the sum of products must be 1
        # If only one pattern covers (i,j,k), then a*b*c must be 1
        # If multiple patterns cover, we need sum = 1
        
        # This is a constraint satisfaction problem
        # Let's try a simple approach: for each (i,j,k), if only one pattern, set product=1
        # Otherwise, try to satisfy
        
        # Initialize with all +1
        signs = [1] * r
        
        # For each (i,j,k) with exactly one pattern, we can fix the product
        # But we need consistency across patterns
        
        # Simple heuristic: try all 2^r sign assignments for small r
        if r <= 20:
            for sign_combo in product([-1, 1], repeat=r):
                valid = True
                for i in range(n):
                    if not valid:
                        break
                    for j in range(m):
                        if not valid:
                            break
                        for k in range(p):
                            s = 0
                            for idx in pattern_counts[(i, j, k)]:
                                pi, pj, pk = patterns[idx]
                                s += sign_combo[idx] * 1 * 1  # a*b*c, but b and c also have signs
                            # Actually, we need to track a_r, b_r, c_r separately
                            # Let's redo: each pattern r has three signs: a_r for U, b_r for V, c_r for W
                            # Product a_r*b_r*c_r contributes to (i_r,j_r,k_r)
                            pass
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    # Build matrices
                    U = np.zeros((n*m, r))
                    V = np.zeros((m*p, r))
                    W = np.zeros((p*n, r))
                    for idx, (i, j, k) in enumerate(patterns):
                        U[i*m + j, idx] = sign_combo[idx]
                        V[j*p + k, idx] = 1
                        W[k*n + i, idx] = 1
                    if verify(U, V, W):
                        return (U, V, W)
        
        # For larger r, use a smarter approach
        # Group patterns by their (i,j,k)
        # For each group, the sum of products must be 1
        # This is a system of equations
        
        # Try a greedy approach
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        for idx, (i, j, k) in enumerate(patterns):
            U[i*m + j, idx] = 1
            V[j*p + k, idx] = 1
            W[k*n + i, idx] = 1
        
        # Now adjust signs to satisfy constraints
        # This is complex; for now, return None and let random search handle it
        return None
    
    # Try structured search for ranks below schoolbook
    np.random.seed(42)
    
    for target_rank in range(30, rank_school):
        if time.time() > deadline - 15:
            break
        
        # Try random subsets of patterns
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            # Select target_rank random patterns
            selected = np.random.choice(len(all_patterns), size=target_rank, replace=False)
            patterns_subset = [all_patterns[idx] for idx in selected]
            
            result = try_rank(target_rank, patterns_subset)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
        
        if best_rank < rank_school:
            break
    
    # Fallback: original random search
    if best_rank == rank_school:
        for target_rank in range(46, rank_school):
            if time.time() > deadline - 10:
                break
            
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                
                U_rand = np.zeros((n*m, target_rank))
                V_rand = np.zeros((m*p, target_rank))
                W_rand = np.zeros((p*n, target_rank))
                
                for r in range(target_rank):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    
                    U_rand[i*m+j, r] = np.random.choice([-1, 1])
                    V_rand[j*p+k, r] = np.random.choice([-1, 1])
                    W_rand[k*n+i, r] = np.random.choice([-1, 1])
                
                if verify(U_rand, V_rand, W_rand):
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = target_rank
                    break
            
            if best_rank < rank_school:
                break
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
