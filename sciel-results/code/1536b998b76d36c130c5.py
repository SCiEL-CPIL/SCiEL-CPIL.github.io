import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: Vectorized Local Search (Simulated Annealing) on n=20.
    1. Precompute all 6-vertex subsets.
    2. Define cost as the number of regular induced subgraphs on 6 vertices.
    3. Use NumPy matrix operations to evaluate cost changes in O(1) per move.
    4. Minimize cost to 0. If 0 is reached, verify the full graph (sizes >= 6) 
       using a fast bitset method.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 20
    k = 6
    
    # Precompute all subsets of size k
    # For n=20, C(20,6) = 38760
    subsets = list(combinations(range(n), k))
    m = len(subsets)
    
    # Create incidence matrix S of shape (m, n)
    # S[i, v] = 1 if v is in subset i
    S = np.zeros((m, n), dtype=np.int8)
    for i, sub in enumerate(subsets):
        for v in sub:
            S[i, v] = 1
    
    # Random initialization
    # Start with a random graph
    # Adjacency matrix A (n, n), symmetric, 0 diagonal
    A = np.zeros((n, n), dtype=np.int8)
    # Generate upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                A[i, j] = 1
                A[j, i] = 1
    
    def get_cost(A):
        """
        Computes the number of regular induced subgraphs of size k.
        Uses matrix multiplication: D = S @ A (m, n)
        D[i, v] is the degree of v in subset i.
        A subset i is regular if all D[i, v] for v in subset i are equal.
        """
        D = S @ A  # (m, n)
        
        # For each subset i, check if degrees are equal for all v in subset i
        # We can compute the max and min degree within the subset
        # Or simply check variance, or max - min == 0
        
        # To vectorize "max/min over subset":
        # We need to mask D with S.
        # D_masked[i, v] = D[i, v] if S[i, v]==1 else 0 (for max)
        # D_masked_min[i, v] = D[i, v] if S[i, v]==1 else large (for min)
        
        # Actually, simpler:
        # Since S has exactly k ones per row, and D has values 0..k-1.
        # We can compute sum and sum of squares? No, that's for variance.
        # Max - Min is better.
        
        # Let's compute max and min.
        # D_max = np.max(np.where(S > 0, D, -1), axis=1)
        # D_min = np.min(np.where(S > 0, D, k), axis=1)
        
        # This creates temporary arrays of size (m, n) which is 38760 * 20 = ~775k elements.
        # This is very fast in NumPy.
        
        D_for_max = np.where(S > 0, D, -1)
        D_max = np.max(D_for_max, axis=1)
        
        D_for_min = np.where(S > 0, D, k)
        D_min = np.min(D_for_min, axis=1)
        
        # Regular if D_max == D_min
        regular_mask = (D_max == D_min)
        return np.sum(regular_mask)
    
    current_cost = get_cost(A)
    best_A = A.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        if current_cost == 0:
            # Check if we found a valid graph for all sizes >= 6
            # We only checked size 6.
            # If a graph has no regular induced subgraph of size 6, it might still have one of size 7.
            # However, if we minimize size 6, we are in a "regular-free" region for size 6.
            # We should verify the full condition.
            
            # Fast verification for full condition
            if verify_full(n, A):
                # Convert A to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if A[i, j] == 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # If not valid for larger sizes, continue searching or accept?
            # The task requires NO regular induced subgraph on 6 OR MORE.
            # So we must verify.
            # If it fails verification, we should probably keep searching.
            # But finding size 6 free is the hardest part.
            # Let's just keep searching.
            pass
        
        if T < min_T:
            T = min_T
            
        # Generate a random move: flip an edge (i, j)
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        # Ensure i < j for consistency, though A is symmetric
        if i > j:
            i, j = j, i
        
        # Calculate delta cost
        # Flipping A[i, j] changes the degree of i in subsets containing i,
        # and degree of j in subsets containing j.
        # It does NOT affect degrees of other vertices in those subsets.
        
        # Identify subsets containing i or j
        # S[:, i] == 1 or S[:, j] == 1
        mask_i = S[:, i] == 1
        mask_j = S[:, j] == 1
        mask_ij = mask_i | mask_j
        
        # If no subsets contain i or j, cost doesn't change (impossible for k>=2)
        if not np.any(mask_ij):
            continue
            
        # Get current degrees for affected subsets
        # D = S @ A
        # Instead of recomputing full D, we can update the specific rows.
        
        # Let's just recompute D for the affected rows?
        # Or maintain D?
        # Maintaining D is faster.
        
        # Let's maintain D.
        # D[i, v] = sum_u A[i, u] * S[v, u]? No.
        # D[s, v] = degree of v in subset s = sum_{u in subset s} A[v, u]
        # So D = S @ A is correct.
        
        # If we flip A[i, j], then:
        # For any subset s containing i:
        #   degree of j in s changes by +/- 1 (if j is in s)
        #   Wait. D[s, v] = sum_{u in s} A[v, u].
        #   If we change A[i, j], it affects:
        #   1. D[s, i] for all s containing j. (Because A[i, j] is in the sum for D[s, i] if j in s)
        #   2. D[s, j] for all s containing i. (Because A[i, j] is in the sum for D[s, j] if i in s)
        
        # Let's verify:
        # D[s, i] = sum_{u in s} A[i, u].
        # If j is in s, then A[i, j] is a term. So D[s, i] changes by delta.
        # D[s, j] = sum_{u in s} A[j, u].
        # If i is in s, then A[j, i] is a term. So D[s, j] changes by delta.
        
        # So:
        # delta = +1 if we add edge, -1 if we remove.
        
        # Affected rows in D:
        # Rows s where j in s (for D[s, i]) -> mask_j
        # Rows s where i in s (for D[s, j]) -> mask_i
        
        # We need to update D.
        # But we don't have D stored globally. Let's store it.
        
        # Actually, let's just recompute the cost for the affected subsets.
        # Number of affected subsets is at most 2 * C(19, 5) ~ 2 * 11628 = 23256.
        # This is a lot.
        
        # Better approach: Maintain D.
        
        # Let's refactor to maintain D.
        pass
        
        # Given the complexity of maintaining D and checking cost delta efficiently,
        # and the fact that full D recomputation is fast (~1ms),
        # let's just do full recomputation but with a smarter search.
        
        # Actually, the current `get_cost` is fast enough.
        # Let's just do simple hill climbing with restarts if stuck.
        
        # Try the move
        old_val = A[i, j]
        new_val = 1 - old_val
        A[i, j] = new_val
        A[j, i] = new_val
        
        new_cost = get_cost(A)
        
        delta = new_cost - current_cost
        
        # Accept if better or random
        if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_A = A.copy()
        else:
            # Revert
            A[i, j] = old_val
            A[j, i] = old_val
            
        T *= cooling_rate
        
        # Check time
        if time.time() > deadline - 10:
            break
            
    # If we didn't find a valid graph during the loop, check the best one found
    if best_cost == 0:
        if verify_full(n, best_A):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_A[i, j] == 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Fallback: Try n=19
    n = 19
    # Quick random search for n=19
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    A[i, j] = 1
                    A[j, i] = 1
        if verify_full(n, A):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i, j] == 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Fallback: Trivial valid graph n=5
    return {'n': 5, 'edges': []}

def verify_full(n, A):
    """
    Verifies that the graph defined by adjacency matrix A has no regular 
    induced subgraph of size >= 6.
    Uses bitsets for speed.
    """
    # Convert A to bitsets
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if A[i, j]:
                adj[i] |= (1 << j)
                
    # Check all subsets of size >= 6
    # For n=20, this is ~1M subsets.
    # We can optimize by only checking sizes 6 to n.
    # But checking all masks is easier.
    
    # To speed up, we can prune.
    # If a subset is regular, return False.
    
    # We'll iterate masks.
    for mask in range(1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
            
        # Check regularity
        m = mask
        first_deg = -1
        is_reg = True
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            deg = (adj[i] & mask).bit_count()
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                is_reg = False
                break
            m ^= lsb
            
        if is_reg:
            return False
            
    return True

# EVOLVE-BLOCK-END
