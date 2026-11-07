import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # 1. Identify primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    # Strategy: Large Primes + Small Smooth Composites
    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 13]
    
    # Generate 13-smooth numbers
    smooth_candidates = []
    for n in range(2, N+1):
        if is_prime[n]:
            if n in small_primes:
                smooth_candidates.append(n)
        else:
            m = n
            smooth = True
            for p in small_primes:
                while m % p == 0:
                    m //= p
            if m == 1:
                smooth_candidates.append(n)
            
    # Map smooth candidates to indices for bitset operations
    num_smooth = len(smooth_candidates)
    smooth_idx = {v: i for i, v in enumerate(smooth_candidates)}
    
    # Precompute conflict bitsets:
    # For each pair (i,j) in smooth candidates, compute which k conflicts with (i,j)
    # A triple (a,b,c) is invalid if a divides b*c
    # We need: for any three distinct a,b,c in our set, a does NOT divide b*c
    
    # conflict_bitset[i][j] = bitmask of all k != i,j such that smooth_candidates[k] divides smooth_candidates[i]*smooth_candidates[j]
    # OR smooth_candidates[i] divides smooth_candidates[j]*smooth_candidates[k]
    # OR smooth_candidates[j] divides smooth_candidates[i]*smooth_candidates[k]
    
    # Actually, let's think more carefully.
    # For a set S to be valid: for every triple (a,b,c) of distinct elements in S, a does not divide b*c.
    # 
    # Let's precompute: for each pair (i,j), the set of k values that would create a conflict.
    # A conflict exists in triple (i,j,k) if:
    #   smooth[i] divides smooth[j]*smooth[k], OR
    #   smooth[j] divides smooth[i]*smooth[k], OR
    #   smooth[k] divides smooth[i]*smooth[j]
    
    # conflict_pair[i][j] = bitmask of k (k != i, k != j) such that triple (i,j,k) is invalid
    
    conflict_pair = [[0] * num_smooth for _ in range(num_smooth)]
    
    for i in range(num_smooth):
        si = smooth_candidates[i]
        for j in range(num_smooth):
            if i == j:
                continue
            sj = smooth_candidates[j]
            bitmask = 0
            for k in range(num_smooth):
                if k == i or k == j:
                    continue
                sk = smooth_candidates[k]
                # Check if any of the three divisibility conditions hold
                if (sj * sk) % si == 0 or (si * sk) % sj == 0 or (si * sj) % sk == 0:
                    bitmask |= (1 << k)
            conflict_pair[i][j] = bitmask
    
    # Also precompute: for adding a new element y to a valid set S,
    # check if y creates conflicts with any pair in S.
    # y conflicts with pair (i,j) in S if:
    #   smooth[i] divides smooth[j]*smooth[y], OR
    #   smooth[j] divides smooth[i]*smooth[y], OR
    #   smooth[y] divides smooth[i]*smooth[j]
    
    # Let's precompute for each y, a list of (i,j) pairs that conflict with y
    # Actually, let's use a different approach for the branch and bound.
    
    # For the branch and bound on smooth candidates:
    # We want to find the maximum independent set in the 3-uniform hypergraph.
    # 
    # Key insight: if we have a current set S (as a bitmask), and we consider adding vertex v,
    # we need to check that no existing pair (i,j) in S conflicts with v.
    # 
    # Let's precompute: conflict_with_v[v] = for each v, a list of pairs (i,j) such that
    # triple (i,j,v) is invalid. Then when checking if v can be added to S, we check
    # if any pair in conflict_with_v[v] is fully contained in S.
    
    # Alternative: precompute for each v, a bitmask-based structure.
    # For vertex v, define: 
    #   div_by_v[i] = True if smooth[v] divides smooth[i]*smooth[j] for some j... 
    # This gets complex. Let me use a simpler approach.
    
    # For each vertex v, precompute the set of pairs (i,j) with i<j, i!=v, j!=v,
    # such that (i,j,v) is an invalid triple.
    
    invalid_pairs_for_v = [[] for _ in range(num_smooth)]
    for i in range(num_smooth):
        for j in range(num_smooth):
            if i >= j:
                continue
            for k in range(num_smooth):
                if k == i or k == j:
                    continue
                si, sj, sk = smooth_candidates[i], smooth_candidates[j], smooth_candidates[k]
                if (sj * sk) % si == 0 or (si * sk) % sj == 0 or (si * sj) % sk == 0:
                    # Triple (i,j,k) is invalid
                    invalid_pairs_for_v[k].append((i, j))
    
    def can_add_to_set(S_mask, v):
        """Check if vertex v can be added to the set represented by S_mask."""
        if S_mask & (1 << v):
            return False
        for (i, j) in invalid_pairs_for_v[v]:
            if (S_mask & (1 << i)) and (S_mask & (1 << j)):
                return False
        return True
    
    def get_conflicts_with(S_mask, v):
        """Get the set of vertices in S_mask that conflict with v (i.e., must be removed)."""
        # v conflicts with vertex u in S if there exists w in S, w != u, w != v,
        # such that (u, w, v) is an invalid triple.
        # Actually, we need: for the set S ∪ {v} to be invalid, there must exist
        # a pair (i,j) in S such that (i,j,v) is invalid.
        # The vertices that "cause" the conflict are i and j.
        # But for pruning, we want: if we add v, which existing vertices must be removed?
        # If (i,j,v) is invalid, then at least one of i or j must be removed.
        # This is complex. Let's just use can_add_to_set for now.
        pass
    
    # Branch and bound to find maximum independent set in smooth candidates
    # Use a recursive approach with bitset pruning
    
    best_smooth_set = [0] * num_smooth  # 1 if included
    best_smooth_size = 0
    
    # Order vertices by degree (number of conflicts) - process high-degree first
    vertex_degrees = [len(invalid_pairs_for_v[v]) for v in range(num_smooth)]
    vertices_ordered = sorted(range(num_smooth), key=lambda x: -vertex_degrees[x])
    
    # For efficiency, precompute for each vertex v, the bitmask of all vertices
    # that are in any invalid pair with v (i.e., vertices u such that (u, w, v) is
    # invalid for some w)
    
    def branch_and_bound(pos, current_mask, current_size):
        nonlocal best_smooth_set, best_smooth_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_size + remaining vertices
        remaining = num_smooth - pos
        if current_size + remaining <= best_smooth_size:
            return
        
        if pos == num_smooth:
            if current_size > best_smooth_size:
                best_smooth_size = current_size
                best_smooth_set = [(1 if (current_mask & (1 << i)) else 0) for i in range(num_smooth)]
            return
        
        v = vertices_ordered[pos]
        
        # Option 1: Don't include v
        branch_and_bound(pos + 1, current_mask, current_size)
        
        if time.time() > deadline:
            return
        
        # Option 2: Include v (if possible)
        if can_add_to_set(current_mask, v):
            new_mask = current_mask | (1 << v)
            branch_and_bound(pos + 1, new_mask, current_size + 1)
    
    # The above is too slow for 50+ vertices. Let's use a better approach.
    # Use a greedy + local search hybrid with bitset operations.
    
    # First, use multiple greedy orderings
    def greedy_construct(ordering):
        """Greedy: add vertices in the given order if they don't create conflicts."""
        S_mask = 0
        S_list = []
        for v in ordering:
            if can_add_to_set(S_mask, v):
                S_mask |= (1 << v)
                S_list.append(v)
        return S_mask, S_list
    
    best_mask = 0
    best_list = []
    
    # Try different orderings
    orderings = [
        sorted(range(num_smooth), key=lambda x: -vertex_degrees[x]),  # high degree first
        sorted(range(num_smooth), key=lambda x: vertex_degrees[x]),   # low degree first
        sorted(range(num_smooth), key=lambda x: -smooth_candidates[x]),  # large values first
        sorted(range(num_smooth), key=lambda x: smooth_candidates[x]),   # small values first
    ]
    
    # Add some random orderings
    for _ in range(5):
        random_order = list(range(num_smooth))
        random.shuffle(random_order)
        orderings.append(random_order)
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        mask, lst = greedy_construct(ordering)
        if len(lst) > len(best_list):
            best_list = lst
            best_mask = mask
    
    # Local search: try 1-for-1 swaps
    S = set(best_list)
    C = [v for v in range(num_smooth) if v not in S]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        C_sorted = sorted(C, key=lambda x: -vertex_degrees[x])
        
        for x in S_list:
            if time.time() > deadline - 5:
                break
            S_minus_x = S - {x}
            
            for y in C_sorted:
                # Check if S_minus_x ∪ {y} is valid
                valid = True
                for (i, j) in invalid_pairs_for_v[y]:
                    if i in S_minus_x and j in S_minus_x:
                        valid = False
                        break
                if valid:
                    S = S_minus_x | {y}
                    C = [c for c in C if c != y]
                    improved = True
                    break
            if improved:
                break
    
    # Try 2-for-1 swaps
    S = set(best_list)
    C = [v for v in range(num_smooth) if v not in S]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        C_sorted = sorted(C, key=lambda x: -vertex_degrees[x])
        
        for x in S_list:
            if time.time() > deadline - 5:
                break
            S_minus_x = S - {x}
            
            valid_ys = []
            for y in C_sorted:
                valid = True
                for (i, j) in invalid_pairs_for_v[y]:
                    if i in S_minus_x and j in S_minus_x:
                        valid = False
                        break
                if valid:
                    valid_ys.append(y)
            
            found_swap = False
            for y in valid_ys:
                if found_swap:
                    break
                S_prime = S_minus_x | {y}
                for z in C_sorted:
                    if z == y:
                        continue
                    valid = True
                    for (i, j) in invalid_pairs_for_v[z]:
                        if i in S_prime and j in S_prime:
                            valid = False
                            break
                    if valid:
                        S = S_prime | {z}
                        C = [c for c in C if c not in {y, z}]
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break
    
    smooth_selected = [smooth_candidates[v] for v in S]
    
    # Combine with large primes
    final_set = list(set(large_primes + smooth_selected))
    
    # Verify final set
    def is_valid_subset(subset):
        s = set(subset)
        items = list(s)
        n = len(items)
        for i in range(n):
            a = items[i]
            for j in range(n):
                if i == j: continue
                b = items[j]
                for k in range(n):
                    if i == k or j == k: continue
                    c = items[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    if not is_valid_subset(final_set):
        # Fallback: just use large primes
        final_set = large_primes
    
    # Ensure all elements are Python ints
    final_set = [int(x) for x in final_set]
    
    return final_set
