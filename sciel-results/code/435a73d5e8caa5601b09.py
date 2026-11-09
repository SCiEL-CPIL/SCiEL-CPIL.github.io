"""Branch-and-bound maximum independent set in 3-uniform hypergraph."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Candidate set: all integers from 2 to 256
    all_candidates = list(range(2, 257))
    n_total = len(all_candidates)
    # Index mapping: candidate value -> index 0..254
    val_to_idx = {v: i for i, v in enumerate(all_candidates)}
    idx_to_val = all_candidates  # idx_to_val[i] = i+2
    
    # Build conflict structure:
    # For each pair (i,j) with i<j, determine which k (k!=i,j) conflict with {i,j}
    # Conflict: for distinct a,b,c in S, a ∤ b*c
    # So triple {i,j,k} is invalid if any of:
    #   idx_to_val[i] | (idx_to_val[j] * idx_to_val[k])
    #   idx_to_val[j] | (idx_to_val[i] * idx_to_val[k])
    #   idx_to_val[k] | (idx_to_val[i] * idx_to_val[j])
    
    # Precompute: for each pair (i,j), the set of k that conflict
    conflict_with_pair = [[None] * n_total for _ in range(n_total)]
    
    for i in range(n_total):
        vi = idx_to_val[i]
        for j in range(i + 1, n_total):
            vj = idx_to_val[j]
            prod_ij = vi * vj
            bad_ks = set()
            for k in range(n_total):
                if k == i or k == j:
                    continue
                vk = idx_to_val[k]
                if (vj * vk) % vi == 0:
                    bad_ks.add(k)
                    continue
                if (vi * vk) % vj == 0:
                    bad_ks.add(k)
                    continue
                if prod_ij % vk == 0:
                    bad_ks.add(k)
                    continue
            conflict_with_pair[i][j] = bad_ks
            conflict_with_pair[j][i] = bad_ks
    
    # Precompute v_bad_pairs: for each vertex v, list of (a,b) pairs such that v conflicts with pair {a,b}
    v_bad_pairs = [set() for _ in range(n_total)]
    for i in range(n_total):
        for j in range(i + 1, n_total):
            for k in conflict_with_pair[i][j]:
                v_bad_pairs[k].add((i, j))
    
    # Convert to lists for faster iteration
    v_bad_pairs_list = [list(s) for s in v_bad_pairs]
    
    def is_compatible(v, S_mask):
        """Check if vertex v can be added to set represented by S_mask."""
        for (a, b) in v_bad_pairs_list[v]:
            if (S_mask >> a) & 1 and (S_mask >> b) & 1:
                return False
        return True
    
    # Greedy initial solution
    def greedy_solution(order):
        """Build a solution greedily following the given order."""
        S_mask = 0
        S_set = set()
        for v in order:
            if is_compatible(v, S_mask):
                S_mask |= (1 << v)
                S_set.add(v)
        return S_mask, S_set
    
    # 1-for-1 swap local search: try removing one element and adding another
    def local_search_1swap(S_mask, S_set, max_iterations=100):
        """Try 1-for-1 swaps to improve the solution."""
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 10:
            improved = False
            iterations += 1
            S_list = list(S_set)
            for i, rem in enumerate(S_list):
                # Try removing rem and adding some other vertex
                S_mask_without = S_mask & ~(1 << rem)
                S_set_without = S_set - {rem}
                for add in range(n_total):
                    if add in S_set:
                        continue
                    if not is_compatible(add, S_mask_without):
                        continue
                    # Check if the new set is valid
                    new_S_mask = S_mask_without | (1 << add)
                    new_S_set = S_set_without | {add}
                    # This is always valid if is_compatible passed, since we're only adding one element
                    # and it's compatible with all pairs in S_set_without
                    if len(new_S_set) > len(S_set):
                        # Can't happen since we remove one and add one
                        pass
                    # Actually 1-for-1 swap keeps same size, so we need a different criterion
                    # Let's skip this and do 1-for-2 or just accept if it enables further improvements
                    # For now, let's just try to add elements without removing
            # Try adding elements without removing
            for add in range(n_total):
                if add in S_set:
                    continue
                if is_compatible(add, S_mask):
                    S_mask |= (1 << add)
                    S_set.add(add)
                    improved = True
                    break
        return S_mask, S_set
    
    # Try multiple greedy orderings
    best_mask = 0
    best_size = 0
    best_set = set()
    
    orders = [
        list(range(n_total)),  # ascending
        list(range(n_total - 1, -1, -1)),  # descending
    ]
    random.seed(42)
    for _ in range(40):
        shuffled = list(range(n_total))
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline - 25:
            break
        mask, s = greedy_solution(order)
        # Apply local search
        mask, s = local_search_1swap(mask, s, max_iterations=50)
        if len(s) > best_size:
            best_mask = mask
            best_size = len(s)
            best_set = s
    
    # Branch and bound
    def branch_and_bound(S_mask, candidates_mask, current_size):
        """
        S_mask: bitmask of vertices in current solution
        candidates_mask: bitmask of vertices that can potentially be added (compatible with S)
        current_size: number of vertices in current solution
        """
        nonlocal best_mask, best_size
        
        if time.time() > deadline - 5:
            return
        
        # Update upper bound
        upper_bound = current_size + bin(candidates_mask).count('1')
        if upper_bound <= best_size:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_mask = S_mask
                best_size = current_size
            return
        
        # Select a vertex to branch on: pick the most constrained (most bad pairs among candidates)
        # For simplicity, pick lowest-index candidate
        v = (candidates_mask & (-candidates_mask)).bit_length() - 1
        
        # Branch 1: include v
        new_S_mask = S_mask | (1 << v)
        new_candidates = 0
        temp_mask = candidates_mask
        while temp_mask:
            u = (temp_mask & (-temp_mask)).bit_length() - 1
            temp_mask &= temp_mask - 1
            if u != v and is_compatible(u, new_S_mask):
                new_candidates |= (1 << u)
        
        branch_and_bound(new_S_mask, new_candidates, current_size + 1)
        
        # Branch 2: exclude v
        new_candidates2 = candidates_mask & ~(1 << v)
        branch_and_bound(S_mask, new_candidates2, current_size)
    
    # Run branch and bound from the greedy solution
    init_candidates = 0
    for v in range(n_total):
        if not (best_mask >> v) & 1 and is_compatible(v, best_mask):
            init_candidates |= (1 << v)
    
    branch_and_bound(best_mask, init_candidates, best_size)
    
    # Final verification
    result = [idx_to_val[i] for i in range(n_total) if (best_mask >> i) & 1]
    
    def is_valid_set(s_list):
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    b = s_list[j]
                    c = s_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    if not is_valid_set(result):
        current = set()
        current_mask = 0
        for c in all_candidates:
            idx = val_to_idx[c]
            if is_compatible(idx, current_mask):
                current.add(c)
                current_mask |= (1 << idx)
        result = sorted(current)
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
