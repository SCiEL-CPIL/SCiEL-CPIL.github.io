import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    # Build adjacency matrix
    adj = [0] * N
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    all_mask = (1 << N) - 1
    
    # Precompute valid parts
    # A mask is a clique if for all v in mask, (mask ^ (1<<v)) is subset of adj[v]
    # A mask is an independent set if for all v in mask, (mask ^ (1<<v)) is disjoint from adj[v]
    
    valid_parts = []
    for mask in range(1, 1 << N):
        is_clique = True
        is_indep = True
        # Optimization: check only if mask is not 0
        # Iterate bits
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            if rest & ~adj[v]:
                is_clique = False
            if rest & adj[v]:
                is_indep = False
            if not is_clique and not is_indep:
                break
            m ^= lsb
            
        if is_clique or is_indep:
            valid_parts.append(mask)
            
    # Sort by size descending to find large covers first
    valid_parts.sort(key=lambda x: -x.bit_count())
    
    # Helper to check if mask is a valid part
    # We can use a set for O(1) lookup
    valid_set = set(valid_parts)
    
    # Lower bound function
    def lower_bound(rem_mask):
        if rem_mask == 0:
            return 0
        if rem_mask in valid_set:
            return 1
        # Check if coverable by 2
        # Iterate valid parts S subset of rem_mask
        # If rem_mask ^ S is in valid_set, return 2
        # To speed up, we can just return 2 as a heuristic if not 1?
        # Returning 2 is safe but weak.
        # Let's try to check for 2 quickly.
        # We only need to check S that are "large" or specific?
        # Given N=14, checking all subsets S of rem_mask is 2^k where k is popcount(rem_mask).
        # Max k=14, 2^14=16k checks.
        # If B&B visits 1000 nodes, 16M checks. Feasible.
        # But we can prune: only check S in valid_parts that are subsets of rem_mask.
        # This is still many.
        # Let's just return 2 for now to keep it fast, or 1 if valid.
        # Actually, if it's not 1, it's at least 2.
        return 2

    best_k = N # Max parts is N (singletons)
    best_solution = [[i] for i in range(N)]
    
    # Greedy initial solution
    def greedy():
        rem = all_mask
        parts = []
        while rem:
            # Find largest valid part subset of rem
            best_part = 0
            best_size = 0
            for p in valid_parts:
                if p & ~rem == 0:
                    sz = p.bit_count()
                    if sz > best_size:
                        best_size = sz
                        best_part = p
            if best_part == 0:
                # Should not happen, singletons are valid
                best_part = rem & -rem
            parts.append(best_part)
            rem ^= best_part
        return parts

    g_parts = greedy()
    best_k = len(g_parts)
    best_solution = g_parts

    node_count = 0
    
    def search(rem_mask, chosen, count):
        nonlocal best_k, best_solution, node_count
        
        if time.time() - start_time > DEADLINE:
            return False
            
        node_count += 1
        
        if rem_mask == 0:
            if count < best_k:
                best_k = count
                best_solution = list(chosen)
            return True
            
        # Prune
        lb = lower_bound(rem_mask)
        if count + lb >= best_k:
            return False
            
        # Heuristic: Try parts that cover the vertex with highest degree in rem_mask?
        # Or just iterate valid parts.
        # To avoid duplicate permutations, we can enforce that the next part chosen
        # must contain the smallest vertex in rem_mask?
        # No, that's for set cover where order doesn't matter.
        # Here, the parts are disjoint.
        # We can enforce: Let v_min be the smallest vertex in rem_mask.
        # The next part MUST contain v_min.
        # This reduces branching significantly.
        
        v_min_bit = rem_mask & -rem_mask
        v_min = v_min_bit.bit_length() - 1
        
        # Find all valid parts that are subsets of rem_mask and contain v_min
        candidates = []
        for p in valid_parts:
            if (p & ~rem_mask) == 0 and (p & v_min_bit):
                candidates.append(p)
                
        # Sort candidates by size descending
        candidates.sort(key=lambda x: -x.bit_count())
        
        for p in candidates:
            chosen.append(p)
            if search(rem_mask ^ p, chosen, count + 1):
                chosen.pop()
                return True
            chosen.pop()
            
        return False

    # Start search
    # We know best_k from greedy.
    # We can try to find a solution with best_k - 1, then best_k - 2, etc.
    # Or just let B&B find the min.
    
    # To be safe and fast, we can run B&B looking for any solution < best_k.
    # If found, update best_k and restart?
    # Or just let the recursion handle it.
    
    # The current search function returns True if it finds a solution with count < best_k?
    # No, it updates best_k.
    # We need to ensure it explores properly.
    
    # Let's run the search.
    # The "v_min" constraint ensures we don't permute the order of disjoint sets.
    # This is a standard optimization for set partitioning.
    
    search(all_mask, [], 0)
    
    # Convert best_solution (list of masks) to list of lists of ints
    final_result = []
    for mask in best_solution:
        lst = []
        for i in range(N):
            if (mask >> i) & 1:
                lst.append(i)
        final_result.append(lst)
        
    # Ensure types are Python int
    for i in range(len(final_result)):
        final_result[i] = [int(x) for x in final_result[i]]
        
    return final_result

if __name__ == "__main__":
    construct()
