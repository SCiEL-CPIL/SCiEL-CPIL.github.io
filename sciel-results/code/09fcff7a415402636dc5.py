import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    N = 512
    candidates = list(range(1, N))
    weights = {a: 1.0 / (N - a) for a in candidates}
    
    # Precompute prime factors and masks
    primes = []
    for i in range(2, N):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(i)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    factor_masks = [0] * N
    for a in range(1, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_idx[x])
        factor_masks[a] = mask
    
    # Build conflict graph
    adj = {a: set() for a in candidates}
    for i in range(len(candidates)):
        u = candidates[i]
        for j in range(i + 1, len(candidates)):
            v = candidates[j]
            if factor_masks[u] & factor_masks[v]:
                adj[u].add(v)
                adj[v].add(u)
    
    degrees = {a: len(adj[a]) for a in candidates}
    
    # Greedy initial solution
    def greedy_init():
        avail = set(candidates)
        score = 0.0
        selected = set()
        while avail:
            best_node = max(avail, key=lambda x: weights[x])
            score += weights[best_node]
            selected.add(best_node)
            avail.remove(best_node)
            for nb in list(adj[best_node]):
                if nb in avail:
                    avail.remove(nb)
        return score, selected
    
    best_score, best_set = greedy_init()
    
    # Branch and Bound with tight upper bound using greedy coloring of conflict graph
    # Key insight: A proper coloring of the conflict graph partitions nodes into
    # independent sets (color classes). Any independent set in the conflict graph
    # can contain at most one node from each color class.
    # So upper bound = current_score + sum over color classes of max weight in class.
    
    def greedy_color_bound(avail):
        """
        Compute upper bound via greedy coloring of conflict graph induced on avail.
        Returns sum of max weights per color class.
        """
        if not avail:
            return 0.0
        
        # Order nodes by descending weight for tighter bound
        sorted_avail = sorted(avail, key=lambda x: -weights[x])
        
        # Greedy coloring: assign smallest available color
        # color_max[w] = max weight assigned to color w
        color_max = {}
        # For each node, track which colors are forbidden (neighbors already colored)
        node_forbidden = {}
        
        for u in sorted_avail:
            # Find forbidden colors: colors used by neighbors of u that are already colored
            forbidden = set()
            for nb in adj[u]:
                if nb in avail and nb in node_forbidden:
                    # nb is already processed and has a color
                    pass
            # Actually, let me redo this more carefully
            pass
        
        # Simpler approach: sequential coloring
        # Process nodes in order, assign first available color
        node_color = {}
        for u in sorted_avail:
            # Find colors used by already-colored neighbors
            used_colors = set()
            for nb in adj[u]:
                if nb in node_color:
                    used_colors.add(node_color[nb])
            # Assign smallest non-used color
            c = 0
            while c in used_colors:
                c += 1
            node_color[u] = c
            if c not in color_max:
                color_max[c] = weights[u]
            else:
                if weights[u] > color_max[c]:
                    color_max[c] = weights[u]
        
        return sum(color_max.values())
    
    def dfs(avail, current_score, current_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        if not avail:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
            
        # Upper bound using greedy coloring
        bound = current_score + greedy_color_bound(avail)
        if bound <= best_score:
            return
            
        # Pick node with highest weight to branch on
        v = max(avail, key=lambda x: weights[x])
        
        # Option 1: Include v
        new_avail_1 = set(avail)
        new_avail_1.remove(v)
        for nb in adj[v]:
            if nb in new_avail_1:
                new_avail_1.remove(nb)
        
        dfs(new_avail_1, current_score + weights[v], current_set | {v})
        
        # Option 2: Exclude v
        new_avail_2 = set(avail)
        new_avail_2.remove(v)
        
        dfs(new_avail_2, current_score, current_set)
    
    initial_avail = set(candidates)
    
    try:
        dfs(initial_avail, 0.0, set())
    except RecursionError:
        pass

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Selected: {res}")
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                print(f"ERROR: {res[i]} and {res[j]} not coprime")
                valid = False
    if valid:
        print("Valid")
    print("Done")
