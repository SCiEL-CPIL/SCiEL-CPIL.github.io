import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 192
    max_k = 48
    
    # Precompute pair conflicts using bitsets
    # conflict[a] is a Python int used as a bitmask: bit b is 1 iff a and b conflict
    conflict = [0] * (n + 1)
    
    # For each pair (a, b), check if they conflict
    # a and b conflict if:
    #   exists distinct c, d not in {a,b} with a | (b + c + d)
    #   OR exists distinct c, d not in {a,b} with b | (a + c + d)
    
    # Precompute for each a, the set of residues that are "bad"
    # For a to not divide (b + c + d), we need (b + c + d) % a != 0
    # i.e., (c + d) % a != (-b) % a
    
    # Let's precompute for each modulus m, the set of pairs (c,d) with c+d ≡ r (mod m)
    # This is expensive. Instead, let's just check each pair directly but efficiently.
    
    def check_pair(a, b):
        """Returns True if a and b conflict."""
        # Check if exists distinct c,d not in {a,b} with a | (b+c+d)
        # i.e., (c+d) % a == (-b) % a
        target = (-b) % a
        # We need c,d in [1..n], c!=d, c!=a, c!=b, d!=a, d!=b, (c+d)%a == target
        # Iterate over possible c values
        for c in range(1, n + 1):
            if c == a or c == b:
                continue
            # d must satisfy (c + d) % a == target => d % a == (target - c) % a
            d_rem = (target - c) % a
            # d can be d_rem, d_rem+a, d_rem+2a, ... up to n
            # But d >= 1, so if d_rem == 0, d starts at a; else d starts at d_rem
            d_start = d_rem if d_rem > 0 else a
            for d in range(d_start, n + 1, a):
                if d == c or d == a or d == b:
                    continue
                return True
        
        # Check if exists distinct c,d not in {a,b} with b | (a+c+d)
        target2 = (-a) % b
        for c in range(1, n + 1):
            if c == a or c == b:
                continue
            d_rem = (target2 - c) % b
            d_start = d_rem if d_rem > 0 else b
            for d in range(d_start, n + 1, b):
                if d == c or d == a or d == b:
                    continue
                return True
        
        return False
    
    # Build conflict bitmasks
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if check_pair(a, b):
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)
    
    # Now solve Maximum Independent Set on this graph using branch and bound
    # with bitset acceleration.
    
    # For each vertex v, compat[v] = bitmask of vertices compatible with v (non-neighbors excluding v)
    compat = [0] * (n + 1)
    all_mask = 0
    for i in range(1, n + 1):
        all_mask |= (1 << i)
    for v in range(1, n + 1):
        compat[v] = all_mask & ~(1 << v) & ~conflict[v]
    
    best_set = []
    best_size = 0
    
    # Greedy coloring bound for MIS
    def greedy_color_bound(candidates):
        """
        Given a bitmask of candidates, return an upper bound on MIS size
        using greedy coloring on the induced subgraph.
        """
        if candidates == 0:
            return 0
        # Greedy: repeatedly pick a vertex, remove it and its neighbors, increment color
        color_count = 0
        remaining = candidates
        while remaining:
            color_count += 1
            # Pick a vertex (use lowest set bit for determinism)
            v = 1
            while not (remaining & (1 << v)):
                v += 1
                if v > n:
                    break
            if v > n:
                break
            # Remove v and all its neighbors from remaining
            # Neighbors of v are those NOT in compat[v] and not v itself
            # Actually, in the conflict graph, neighbors of v are conflict[v]
            # So non-neighbors (including v) are ~conflict[v]
            # We remove v and all vertices that conflict with v
            remaining &= ~conflict[v] & ~(1 << v)
        return color_count
    
    def mis_bnb(available, current_size, current_set):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound
        ub = current_size + greedy_color_bound(available)
        if ub <= best_size:
            return
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with minimum degree in the available subgraph
        # For simplicity, pick the one that appears in fewest compat sets
        # Or just pick lowest set bit
        v = 1
        while not (available & (1 << v)):
            v += 1
            if v > n:
                break
        if v > n:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Branch: include v
        new_available = available & compat[v]
        new_set = current_set + [v]
        mis_bnb(new_available, current_size + 1, new_set)
        
        # Branch: exclude v
        mis_bnb(available & ~(1 << v), current_size, current_set)
    
    # Initial greedy solution to set a lower bound
    greedy_set = []
    remaining = all_mask
    while remaining:
        # Pick vertex with fewest conflicts among remaining
        best_v = -1
        best_conf = n + 1
        for v in range(1, n + 1):
            if remaining & (1 << v):
                conf_count = bin(conflict[v] & remaining).count('1')
                if conf_count < best_conf:
                    best_conf = conf_count
                    best_v = v
        if best_v == -1:
            break
        greedy_set.append(best_v)
        remaining &= compat[best_v] & ~(1 << best_v)
    
    best_set = greedy_set[:]
    best_size = len(greedy_set)
    
    # Now run branch and bound
    mis_bnb(all_mask, 0, [])
    
    return best_set
