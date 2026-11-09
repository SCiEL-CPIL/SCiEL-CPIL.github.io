import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for evaluation overhead

    def check_validity_fast(n, adj_masks):
        """
        Fast validity check using bitmasks.
        adj_masks[i] is an integer bitmask of neighbors of vertex i.
        Returns True if no regular induced subgraph of size >= 6 exists.
        """
        vertices = list(range(n))
        for k in range(6, n + 1):
            for comb in itertools.combinations(vertices, k):
                # Check if induced subgraph on comb is regular
                degrees = []
                for v in comb:
                    # Count neighbors of v that are also in comb
                    mask = 0
                    for u in comb:
                        if u != v and (adj_masks[v] >> u) & 1:
                            mask |= (1 << u)
                    degrees.append(bin(mask).count('1'))
                if len(set(degrees)) == 1:
                    return False
        return True

    def count_bad_6_fast(n, adj_masks, comb_list):
        """Count 6-vertex subsets inducing regular subgraphs using bitmasks."""
        count = 0
        for comb in comb_list:
            degrees = []
            for v in comb:
                deg = 0
                for u in comb:
                    if u != v and (adj_masks[v] >> u) & 1:
                        deg += 1
                degrees.append(deg)
            if len(set(degrees)) == 1:
                count += 1
        return count

    def greedy_sequential(target_n, max_restarts=5, time_budget=40):
        """
        Greedy sequential vertex addition.
        Start with empty graph, add vertices one by one.
        For each new vertex, try all 2^k connection patterns to existing k vertices.
        Pick the pattern that minimizes new regular 6+ subgraphs created.
        """
        t_start = time.time()
        
        for restart in range(max_restarts):
            if time.time() > t_start + time_budget:
                break
            
            # Randomize vertex ordering for diversity
            order = list(range(target_n))
            random.shuffle(order)
            
            # adj_masks[i] = bitmask of neighbors of vertex i
            adj_masks = [0] * target_n
            # Track which vertices have been "added"
            added = [False] * target_n
            added[order[0]] = True
            current_n = 1
            
            success = True
            for step in range(1, target_n):
                if time.time() > deadline:
                    return None
                if time.time() > t_start + time_budget:
                    success = False
                    break
                
                new_v = order[step]
                existing = [v for v in range(target_n) if added[v] and v != new_v]
                k = len(existing)
                
                if k == 0:
                    # First vertex, no edges possible
                    continue
                
                # Try all 2^k connection patterns
                # For large k, this is expensive. Use sampling for k > 12
                if k <= 12:
                    patterns = list(range(1 << k))
                    random.shuffle(patterns)
                else:
                    # Sample patterns
                    patterns = []
                    # Always include all-zeros and all-ones
                    patterns.append(0)
                    patterns.append((1 << k) - 1)
                    # Random patterns
                    for _ in range(200):
                        patterns.append(random.getrandbits(k))
                    random.shuffle(patterns)
                
                best_pattern = None
                best_score = float('inf')
                best_adj = None
                
                for pat in patterns:
                    # Check time
                    if time.time() > deadline:
                        return None
                    
                    # Create candidate: new_v connects to existing[i] if bit i is set
                    candidate_masks = adj_masks[:]
                    for i, ev in enumerate(existing):
                        if (pat >> i) & 1:
                            candidate_masks[new_v] |= (1 << ev)
                            candidate_masks[ev] |= (1 << new_v)
                    
                    # Count new bad 6-sets involving new_v
                    # A 6-set with new_v: choose 5 from existing
                    new_bad = 0
                    if k >= 5:
                        for five in itertools.combinations(existing, 5):
                            comb = five + (new_v,)
                            # Check if this 6-set is regular
                            degrees = []
                            for v in comb:
                                deg = 0
                                for u in comb:
                                    if u != v and (candidate_masks[v] >> u) & 1:
                                        deg += 1
                                degrees.append(deg)
                            if len(set(degrees)) == 1:
                                new_bad += 1
                    
                    # Also check if adding new_v creates bad 7+ sets
                    # For efficiency, only check 6-sets here; full check at end
                    if new_bad < best_score:
                        best_score = new_bad
                        best_pattern = pat
                        best_adj = candidate_masks[:]
                    
                    if best_score == 0:
                        break  # Can't do better than 0
                
                if best_pattern is None:
                    success = False
                    break
                
                # Apply best pattern
                adj_masks = best_adj
                added[new_v] = True
                current_n += 1
            
            if not success:
                continue
            
            # Full validity check
            if check_validity_fast(target_n, adj_masks):
                # Convert to edge list
                edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if (adj_masks[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': target_n, 'edges': edges}
        
        return None

    def local_search_optimized(n, max_iters=500, time_budget=30):
        """
        Optimized local search using bitmask representation.
        """
        t_start = time.time()
        
        # Precompute all 6-combinations
        comb_list = list(itertools.combinations(range(n), 6))
        
        for restart in range(5):
            if time.time() > t_start + time_budget:
                break
            
            # Initialize random graph as bitmask
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            current_score = count_bad_6_fast(n, adj_masks, comb_list)
            if current_score == 0:
                if check_validity_fast(n, adj_masks):
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj_masks[u] >> v) & 1:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
            
            best_adj = adj_masks[:]
            best_score = current_score
            
            edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
            
            for it in range(max_iters):
                if time.time() > deadline or time.time() > t_start + time_budget:
                    break
                
                u, v = random.choice(edges_list)
                
                # Flip edge (u, v)
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                
                new_score = count_bad_6_fast(n, adj_masks, comb_list)
                
                if new_score <= current_score:
                    current_score = new_score
                    if current_score == 0:
                        if check_validity_fast(n, adj_masks):
                            edges = []
                            for uu in range(n):
                                for vv in range(uu + 1, n):
                                    if (adj_masks[uu] >> vv) & 1:
                                        edges.append([uu, vv])
                            return {'n': n, 'edges': edges}
                else:
                    # Revert
                    adj_masks[u] ^= (1 << v)
                    adj_masks[v] ^= (1 << u)
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj_masks[:]
        
        return None

    # Strategy: Try greedy sequential for n=21, then n=20
    # Then fall back to local search

    # Try n=21 with greedy
    result = greedy_sequential(21, max_restarts=3, time_budget=30)
    if result is not None:
        return result

    # Try n=20 with greedy
    result = greedy_sequential(20, max_restarts=5, time_budget=30)
    if result is not None:
        return result

    # Try n=21 with local search
    result = local_search_optimized(21, max_iters=300, time_budget=20)
    if result is not None:
        return result

    # Try n=20 with local search
    result = local_search_optimized(20, max_iters=500, time_budget=30)
    if result is not None:
        return result

    # Fallback: n=15 with local search
    result = local_search_optimized(15, max_iters=300, time_budget=10)
    if result is not None:
        return result

    # Ultimate fallback: known valid small graph
    n = 6
    edges = []
    for u in range(6):
        for v in range(u + 1, 6):
            if not (u == 0 and v == 1):
                edges.append([u, v])
    return {'n': n, 'edges': edges}

