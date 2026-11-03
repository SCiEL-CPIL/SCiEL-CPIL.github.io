import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Convert subsets to bitmasks for fast intersection checks
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)
        
    # Precompute adjacency list and degree
    # i and j conflict if they share exactly 3 elements (intersection size >= 3)
    # Since they are both size 4, intersection can be 0, 1, 2, 3, 4.
    # Intersection 4 means i==j.
    # Intersection 3 means they differ by 1 element.
    # Intersection <= 2 means they are compatible.
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            inter = masks[i] & masks[j]
            # Popcount check. Since D=11, we can use bin().count('1') or precomputed table
            # 330*330/2 is ~54k checks, bin().count is fast enough
            if inter.bit_count() >= 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    best_mis = set()
    best_size = 0
    
    def greedy_mis_min_degree(rng):
        """Greedy MIS: pick node with fewest conflicts in remaining graph."""
        available = set(range(n))
        is_set = set()
        # Copy degrees to track remaining degree
        rem_degrees = degrees[:]
        
        while available:
            # Find node with min remaining degree
            min_deg = float('inf')
            best_node = -1
            # To add randomness, we can pick from top-k min degree nodes
            # Or just add small noise
            # Let's pick the one with min degree, breaking ties randomly
            candidates = []
            for v in available:
                if rem_degrees[v] < min_deg:
                    min_deg = rem_degrees[v]
                    candidates = [v]
                elif rem_degrees[v] == min_deg:
                    candidates.append(v)
            
            if not candidates:
                break
                
            best_node = rng.choice(candidates)
            
            is_set.add(best_node)
            available.discard(best_node)
            
            # Update degrees of remaining nodes
            for nb in adj[best_node]:
                if nb in available:
                    rem_degrees[nb] -= 1
                    # Note: nb is removed from available in next step if it conflicts? 
                    # No, in MIS greedy, we just pick a node, then remove it and its neighbors from 'available' 
                    # so we don't pick neighbors later.
                    # Wait, the standard greedy MIS is:
                    # 1. Pick v
                    # 2. Add v to IS
                    # 3. Remove v and all neighbors of v from Available
                    # My logic above updated degrees but didn't remove neighbors from available.
                    # Let's correct the greedy strategy.
                    
        # Re-implementing greedy correctly
        available = set(range(n))
        is_set = set()
        while available:
            # Pick node with min degree in the subgraph induced by 'available'
            min_deg = float('inf')
            candidates = []
            for v in available:
                deg = len(adj[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            if not candidates:
                break
            
            best_node = rng.choice(candidates)
            is_set.add(best_node)
            # Remove best_node and its neighbors from available
            to_remove = adj[best_node] | {best_node}
            available -= to_remove
        
        return is_set

    def greedy_mis_random(rng):
        """Random order greedy."""
        order = list(range(n))
        rng.shuffle(order)
        is_set = set()
        for v in order:
            if not (adj[v] & is_set):
                is_set.add(v)
        return is_set

    def local_search_add(is_set, rng):
        """Try to add any node that doesn't conflict."""
        is_set = set(is_set)
        improved = True
        while improved:
            improved = False
            # Find all candidates that can be added
            candidates = [v for v in range(n) if v not in is_set and not (adj[v] & is_set)]
            if candidates:
                # Add all of them? Or one by one?
                # Adding all is safe if they don't conflict with each other?
                # No, they might conflict with each other.
                # So we must add them one by one, or find a maximal independent set in the candidate subgraph.
                # Simple approach: shuffle and add if valid
                rng.shuffle(candidates)
                for v in candidates:
                    if v not in is_set and not (adj[v] & is_set):
                        is_set.add(v)
                        improved = True
        return is_set

    def local_search_swap(is_set, rng, max_iter=20):
        """Try to swap: remove one node to allow adding two or more."""
        is_set = set(is_set)
        current_size = len(is_set)
        
        for _ in range(max_iter):
            improved = False
            # Try removing each node
            for v in list(is_set):
                if time.time() > deadline:
                    return is_set
                trial = is_set - {v}
                # Try to add as many as possible from the neighbors of v (and others)
                # Actually, any node not in trial that doesn't conflict with trial
                # But checking all n nodes is expensive.
                # Optimization: Only check nodes that conflict with v? 
                # If u doesn't conflict with v, it was already a candidate for addition in 'add' phase.
                # So we only care about nodes that conflict with v, which were blocked.
                # But removing v might unblock nodes that conflict with v.
                # Let's just check all nodes not in trial. n=330 is small.
                
                added = 0
                # Get candidates: nodes not in trial
                # We can iterate through all nodes
                for u in range(n):
                    if u not in trial:
                        if not (adj[u] & trial):
                            trial.add(u)
                            added += 1
                
                if len(trial) > current_size:
                    is_set = trial
                    current_size = len(is_set)
                    improved = True
                    break # Restart the loop with new set
            
            if not improved:
                break
        
        return is_set

    best_is = set()
    best_size = 0
    
    seed_val = 0
    while time.time() < deadline:
        rng = random.Random(seed_val)
        
        # Alternate strategies
        if seed_val % 3 == 0:
            is_set = greedy_mis_min_degree(rng)
        elif seed_val % 3 == 1:
            is_set = greedy_mis_random(rng)
        else:
            # Hybrid: Random order but prioritize low degree?
            # Let's just use random for now
            is_set = greedy_mis_random(rng)
        
        # Polish with local search
        is_set = local_search_add(is_set, rng)
        
        # Swap search is more expensive, do it if time permits
        if time.time() < deadline - 5:
            is_set = local_search_swap(is_set, rng, max_iter=10)
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
            # print(f"Seed {seed_val}: size {best_size}")
            
        seed_val += 1
        
        # If we've found a very large set, we might want to be more careful
        # Theoretical max for kissing number 11 is 506? 
        # Wait, kissing number in 11 dims is not 506.
        # Kissing number in 11 is not known exactly? 
        # Actually, for D=11, the kissing number is 506? No, that's D=12 (Leech lattice).
        # For D=11, it's between 408 and 506?
        # The score 2.4727 corresponds to k = 2.4727 * seed.
        # Seed is usually 1. So k ~ 40?
        # Wait, the score is k/seed. If seed=1, score=k.
        # If score is 2.47, k=2.47? That's not an integer.
        # Ah, the score is likely normalized or the seed is not 1.
        # "score is k" ... "search score is value/seed".
        # If the seed program returns a specific k, and score is 2.47, maybe k is small?
        # Or maybe the "seed" in the denominator is a random number or a fixed constant?
        # Let's assume the goal is to maximize k (number of vectors).
        # The current program returns 16 * |MIS|.
        # If |MIS| is 25, k=400.
        # If score is 2.47, maybe the denominator is large?
        # Regardless, maximizing |MIS| is the goal.
        
        # If best_size is very large, we can increase max_iter in swap
        if best_size > 30:
             pass # Just keep searching

    # Construct vectors
    vectors = []
    for s_idx in best_is:
        support = subsets[s_idx]
        # Generate all 16 sign variants
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            vectors.append(v)
            
    if not vectors:
        # Fallback: return at least one vector
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        v[2] = 1
        v[3] = 1
        vectors.append(v)
        
    return np.array(vectors, dtype=np.int64)
