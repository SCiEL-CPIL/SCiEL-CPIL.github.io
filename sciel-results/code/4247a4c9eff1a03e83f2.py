import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115  # Safety margin
    
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
    # i and j conflict if they share exactly 3 elements
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            inter = (masks[i] & masks[j])
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute degrees for weighted greedy
    degrees = [len(adj[i]) for i in range(n)]
    
    def greedy_mis_weighted(seed_val):
        """Weighted greedy: pick node with fewest conflicts among remaining."""
        rng = random.Random(seed_val)
        available = set(range(n))
        is_set = set()
        while available:
            # Pick the node with minimum degree in the remaining graph
            # Add small random tie-breaking
            best_node = None
            best_score = float('inf')
            for v in available:
                conflicts = len(adj[v] & is_set)
                # We want nodes with no conflicts; among those, prefer low degree
                if conflicts == 0:
                    score = degrees[v] + rng.random() * 0.01
                else:
                    score = float('inf')
                if score < best_score:
                    best_score = score
                    best_node = v
            
            if best_node is None or best_score == float('inf'):
                break
            
            is_set.add(best_node)
            available.discard(best_node)
            # Remove neighbors of best_node from available
            for nb in adj[best_node]:
                available.discard(nb)
        
        return is_set

    def greedy_mis_random(seed_val):
        """Random order greedy."""
        rng = random.Random(seed_val)
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
        while True:
            improved = False
            candidates = [v for v in range(n) if v not in is_set]
            rng.shuffle(candidates)
            for v in candidates:
                if not (adj[v] & is_set):
                    is_set.add(v)
                    improved = True
            if not improved:
                break
        return is_set

    def local_search_swap(is_set, rng, max_iter=50):
        """Try to swap: remove one node to allow adding two or more."""
        is_set = set(is_set)
        for _ in range(max_iter):
            improved = False
            # Try removing each node and see if we can add more
            for v in list(is_set):
                if time.time() > deadline:
                    return is_set
                trial = is_set - {v}
                # Try to add as many as possible
                added = 0
                candidates = [u for u in range(n) if u not in trial]
                rng.shuffle(candidates)
                for u in candidates:
                    if not (adj[u] & trial):
                        trial.add(u)
                        added += 1
                if added >= 2:  # Net gain of at least 1
                    is_set = trial
                    improved = True
                    break
            if not improved:
                break
        return is_set

    best_is = set()
    best_size = 0
    
    current_seed = 0
    while time.time() < deadline:
        # Alternate between weighted and random greedy
        if current_seed % 3 == 0:
            is_set = greedy_mis_weighted(current_seed)
        else:
            is_set = greedy_mis_random(current_seed)
        
        # Apply local search
        rng = random.Random(current_seed)
        is_set = local_search_add(is_set, rng)
        
        # Try swap local search (more expensive)
        if time.time() < deadline - 10:
            is_set = local_search_swap(is_set, random.Random(current_seed), max_iter=20)
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
            
        current_seed += 1
    
    # Build vectors
    vectors = []
    for idx in best_is:
        s = subsets[idx]
        # For each selected subset, add all 16 sign variations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
