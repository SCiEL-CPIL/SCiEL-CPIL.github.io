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
        
    # Precompute adjacency as bitsets for fast operations
    # adj_mask[i] is a Python int with bit j set if i and j conflict
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            inter = (masks[i] & masks[j])
            if inter.bit_count() == 3:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Also keep set version for local search
    adj = [set() for _ in range(n)]
    for i in range(n):
        m = adj_mask[i]
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            adj[i].add(j)
            m ^= lsb
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Greedy coloring for upper bound in branch and bound
    def greedy_color(candidates):
        """Greedy coloring of the subgraph induced by candidates.
        Returns a list of color classes (each a set of nodes)."""
        uncolored = set(candidates)
        color_classes = []
        while uncolored:
            # Find a maximal independent set in uncolored using greedy
            color_class = set()
            # Order by degree within uncolored
            ordered = sorted(uncolored, key=lambda v: len(adj[v] & uncolored))
            for v in ordered:
                if v not in uncolored:
                    continue
                if not (adj_mask[v] & (1 << 0) - 1):  # placeholder, will use different check
                    pass
                # Check if v conflicts with any in current color_class
                conflict = False
                for u in color_class:
                    if (adj_mask[v] >> u) & 1:
                        conflict = True
                        break
                if not conflict:
                    color_class.add(v)
                    uncolored.discard(v)
            color_classes.append(color_class)
        return color_classes
    
    # Better greedy coloring using bit operations
    def greedy_color_v2(candidates):
        """Greedy coloring using bitset operations."""
        uncolored = set(candidates)
        color_classes = []
        while uncolored:
            # Greedy maximal independent set
            color_class = set()
            # Sort by degree in remaining uncolored
            ordered = sorted(uncolored, key=lambda v: bin(adj_mask[v] & ((1 << n) - 1) & 0).count('1'))
            # Actually let's just use a simple approach
            remaining = set(uncolored)
            while remaining:
                # Pick vertex with minimum degree in remaining
                v = min(remaining, key=lambda x: len(adj[x] & remaining))
                color_class.add(v)
                remaining.discard(v)
                # Remove neighbors of v from remaining (they can't be in same color)
                for nb in adj[v]:
                    remaining.discard(nb)
            color_classes.append(color_class)
            uncolored = uncolored - color_class
        return color_classes
    
    # Branch and bound for maximum independent set
    def branch_and_bound(initial_candidates, time_limit):
        """Find maximum independent set using branch and bound with greedy coloring."""
        best_set = set()
        best_size = 0
        
        # Convert to bitset for efficiency
        def to_bitset(s):
            b = 0
            for v in s:
                b |= (1 << v)
            return b
        
        def from_bitset(b):
            s = set()
            while b:
                lsb = b & (-b)
                s.add(lsb.bit_length() - 1)
                b ^= lsb
            return s
        
        def bb(candidates_bitset, current_set_size):
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return
            
            if candidates_bitset == 0:
                if current_set_size > best_size:
                    best_size = current_set_size
                return
            
            # Upper bound: current size + number of colors needed for candidates
            candidates = from_bitset(candidates_bitset)
            color_classes = greedy_color_v2(candidates)
            upper_bound = current_set_size + len(color_classes)
            
            if upper_bound <= best_size:
                return
            
            # Branch: pick a vertex to branch on
            # Use the vertex from the largest color class (or first)
            # Actually, standard approach: pick vertex v, branch on including v or excluding v
            
            # Pick vertex with maximum degree in candidates for better pruning
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            
            # Branch 1: Include v
            new_candidates = candidates_bitset & ~(1 << v) & ~adj_mask[v]
            bb(new_candidates, current_set_size + 1)
            
            # Branch 2: Exclude v
            new_candidates = candidates_bitset & ~(1 << v)
            bb(new_candidates, current_set_size)
        
        return best_set
    
    # Improved branch and bound with better vertex ordering
    def bb_improved(candidates, current_set, time_limit):
        """Branch and bound with dynamic vertex selection."""
        nonlocal best_set_global, best_size_global
        
        if time.time() > time_limit:
            return
        
        if not candidates:
            if len(current_set) > best_size_global:
                best_size_global = len(current_set)
                best_set_global = set(current_set)
            return
        
        # Upper bound using greedy coloring
        color_classes = greedy_color_v2(candidates)
        if len(current_set) + len(color_classes) <= best_size_global:
            return
        
        # Select vertex to branch on
        # Choose vertex with highest degree in candidates
        v = max(candidates, key=lambda x: len(adj[x] & candidates))
        
        # Branch 1: Include v
        new_candidates = {u for u in candidates if u != v and u not in adj[v]}
        bb_improved(new_candidates, current_set | {v}, time_limit)
        
        # Branch 2: Exclude v
        new_candidates = candidates - {v}
        bb_improved(new_candidates, current_set, time_limit)
    
    best_set_global = set()
    best_size_global = 0
    
    # Run branch and bound on the full graph
    all_nodes = set(range(n))
    bb_improved(all_nodes, set(), deadline - 5)
    
    best_is = best_set_global
    best_size = best_size_global
    
    # If B&B didn't find anything, fall back to greedy
    if best_size == 0:
        def greedy_mis_weighted(seed_val):
            """Weighted greedy: pick node with fewest conflicts among remaining."""
            rng = random.Random(seed_val)
            available = set(range(n))
            is_set = set()
            while available:
                best_node = None
                best_score = float('inf')
                for v in available:
                    conflicts = len(adj[v] & is_set)
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
                for nb in adj[best_node]:
                    available.discard(nb)
            
            return is_set
        
        best_is = greedy_mis_weighted(0)
        best_size = len(best_is)
    
    # Try to improve with local search if time remains
    if time.time() < deadline - 10:
        def local_search_add(is_set, rng):
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
        
        best_is = local_search_add(best_is, random.Random(42))
        best_size = len(best_is)
    
    # Build vectors
    vectors = []
    for idx in best_is:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
