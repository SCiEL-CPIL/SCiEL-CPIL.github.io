import numpy as np
import itertools
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # 1. Generate supports
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports) # 330
    
    # 2. Build Conflict Graph (Adjacency Matrix)
    # Conflict if intersection size is 3
    masks = [0] * N_S
    for i, s in enumerate(supports):
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks[i] = m

    adj = [0] * N_S # Bitmasks for neighbors
    for i in range(N_S):
        for j in range(i + 1, N_S):
            if (masks[i] & masks[j]).bit_count() == 3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # 3. Solve MIS using Branch and Bound with Greedy Coloring
    # We want max independent set.
    
    best_size = 0
    best_set = []
    
    # Precompute degree for ordering
    degrees = [adj[i].bit_count() for i in range(N_S)]
    
    # Greedy coloring for upper bound on MIS size
    # Color the subgraph induced by candidates
    # The number of colors is an upper bound on the size of a clique in the complement,
    # which is an upper bound on the size of an independent set in the original graph.
    
    def greedy_color(candidates):
        # candidates is a list of node indices
        # Returns a list of colors (0-based) for each node in candidates
        # such that adjacent nodes have different colors
        # The max color + 1 is an upper bound on the size of a clique in the complement
        # which is an upper bound on the size of an independent set in the original graph
        
        n = len(candidates)
        if n == 0:
            return []
        
        # Order by degree in the candidate subgraph (descending)
        # For coloring, we can use a simple heuristic
        # Let's just use the order they are given, but sort by degree for better bounds
        
        # Create a mapping from node index to position in candidates list
        # Actually, we can just work with the list directly
        
        # Sort candidates by degree in the candidate subgraph (descending)
        # This often leads to tighter bounds
        cand_set = set(candidates)
        deg_in_cand = [(v, (adj[v] & (sum(1 << u for u in cand_set))).bit_count()) for v in candidates]
        deg_in_cand.sort(key=lambda x: -x[1])
        ordered_cands = [v for v, _ in deg_in_cand]
        
        colors = [-1] * n
        # Map node -> color
        node_color = {}
        
        for i, v in enumerate(ordered_cands):
            # Find the smallest color not used by neighbors
            used_colors = set()
            for u in candidates:
                if u != v and (adj[v] >> u) & 1:
                    if u in node_color:
                        used_colors.add(node_color[u])
            
            c = 0
            while c in used_colors:
                c += 1
            
            node_color[v] = c
            # We need to map back to the original order in candidates
            # Find index of v in candidates
            idx = candidates.index(v)
            colors[idx] = c
        
        return colors

    # Branch and Bound
    # State: current_size, current_set, candidates (nodes that can be added)
    
    def branch_and_bound(current_size, current_set, candidates):
        nonlocal best_size, best_set
        
        if time.time() > deadline - 5:
            return
        
        # Pruning: if current_size + len(candidates) <= best_size, prune
        if current_size + len(candidates) <= best_size:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Upper bound using greedy coloring
        colors = greedy_color(candidates)
        max_color = max(colors) if colors else 0
        upper_bound = current_size + max_color + 1
        
        if upper_bound <= best_size:
            return
        
        # Choose a pivot node
        # Heuristic: choose node with highest degree in candidate subgraph
        cand_set = set(candidates)
        cand_mask = 0
        for v in candidates:
            cand_mask |= (1 << v)
        
        # Find node with max degree in candidate subgraph
        pivot = candidates[0]
        max_deg = 0
        for v in candidates:
            deg = (adj[v] & cand_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Branch 1: Include pivot
        new_set = list(current_set)
        new_set.append(pivot)
        
        # New candidates: candidates that are not adjacent to pivot
        new_candidates = [v for v in candidates if v != pivot and not (adj[pivot] >> v) & 1]
        
        branch_and_bound(current_size + 1, new_set, new_candidates)
        
        # Branch 2: Exclude pivot
        # New candidates: candidates without pivot
        new_candidates = [v for v in candidates if v != pivot]
        
        branch_and_bound(current_size, current_set, new_candidates)

    # Initial candidates: all nodes
    initial_candidates = list(range(N_S))
    
    # To improve performance, we can order the initial candidates
    # Let's sort by degree (descending) for better pruning
    initial_candidates.sort(key=lambda v: -degrees[v])
    
    branch_and_bound(0, [], initial_candidates)
    
    # If B&B didn't find a good solution or timed out, fall back to heuristic
    if best_size == 0 or time.time() > deadline - 20:
        # Simple greedy: pick nodes with lowest degree
        remaining = list(range(N_S))
        s = []
        while remaining:
            # Pick node with lowest degree in remaining
            min_deg = float('inf')
            min_idx = -1
            rem_mask = 0
            for u in remaining:
                rem_mask |= (1 << u)
            
            for i, v in enumerate(remaining):
                deg = (adj[v] & rem_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_idx = i
            
            v = remaining[min_idx]
            s.append(v)
            
            # Remove v and its neighbors
            to_remove = adj[v] | (1 << v)
            new_remaining = []
            for u in remaining:
                if not (to_remove >> u) & 1:
                    new_remaining.append(u)
            remaining = new_remaining
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # 4. Construct Vectors
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)

    return np.array(vectors, dtype=np.int8)
