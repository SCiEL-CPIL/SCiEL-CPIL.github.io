import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate all 4-subsets of {0, ..., 10}
    subsets = []
    for comb in itertools.combinations(range(D), 4):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        subsets.append(mask)
    
    n_subsets = len(subsets)
    
    # Build incompatibility matrix: conflict[i,j] = True if subsets[i] & subsets[j] has > 2 bits
    # We want Max Independent Set in this conflict graph = Max Clique in compatibility graph
    # Let's work with the compatibility graph: compat[i,j] = True if intersection <= 2
    compat = np.zeros((n_subsets, n_subsets), dtype=bool)
    for i in range(n_subsets):
        for j in range(i+1, n_subsets):
            inter = subsets[i] & subsets[j]
            if inter.bit_count() <= 2:
                compat[i, j] = True
                compat[j, i] = True
    
    # Now solve Max Clique in compat graph using branch-and-bound with coloring
    # Greedy coloring gives upper bound on clique size
    
    def greedy_coloring(nodes, compat):
        """Greedy coloring of the subgraph induced by nodes. Returns color count (upper bound on clique size)."""
        if not nodes:
            return 0, []
        # Order nodes by degree in the subgraph (descending) for better bounds
        degrees = [(compat[np.ix_(nodes, nodes)].sum(axis=1)[k], nodes[k]) for k in range(len(nodes))]
        degrees.sort(reverse=True)
        order = [d[1] for d in degrees]
        
        colors = {}
        color_count = 0
        for node in order:
            # Find smallest available color
            used_colors = set()
            for other in order:
                if other != node and other in colors and compat[node, other]:
                    used_colors.add(colors[other])
            c = 0
            while c in used_colors:
                c += 1
            colors[node] = c
            color_count = max(color_count, c + 1)
        return color_count, colors
    
    # Branch and bound for Max Clique
    best_clique = []
    best_size = 0
    
    def branch_bound(candidates, current_clique):
        nonlocal best_clique, best_size, deadline, start_time
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return
        
        # Upper bound: current size + greedy coloring bound on candidates
        bound, _ = greedy_coloring(candidates, compat)
        if len(current_clique) + bound <= best_size:
            return
        
        # Choose pivot node: one with max degree in candidates subgraph
        degs = compat[np.ix_(candidates, candidates)].sum(axis=1)
        pivot_idx = int(np.argmax(degs))
        pivot = candidates[pivot_idx]
        
        # Branch: include pivot or exclude pivot
        # Include pivot
        new_candidates = [c for c in candidates if c != pivot and compat[pivot, c]]
        current_clique.append(pivot)
        branch_bound(new_candidates, current_clique)
        current_clique.pop()
        
        # Exclude pivot
        candidates.remove(pivot)
        branch_bound(candidates, current_clique)
    
    # Start branch and bound
    all_nodes = list(range(n_subsets))
    branch_bound(all_nodes, [])
    
    selected = best_clique
    
    # Construct vectors
    vectors = []
    for idx in selected:
        mask = subsets[idx]
        indices = []
        for i in range(D):
            if mask & (1 << i):
                indices.append(i)
        
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx2, s in zip(indices, signs):
                v[idx2] = s
            vectors.append(v)
            
    if not vectors:
        # Fallback
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
