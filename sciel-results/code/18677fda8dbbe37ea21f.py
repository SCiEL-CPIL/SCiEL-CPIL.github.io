import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    trees = instance_trees(20, 611)
    
    # Precompute edges for each tree
    tree_edges = []
    for t in trees:
        e = set()
        for u, v in t:
            e.add((min(u, v), max(u, v)))
        tree_edges.append(e)
        
    # Host graph K20 edges
    host_edges = set()
    for i in range(20):
        for j in range(i + 1, 20):
            host_edges.add((i, j))
            
    def get_edges_for_mapping(mapping, tree_idx):
        edges = set()
        for u, v in trees[tree_idx]:
            hu, hv = mapping[u], mapping[v]
            edges.add((min(hu, hv), max(hu, hv)))
        return edges

    def find_embedding(tree_idx, available_edges, rng):
        k = tree_idx + 1
        if k == 1:
            return [0] # Single vertex, any host vertex works, pick 0 for consistency
        if k == 2:
            # Need an edge in available_edges
            for e in available_edges:
                return [e[0], e[1]]
            return None
            
        # Heuristic: order tree vertices by degree desc
        degrees = [0] * k
        for u, v in trees[tree_idx]:
            degrees[u] += 1
            degrees[v] += 1
        order = sorted(range(k), key=lambda x: -degrees[x])
        
        # Backtracking search
        # assigned[i] is the host vertex for tree vertex order[i]
        assigned = [-1] * k
        # Map from tree vertex index to its position in 'order'
        pos_in_order = [0] * k
        for i, v in enumerate(order):
            pos_in_order[v] = i
            
        # Neighbors in tree
        neighbors = [[] for _ in range(k)]
        for u, v in trees[tree_idx]:
            neighbors[u].append(v)
            neighbors[v].append(u)
            
        def dfs(idx):
            if idx == k:
                return True
            v = order[idx]
            # Try all host vertices
            # Optimization: if v has neighbors already assigned, restrict candidates
            candidates = list(range(20))
            rng.shuffle(candidates)
            
            for h in candidates:
                if h in assigned:
                    continue
                # Check edges to already assigned neighbors
                valid = True
                for u in neighbors[v]:
                    if pos_in_order[u] < idx: # u is already assigned
                        hu = assigned[pos_in_order[u]]
                        edge = (min(h, hu), max(h, hu))
                        if edge not in available_edges:
                            valid = False
                            break
                if valid:
                    assigned[pos_in_order[v]] = h
                    if dfs(idx + 1):
                        return True
                    assigned[pos_in_order[v]] = -1
            return False
            
        if dfs(0):
            # Construct mapping in original vertex order
            mapping = [0] * k
            for i, v in enumerate(order):
                mapping[v] = assigned[i]
            return mapping
        return None

    best_result = [None] * 20
    best_value = 0
    
    # Try a few permutations for T20
    # Identity is a good start
    perms = [list(range(20))]
    # Add a few random permutations
    rng_master = Random(123)
    for _ in range(10):
        p = list(range(20))
        rng_master.shuffle(p)
        perms.append(p)
        
    for perm in perms:
        if time.time() > deadline:
            break
            
        mappings = [None] * 20
        mappings[19] = perm
        used_edges = get_edges_for_mapping(perm, 19)
        
        # Check if perm is valid (edges in K20)
        # Since perm is a permutation of 0..19, edges are in K20.
        
        current_value = len(tree_edges[19])
        
        # Try to pack T19 down to T1
        for k_idx in range(18, -1, -1):
            if time.time() > deadline:
                break
                
            # Try a few random seeds for this tree
            found = False
            for seed in range(5):
                rng = Random(seed * 1000 + k_idx)
                m = find_embedding(k_idx, host_edges - used_edges, rng)
                if m is not None:
                    mappings[k_idx] = m
                    used_edges |= get_edges_for_mapping(m, k_idx)
                    current_value += len(tree_edges[k_idx])
                    found = True
                    break
            
            if not found:
                mappings[k_idx] = None
                # Continue to smaller trees, they might still fit
            
        if current_value > best_value:
            best_value = current_value
            best_result = mappings
            
    return best_result
