"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = time.time() + 115.0  # safety margin
    n = 20
    trees = instance_trees(n, 611)
    
    # Track used edges
    used_edges = set()
    result = [None] * n
    
    # Greedy: try to embed trees from largest to smallest
    # For each tree, try random permutations of host vertices
    order = list(range(n - 1, -1, -1))  # T20 down to T1
    
    for idx in order:
        if time.time() > deadline:
            break
        k = idx + 1
        tree_edges = trees[idx]
        
        if k == 1:
            # Single vertex, any unused vertex works
            for v in range(n):
                if v not in used_edges:
                    result[idx] = [v]
                    break
            continue
        
        if k == 2:
            # Single edge
            found = False
            for _ in range(10000):
                u, v = random.sample(range(n), 2)
                if (u, v) not in used_edges and (v, u) not in used_edges:
                    result[idx] = [u, v]
                    used_edges.add((u, v))
                    used_edges.add((v, u))
                    found = True
                    break
            if not found:
                # Try systematically
                for u in range(n):
                    for v in range(u + 1, n):
                        if (u, v) not in used_edges and (v, u) not in used_edges:
                            result[idx] = [u, v]
                            used_edges.add((u, v))
                            used_edges.add((v, u))
                            found = True
                            break
                    if found:
                        break
            continue
        
        # For larger trees, use backtracking with random restarts
        max_restarts = 500
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initial assignment
            perm = list(range(n))
            random.shuffle(perm)
            
            # Try to embed tree using the permutation
            success = _try_embed_tree(tree_edges, perm, used_edges, n)
            if success:
                result[idx] = perm[:k]
                # Add edges to used set
                for (a, b) in tree_edges:
                    used_edges.add((perm[a], perm[b]))
                    used_edges.add((perm[b], perm[a]))
                break
    
    # Convert to proper format
    final_result = []
    for i in range(n):
        if result[i] is None:
            final_result.append(None)
        else:
            final_result.append(list(result[i]))
    
    return final_result

def _try_embed_tree(tree_edges, perm, used_edges, n):
    """Try to embed tree_edges using vertex mapping perm[0..k-1] -> host vertices.
    Returns True if embedding is valid and edge-disjoint."""
    k = len(perm)
    
    # Check if all mapped vertices are valid
    if len(set(perm[:k])) != k:
        return False
    
    # Check edge disjointness
    for (a, b) in tree_edges:
        u, v = perm[a], perm[b]
        if (u, v) in used_edges or (v, u) in used_edges:
            return False
    
    return True
# EVOLVE-BLOCK-END
