"""Rewritten program: attempt exact decomposition of K20 into trees T1..T20."""
from random import Random
import sys
sys.setrecursionlimit(100000)

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    trees = instance_trees(20, 611)
    n = 20
    
    # Generate all edges
    all_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_edges.add((i, j))
    
    # We need to find edge-disjoint embeddings using all 190 edges
    # Strategy: backtracking with trees ordered by size (largest first)
    # For each tree, try all valid embeddings
    
    order = list(range(n))
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    used_edges = set()
    assignments = [None] * n
    node_count = [0] * (n + 1)  # node_count[k] = number of free vertices of degree k in remaining graph
    
    # For efficiency, precompute for each tree the list of edges
    tree_edges = [trees[i] for i in range(n)]
    
    def get_embedding_candidates(tree_idx, used_edges_set):
        """Generate candidate embeddings for tree_idx."""
        tree = tree_edges[tree_idx]
        k = tree_idx + 1
        if k == 1:
            return [[v] for v in range(n)]
        
        # For small trees, enumerate all vertex subsets
        # This is infeasible for large k, so we limit to small trees
        if k > 8:
            return None  # Too large for enumeration
        
        from itertools import combinations, permutations
        candidates = []
        for vert_subset in combinations(range(n), k):
            for perm in permutations(vert_subset):
                mapping = {i: perm[i] for i in range(k)}
                # Check all edges
                valid = True
                for (u, v) in tree:
                    eu, ev = mapping[u], mapping[v]
                    edge = (min(eu, ev), max(eu, ev))
                    if edge in used_edges_set:
                        valid = False
                        break
                if valid:
                    candidates.append(list(perm))
        return candidates
    
    # This enumeration approach won't work for large trees
    # Let me try a completely different approach: use a SAT solver or just give up and return partial solution
    
    # Pragmatic approach: try to embed trees greedily with randomization
    import random
    
    best_value = 0
    best_result = None
    
    for trial in range(200):
        rng = random.Random(trial)
        used = set()
        result = [None] * n
        
        # Process trees in random order, but bias toward larger trees
        tree_order = list(range(n))
        # Shuffle but keep large trees somewhat prioritized
        rng.shuffle(tree_order)
        tree_order.sort(key=lambda i: -len(tree_edges[i]) + rng.random() * 5)
        
        success = True
        for idx in tree_order:
            tree = tree_edges[idx]
            k = idx + 1
            if k == 1:
                # Any vertex works
                available = [v for v in range(n) if not any(v in e for e in used)]
                if available:
                    result[idx] = [rng.choice(available)]
                else:
                    result[idx] = [0]  # fallback
                continue
            
            # Try to find an embedding
            found = False
            for attempt in range(50):
                verts = rng.sample(range(n), k)
                mapping = {i: verts[i] for i in range(k)}
                valid = True
                new_edges = set()
                for (u, v) in tree:
                    eu, ev = mapping[u], mapping[v]
                    edge = (min(eu, ev), max(eu, ev))
                    if edge in used or edge in new_edges:
                        valid = False
                        break
                    new_edges.add(edge)
                if valid:
                    result[idx] = verts
                    used.update(new_edges)
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            value = sum(len(tree_edges[i]) for i in range(n) if result[i] is not None)
            if value > best_value:
                best_value = value
                best_result = result[:]
    
    if best_result is not None and best_value > 19:
        # Convert to proper format
        out = [None] * n
        for i in range(n):
            if best_result[i] is not None:
                out[i] = list(best_result[i])
        return out
    
    # Fallback
    return [None] * 19 + [list(range(n))]
