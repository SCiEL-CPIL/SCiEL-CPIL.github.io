"""Packing trees T1..T20 into K20."""
import time
import sys
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
    deadline = start_time + 110  # safety margin before 120s limit
    
    trees = instance_trees(20, 611)
    
    # Total edges across all trees
    total_edges = sum(len(t) for t in trees)
    # K20 has 190 edges total
    
    # Strategy: embed trees one by one using backtracking with randomization
    # We process larger trees first (harder to place) then smaller ones
    
    # Order: T20, T19, T18, ..., T1
    order = list(range(20))  # index 0 = T1, index 19 = T20
    # We'll embed in reverse order: largest first
    embed_order = list(range(19, -1, -1))  # indices 19, 18, ..., 0
    
    # For each tree, we need to find an embedding (mapping from tree vertices to K20 vertices)
    # such that no edge is shared with previously embedded trees
    
    def solve_with_backtrack(trees, embed_order, time_limit):
        """Try to embed all trees, returning the list of embeddings."""
        # used_edges: set of (min(u,v), max(u,v)) for edges already used
        used_edges = set()
        # embeddings[i] = list of 20 ints (the image of each vertex in T_{i+1})
        embeddings = [None] * 20
        
        def embed_tree(tree_idx):
            """Embed tree tree_idx (0-indexed, so tree_idx=19 is T20)."""
            tree = trees[tree_idx]
            n_tree = tree_idx + 1  # number of vertices in this tree
            
            if n_tree == 1:
                # Single vertex, any unused vertex works
                for v in range(20):
                    embeddings[tree_idx] = [v]
                    return True
                return False
            
            # Build adjacency for the tree
            adj = [[] for _ in range(n_tree)]
            for u, v in tree:
                adj[u].append(v)
                adj[v].append(u)
            
            # Use backtracking to find an embedding
            # State: mapping from tree vertices to K20 vertices
            mapping = [-1] * n_tree
            
            def backtrack(pos):
                """Assign tree vertices 0..pos-1, now assign pos."""
                if pos == n_tree:
                    # Check all edges are valid (not in used_edges)
                    for u, v in tree:
                        eu, ev = mapping[u], mapping[v]
                        ekey = (min(eu, ev), max(eu, ev))
                        if ekey in used_edges:
                            return False
                    # Also check no vertex conflicts (shouldn't happen if we track used vertices)
                    # But we need to ensure the embedding is injective
                    if len(set(mapping)) != n_tree:
                        return False
                    # Save embedding
                    embeddings[tree_idx] = list(mapping)
                    # Add edges to used_edges
                    for u, v in tree:
                        eu, ev = mapping[u], mapping[v]
                        used_edges.add((min(eu, ev), max(eu, ev)))
                    return True
                
                # Get candidates for vertex pos
                # A vertex can be mapped to any K20 vertex not already used
                used_verts = set(mapping[:pos])
                
                # Heuristic: try vertices that don't create immediate edge conflicts
                candidates = []
                for v in range(20):
                    if v in used_verts:
                        continue
                    # Check if any edge from pos to already-assigned neighbors would conflict
                    conflict = False
                    for nb in adj[pos]:
                        if mapping[nb] != -1:
                            ekey = (min(v, mapping[nb]), max(v, mapping[nb]))
                            if ekey in used_edges:
                                conflict = True
                                break
                    if not conflict:
                        candidates.append(v)
                
                # Shuffle for randomness
                Random(42 + pos + tree_idx).shuffle(candidates)
                
                for v in candidates:
                    mapping[pos] = v
                    if backtrack(pos + 1):
                        return True
                    mapping[pos] = -1
                
                return False
            
            # Start with vertex 0 mapped to vertex 0 (symmetry breaking)
            mapping[0] = 0
            return backtrack(1)
        
        # Process trees in embed_order
        for tree_idx in embed_order:
            if time.time() > time_limit:
                break
            if not embed_tree(tree_idx):
                break
        
        return embeddings
    
    # First attempt: deterministic backtracking
    result = solve_with_backtrack(trees, embed_order, deadline)
    
    # Count how many trees were successfully embedded
    embedded_count = sum(1 for e in result if e is not None)
    
    # If we couldn't embed all trees, try randomized restarts
    if embedded_count < 20 and time.time() < deadline - 10:
        best_result = result
        best_count = embedded_count
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Use randomized backtracking
            used_edges = set()
            embeddings = [None] * 20
            
            def embed_tree_random(tree_idx, rng):
                tree = trees[tree_idx]
                n_tree = tree_idx + 1
                
                if n_tree == 1:
                    v = rng.randrange(20)
                    embeddings[tree_idx] = [v]
                    return True
                
                adj = [[] for _ in range(n_tree)]
                for u, v in tree:
                    adj[u].append(v)
                    adj[v].append(u)
                
                mapping = [-1] * n_tree
                
                def backtrack(pos):
                    if time.time() > deadline - 5:
                        return False
                    
                    if pos == n_tree:
                        for u, v in tree:
                            eu, ev = mapping[u], mapping[v]
                            ekey = (min(eu, ev), max(eu, ev))
                            if ekey in used_edges:
                                return False
                        if len(set(mapping)) != n_tree:
                            return False
                        embeddings[tree_idx] = list(mapping)
                        for u, v in tree:
                            eu, ev = mapping[u], mapping[v]
                            used_edges.add((min(eu, ev), max(eu, ev)))
                        return True
                    
                    used_verts = set(mapping[:pos])
                    candidates = []
                    for v in range(20):
                        if v in used_verts:
                            continue
                        conflict = False
                        for nb in adj[pos]:
                            if mapping[nb] != -1:
                                ekey = (min(v, mapping[nb]), max(v, mapping[nb]))
                                if ekey in used_edges:
                                    conflict = True
                                    break
                        if not conflict:
                            candidates.append(v)
                    
                    rng.shuffle(candidates)
                    
                    for v in candidates:
                        mapping[pos] = v
                        if backtrack(pos + 1):
                            return True
                        mapping[pos] = -1
                    
                    return False
                
                mapping[0] = rng.randrange(20)
                return backtrack(1)
            
            for tree_idx in embed_order:
                if time.time() > deadline - 5:
                    break
                rng = Random(attempt * 1000 + tree_idx)
                if not embed_tree_random(tree_idx, rng):
                    break
            
            count = sum(1 for e in embeddings if e is not None)
            if count > best_count:
                best_count = count
                best_result = embeddings
                if best_count == 20:
                    break
        
        result = best_result
    
    # Convert to proper format: list of 20 entries, each None or list of ints
    final_result = []
    for i in range(20):
        if result[i] is None:
            final_result.append(None)
        else:
            # Ensure all values are Python ints
            final_result.append([int(x) for x in result[i]])
    
    return final_result
