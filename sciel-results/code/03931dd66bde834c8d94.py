"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    deadline = start_time + 110.0
    
    trees = instance_trees(20, 611)
    n_trees = 20
    N = 20  # host vertices
    
    # Precompute adjacency for each tree
    tree_adjs = []
    for i, edges in enumerate(trees):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adjs.append(adj)
    
    # Order trees by decreasing size (harder first)
    order = list(range(n_trees - 1, -1, -1))
    
    def backtrack(idx, used_edges, mappings):
        if time.time() > deadline:
            return None
        
        if idx == n_trees:
            return mappings[:]
        
        tree_idx = order[idx]
        k = tree_idx + 1
        edges = trees[tree_idx]
        adj = tree_adjs[tree_idx]
        
        if k == 1:
            mappings[tree_idx] = [0]
            result = backtrack(idx + 1, used_edges, mappings)
            if result is not None:
                return result
            mappings[tree_idx] = None
            return None
        
        if k == 2:
            # Try to find an available edge
            found = False
            for u in range(N):
                for v in range(u + 1, N):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        mappings[tree_idx] = [u, v]
                        used_edges.add(e_key)
                        result = backtrack(idx + 1, used_edges, mappings)
                        if result is not None:
                            return result
                        used_edges.remove(e_key)
                        mappings[tree_idx] = None
            return None
        
        # For k > 2, use randomized backtracking
        # Try multiple random mappings
        max_attempts = 30
        rng = Random(time.time_ns() % (2**32))
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random permutation of host vertices
            host_verts = list(range(N))
            rng.shuffle(host_verts)
            mapping = host_verts[:k]
            
            # Check if all edges map to available edges
            valid = True
            edges_to_add = []
            for u, v in edges:
                hu = mapping[u]
                hv = mapping[v]
                e_key = (min(hu, hv), max(hu, hv))
                if e_key in used_edges:
                    valid = False
                    break
                edges_to_add.append(e_key)
            
            if valid:
                # Apply mapping
                for e in edges_to_add:
                    used_edges.add(e)
                mappings[tree_idx] = mapping
                
                result = backtrack(idx + 1, used_edges, mappings)
                if result is not None:
                    return result
                
                # Undo
                for e in edges_to_add:
                    used_edges.remove(e)
                mappings[tree_idx] = None
        
        return None
    
    # Try multiple restarts
    best_mappings = None
    best_score = 0
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        used_edges = set()
        mappings = [None] * n_trees
        
        # Try to solve
        result = backtrack(0, used_edges, mappings)
        
        if result is not None:
            # Calculate score
            score = 0
            for i in range(n_trees):
                if result[i] is not None:
                    score += len(trees[i])
            
            if score > best_score:
                best_score = score
                best_mappings = result[:]
                
            if best_score == 190:
                break
    
    # Fallback: greedy embedding if backtracking fails
    if best_mappings is None or best_score < 190:
        used_edges = set()
        mappings = [None] * n_trees
        score = 0
        
        for tree_idx in order:
            if time.time() > deadline:
                break
                
            k = tree_idx + 1
            edges = trees[tree_idx]
            
            if k == 1:
                mappings[tree_idx] = [0]
                continue
                
            if k == 2:
                found = False
                for u in range(N):
                    for v in range(u + 1, N):
                        e_key = (u, v)
                        if e_key not in used_edges:
                            mappings[tree_idx] = [u, v]
                            used_edges.add(e_key)
                            score += 1
                            found = True
                            break
                    if found:
                        break
                continue
            
            # Try random mappings
            rng = Random(tree_idx * 1000)
            for attempt in range(50):
                if time.time() > deadline:
                    break
                    
                host_verts = list(range(N))
                rng.shuffle(host_verts)
                mapping = host_verts[:k]
                
                valid = True
                edges_to_add = []
                for u, v in edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        valid = False
                        break
                    edges_to_add.append(e_key)
                
                if valid:
                    for e in edges_to_add:
                        used_edges.add(e)
                    mappings[tree_idx] = mapping
                    score += len(edges)
                    break
        
        if score > best_score:
            best_score = score
            best_mappings = mappings[:]
    
    # Construct final result
    if best_mappings is None:
        best_mappings = [None] * n_trees
    
    final_result = []
    for i in range(n_trees):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])
    
    return final_result

# EVOLVE-BLOCK-END
