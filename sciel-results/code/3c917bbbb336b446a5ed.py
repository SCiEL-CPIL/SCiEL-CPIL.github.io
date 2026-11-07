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
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    n = 20
    trees = instance_trees(n, 610)
    # Actually seed is 611 in the task description, but let's stick to the provided instance_trees call
    trees = instance_trees(n, 611)
    
    # Precompute tree adjacencies
    tree_adjs = []
    for i, edges in enumerate(trees):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adjs.append(adj)
    
    # Order: largest first
    order = list(range(n - 1, -1, -1))
    
    best_score = 0
    best_mappings = [None] * n
    
    # We will use a randomized greedy backtracking approach.
    # Instead of deep backtracking which might be slow, we do many random restarts.
    # In each restart, we pick random embeddings for large trees, then try to fill the rest.
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        used_edges = set()
        mappings = [None] * n
        current_score = 0
        
        # Process trees in 'order'
        success = True
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                # T1: just pick a vertex
                mappings[t_idx] = [0]
                continue
                
            edges = trees[t_idx]
            adj = tree_adjs[t_idx]
            
            # Try to find a valid embedding for this tree
            # We use a randomized DFS/Backtracking specifically for this tree
            # to find an embedding that uses only available edges.
            
            # For small trees (k <= 5), we can be exhaustive or semi-exhaustive quickly.
            # For larger trees, we rely on randomization.
            
            found = False
            attempts = 50 if k > 10 else 200
            
            for attempt in range(attempts):
                if time.time() > deadline:
                    success = False
                    break
                    
                # Randomize the order of vertices to try in the embedding
                # We map tree vertices 0..k-1 to host vertices 0..19
                # We need a bijection from {0..k-1} to a subset of {0..19}
                
                # Strategy: Pick a random root, then BFS/DFS with random ordering
                # Or simpler: Just pick a random permutation of 20 vertices, take first k.
                # This is fast but might be weak.
                # Better: Randomized backtracking on the tree structure.
                
                # Let's use the randomized backtracking from the inspiration but optimized.
                # We'll try to embed the tree using a DFS that respects available edges.
                
                # To speed up, we pick a random root vertex in the host graph
                root_host = random.randint(0, n - 1)
                
                # We need to map tree vertex 0 to root_host? No, we can pick any tree vertex as root.
                # Let's pick tree vertex 0 as root for simplicity, or a high-degree one.
                # Actually, let's just try random mappings of the tree vertices to host vertices
                # subject to edge constraints.
                
                # Efficient method:
                # 1. Pick a random permutation of host vertices: perm
                # 2. Try to map tree vertices in a BFS order.
                # This is complex. Let's stick to the "random subset + check" for speed,
                # but improve it by checking edges incrementally.
                
                # Alternative: Randomized DFS on the tree.
                # State: mapping of tree vertices to host vertices.
                # Start with an empty mapping.
                # Pick a random unassigned tree vertex that is connected to an assigned one (or root if none).
                # Try random host vertices for it that are connected to the assigned parent.
                
                mapping = [-1] * k
                # Assign root
                # Pick a random tree vertex as root
                t_root = random.randint(0, k - 1)
                mapping[t_root] = root_host
                
                # Queue of (tree_vertex, parent_tree_vertex)
                # We need to ensure we visit all tree vertices.
                # Let's do a BFS/DFS from t_root.
                visited = [False] * k
                visited[t_root] = True
                stack = [t_root]
                
                valid_embed = True
                
                while stack:
                    tv = stack.pop()
                    for neighbor in adj[tv]:
                        if visited[neighbor]:
                            continue
                        visited[neighbor] = True
                        # Find candidates for neighbor in host graph
                        # Neighbor must be connected to mapping[tv]
                        host_tv = mapping[tv]
                        candidates = []
                        for hv in range(n):
                            if hv == host_tv:
                                continue
                            # Check if hv is already used in mapping
                            if hv in mapping:
                                continue
                            # Check if edge (host_tv, hv) is available
                            e = (min(host_tv, hv), max(host_tv, hv))
                            if e in used_edges:
                                continue
                            candidates.append(hv)
                        
                        if not candidates:
                            valid_embed = False
                            break
                            
                        # Pick a random candidate
                        hv = random.choice(candidates)
                        mapping[neighbor] = hv
                        stack.append(neighbor)
                        
                        # Optimization: If we run out of candidates later, we might need to backtrack.
                        # But this greedy approach is fast. If it fails, we retry with a new random seed.
                        
                        # Check if all edges are valid?
                        # We only check parent-child edges. Since it's a tree, this covers all edges.
                        
                        # Wait, we need to ensure that the edge (tv, neighbor) is indeed in used_edges?
                        # No, we are ADDING it to used_edges. So it must NOT be in used_edges yet.
                        # We checked that.
                        
                        # But we also need to ensure that we don't create a conflict with future edges.
                        # The greedy nature might lead to dead ends.
                        
                        if not valid_embed:
                            break
                            
                if not valid_embed:
                    continue
                    
                # Check if all tree vertices are mapped
                if not all(visited):
                    continue
                    
                # Verify all edges are available and distinct
                edges_to_add = []
                valid = True
                for u, v in edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    if hu == -1 or hv == -1:
                        valid = False
                        break
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        valid = False
                        break
                    edges_to_add.append(e)
                    
                if not valid:
                    continue
                    
                # Accept this embedding
                for e in edges_to_add:
                    used_edges.add(e)
                mappings[t_idx] = [int(x) for x in mapping]
                current_score += len(edges)
                found = True
                break
                
            if not found:
                # Could not embed this tree
                success = False
                break
        
        if success:
            # All trees embedded
            if current_score > best_score:
                best_score = current_score
                best_mappings = mappings[:]
                
            if best_score == 190:
                break
                
    # If we didn't find a perfect packing, try a greedy fill for the remaining?
    # The problem asks for the raw value, which is the total number of embedded tree edges.
    # If we fail to embed a large tree, we can try to embed smaller ones.
    # But our loop breaks on failure.
    # Let's modify: if a tree fails, skip it and try the next ones.
    # This might yield a partial solution with higher score than 0.
    
    # Let's re-run with a "skip on fail" strategy if we didn't get 190.
    if best_score < 190:
        for restart in range(200):
            if time.time() > deadline:
                break
                
            used_edges = set()
            mappings = [None] * n
            current_score = 0
            
            for t_idx in order:
                k = t_idx + 1
                if k == 1:
                    mappings[t_idx] = [0]
                    continue
                    
                edges = trees[t_idx]
                adj = tree_adjs[t_idx]
                
                found = False
                attempts = 20
                
                for attempt in range(attempts):
                    if time.time() > deadline:
                        break
                        
                    root_host = random.randint(0, n - 1)
                    t_root = random.randint(0, k - 1)
                    mapping = [-1] * k
                    mapping[t_root] = root_host
                    visited = [False] * k
                    visited[t_root] = True
                    stack = [t_root]
                    valid_embed = True
                    
                    while stack:
                        tv = stack.pop()
                        for neighbor in adj[tv]:
                            if visited[neighbor]:
                                continue
                            visited[neighbor] = True
                            host_tv = mapping[tv]
                            candidates = []
                            for hv in range(n):
                                if hv == host_tv:
                                    continue
                                if hv in mapping:
                                    continue
                                e = (min(host_tv, hv), max(host_tv, hv))
                                if e in used_edges:
                                    continue
                                candidates.append(hv)
                                
                            if not candidates:
                                valid_embed = False
                                break
                                
                            hv = random.choice(candidates)
                            mapping[neighbor] = hv
                            stack.append(neighbor)
                            
                            if not valid_embed:
                                break
                                
                    if not valid_embed:
                        continue
                        
                    if not all(visited):
                        continue
                        
                    edges_to_add = []
                    valid = True
                    for u, v in edges:
                        hu = mapping[u]
                        hv = mapping[v]
                        if hu == -1 or hv == -1:
                            valid = False
                            break
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            valid = False
                            break
                        edges_to_add.append(e)
                        
                    if not valid:
                        continue
                        
                    for e in edges_to_add:
                        used_edges.add(e)
                    mappings[t_idx] = [int(x) for x in mapping]
                    current_score += len(edges)
                    found = True
                    break
                    
                # If not found, skip this tree (mappings[t_idx] remains None)
                
            if current_score > best_score:
                best_score = current_score
                best_mappings = mappings[:]
                
            if best_score == 190:
                break
                
    # Final check and formatting
    final_result = []
    for i in range(n):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            # Ensure distinct integers
            # The mapping should be distinct by construction (hv in mapping check)
            final_result.append([int(x) for x in best_mappings[i]])
            
    return final_result

# EVOLVE-BLOCK-END
