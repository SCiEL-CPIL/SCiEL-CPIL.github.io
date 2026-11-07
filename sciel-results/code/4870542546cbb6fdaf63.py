import time
from random import Random

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
    deadline = time.time() + 145  # Safety margin before 150s limit
    trees = instance_trees(20, 611)
    
    # Pre-process trees: adjacency lists
    tree_adj = []
    for k in range(1, 21):
        adj = [[] for _ in range(k)]
        for (u, v) in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        
    # Order trees: Largest first (k=20 down to 1)
    # This is critical for backtracking efficiency
    order = list(range(20, 0, -1))
    
    best_result = [None] * 20
    best_edges = 0
    
    # We will use randomized restarts with backtracking
    # Symmetry breaking: For the first tree (T_20), fix vertex 0 to host vertex 0.
    # Since K_20 is vertex-transitive and no edges are used yet, this is valid.
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        rng = Random(attempt * 12345 + 6789)
        
        used_edges = set()
        result = [None] * 20
        success = True
        
        # Track partial progress for best_edges update if full success fails
        current_edges = 0
        
        for k in order:
            if time.time() > deadline:
                success = False
                break
                
            if k == 1:
                # T_1 is a single vertex. Map to any vertex.
                # To be deterministic and simple, map to 0.
                # Or random? Doesn't matter for edges.
                result[0] = [0]
                continue
                
            if k == 2:
                # T_2 is a single edge. Find any unused edge.
                found = False
                # Systematic search is fast enough
                for a in range(20):
                    for b in range(a+1, 20):
                        edge = (a, b)
                        if edge not in used_edges:
                            used_edges.add(edge)
                            result[1] = [a, b]
                            current_edges += 1
                            found = True
                            break
                    if found: break
                if not found:
                    success = False
                    break
                continue
                
            # For k >= 3
            adj = tree_adj[k-1]
            k_val = k
            
            # Determine vertex ordering for backtracking
            # High degree first
            deg = [len(adj[i]) for i in range(k_val)]
            # Sort indices by degree descending, then by index for stability
            v_order = sorted(range(k_val), key=lambda x: (-deg[x], x))
            
            mapping = [-1] * k_val
            host_used = [False] * 20 # Local to this tree
            
            # Heuristic for candidate ordering:
            # For a tree vertex tv, and a candidate host h,
            # we want to prioritize h that leaves the most flexibility for unassigned neighbors?
            # Or fail fast?
            # Let's use a simple score: number of available edges from h to unassigned neighbors of tv.
            # Lower score = more constrained = try first (fail fast).
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == k_val:
                    return True
                
                tv = v_order[idx]
                
                # Identify unassigned neighbors of tv
                unassigned_nbs = [nb for nb in adj[tv] if mapping[nb] == -1]
                
                candidates = []
                for h in range(20):
                    if host_used[h]:
                        continue
                    
                    # Check edges to assigned neighbors
                    ok = True
                    for nb in adj[tv]:
                        if mapping[nb] != -1:
                            hnb = mapping[nb]
                            edge = (min(h, hnb), max(h, hnb))
                            if edge in used_edges:
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    # Calculate heuristic score
                    # Count available edges from h to unassigned neighbors
                    # This is an estimate of flexibility.
                    # We want to try candidates that are "tight" first to prune search tree.
                    # So, sort by count ASCENDING.
                    avail_count = 0
                    for nb in unassigned_nbs:
                        # nb is unassigned, so we don't know its host yet.
                        # But we can count how many host vertices v (not used yet) have edge (h,v) free.
                        # This is expensive. 
                        # Simpler heuristic: just shuffle.
                        # Or, prefer h with higher degree in the residual graph?
                        # Let's stick to random shuffle for now, but maybe bias towards
                        # vertices that are "central" in the remaining available graph.
                        # For simplicity and speed, random shuffle is robust.
                        pass
                    
                    candidates.append(h)
                        
                if not candidates:
                    return False
                    
                # Shuffle for randomization
                rng.shuffle(candidates)
                
                # Optional: Sort candidates by some heuristic
                # For now, random order is good for diversification across restarts.
                
                for h in candidates:
                    mapping[tv] = h
                    host_used[h] = True
                    
                    if backtrack(idx + 1):
                        return True
                        
                    mapping[tv] = -1
                    host_used[h] = False
                    
                return False
            
            # Symmetry breaking for the first tree (k=20)
            if k == 20:
                # Fix tree vertex 0 to host vertex 0
                # But v_order[0] might not be 0.
                # We need to handle the first assignment specially.
                
                # If v_order[0] is the vertex we want to fix, we can do it.
                # Actually, it's easier to just force the first assignment in the loop.
                # Let's modify the backtrack call or pre-assign.
                
                # Let's pre-assign the first vertex in v_order to a specific host.
                # Which host? 0 is a good choice due to symmetry.
                first_tv = v_order[0]
                
                # Check if host 0 is valid for first_tv
                # Since it's the first tree, no edges are used, so any host is valid for any vertex.
                # So we can safely set mapping[first_tv] = 0.
                
                mapping[first_tv] = 0
                host_used[0] = True
                
                # Now backtrack for the rest
                if backtrack(1):
                    result[k-1] = [mapping[i] for i in range(k_val)]
                    for (u, v) in trees[k-1]:
                        hu, hv = mapping[u], mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    current_edges += k_val - 1
                else:
                    success = False
                    break
            else:
                # No symmetry breaking for subsequent trees
                if backtrack(0):
                    result[k-1] = [mapping[i] for i in range(k_val)]
                    for (u, v) in trees[k-1]:
                        hu, hv = mapping[u], mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    current_edges += k_val - 1
                else:
                    success = False
                    break
            
        if success:
            total_edges = current_edges
            if total_edges > best_edges:
                best_edges = total_edges
                best_result = result[:]
                if best_edges == 190:
                    break
        else:
            # Update best with partial result if better
            # Note: result contains valid mappings for trees packed before failure.
            # We need to ensure we only count edges for successfully packed trees.
            # The `current_edges` variable tracks this.
            if current_edges > best_edges:
                best_edges = current_edges
                # We need to copy the partial result correctly.
                # result[k-1] is None for failed and subsequent trees.
                # This is correct.
                best_result = result[:]

    return best_result
