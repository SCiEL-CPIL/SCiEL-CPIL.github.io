import time
import random
from itertools import combinations

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
    deadline = time.time() + 140
    trees = instance_trees(20, 611)
    
    # Precompute adjacency lists for trees
    tree_adj = []
    for k in range(1, 21):
        adj = [[] for _ in range(k)]
        for (u, v) in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    best_result = [None] * 20
    best_edges = 0
    
    # We will try to pack trees in different orders
    # A good heuristic is often largest first, but we can vary it.
    base_order = list(range(20, 0, -1))
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if time.time() > deadline:
            break
            
        # Generate a deterministic but varying random seed for each attempt
        # We mix the attempt number with a large prime to decorrelate
        seed = attempts * 104729 + 12345
        rng = random.Random(seed)
        
        # Create a variation of the order
        # Mostly keep the largest first, but occasionally shuffle the smaller ones
        # or try a completely random permutation if we are stuck
        if attempts % 5 == 0:
            order = base_order[:]
            rng.shuffle(order)
        else:
            order = base_order[:]
            # Shuffle the last 5 elements (smaller trees) to try different embeddings for them
            # since they are easier to place, the hard part is the big trees
            tail = order[-6:]
            rng.shuffle(tail)
            order = order[:-6] + tail
        
        used_edges = set()
        result = [None] * 20
        total_edges = 0
        
        # Map tree index (1-based) to result index (0-based)
        # result[k-1] corresponds to tree T_k
        
        success = True
        
        for k in order:
            if time.time() > deadline:
                success = False
                break
                
            if k == 1:
                # T1 is a single vertex
                result[0] = [rng.randint(0, 19)]
                continue
                
            if k == 2:
                # T2 is a single edge
                found = False
                # Try to find an unused edge
                # To be faster, we can just iterate
                for a in range(20):
                    for b in range(a + 1, 20):
                        edge_id = a * 20 + b
                        if edge_id not in used_edges:
                            used_edges.add(edge_id)
                            result[1] = [a, b]
                            total_edges += 1
                            found = True
                            break
                    if found:
                        break
                if not found:
                    success = False
                    break
                continue
            
            # For k >= 3, we need to embed the tree T_k
            # T_k has k vertices and k-1 edges
            adj = tree_adj[k - 1]
            
            # Heuristic for vertex ordering in the tree to embed
            # We want to pick vertices that are most constrained.
            # A common heuristic is to order by degree (descending).
            # However, dynamic ordering is better.
            # We will use a backtracking search with MRV (Minimum Remaining Values) heuristic.
            
            # Pre-compute initial candidate lists for host vertices?
            # No, we will do it dynamically.
            
            # State for backtracking:
            # mapping: tree_vertex -> host_vertex
            # used_hosts: set of host vertices already used
            # We also need to check edge constraints.
            
            mapping = [-1] * k
            used_hosts = [False] * 20
            
            # Order in which we assign tree vertices
            # Start with a vertex, then pick the next one that has the most assigned neighbors
            # Or simply pick a static order based on degree first, then refine?
            # Let's try a dynamic MRV approach.
            
            # Initial assignment: pick a random start vertex for the tree
            start_tree_v = rng.randint(0, k - 1)
            start_host_v = rng.randint(0, 19)
            
            mapping[start_tree_v] = start_host_v
            used_hosts[start_host_v] = True
            
            # Stack for DFS
            # Each item: (current_tree_v, next_tree_v_to_assign, assigned_count)
            # Actually, it's easier to just track which tree vertices are assigned.
            
            # Let's use a simple recursive-like stack for the tree vertices.
            # We will maintain a list of unassigned tree vertices.
            # At each step, pick an unassigned tree vertex that has at least one assigned neighbor
            # (to constrain the host vertex choice) or just any if none (only for the first).
            # Prefer vertices with more assigned neighbors (MRV for the host choice).
            
            # We need to track the neighbors in the tree.
            # tree_adj is already built.
            
            unassigned = [i for i in range(k) if i != start_tree_v]
            
            def is_valid(host_v, tree_v):
                # Check if host_v can be used for tree_v
                if used_hosts[host_v]:
                    return False
                # Check edges to already assigned neighbors
                for nb in adj[tree_v]:
                    if mapping[nb] != -1:
                        host_nb = mapping[nb]
                        # Check edge (host_v, host_nb) in K20
                        if host_v < host_nb:
                            e_id = host_v * 20 + host_nb
                        else:
                            e_id = host_nb * 20 + host_v
                        if e_id in used_edges:
                            return False
                return True
            
            # Backtracking search
            # To avoid recursion depth issues and overhead, we can use a stack.
            # But recursion is cleaner and k is max 20, so depth is 20.
            # Python recursion limit is usually 1000, so it's fine.
            # However, function calls are slow.
            # Let's try to optimize the recursive call by passing state as arguments or using non-local.
            
            # Let's use a nested function with non-local state for clarity, 
            # but add a time check.
            
            search_deadline = time.time() + 2.0 # Max 2 seconds per tree
            node_count = 0
            
            def backtrack():
                nonlocal node_count
                if node_count > 100000: # Pruning cap
                    return False
                if time.time() > search_deadline:
                    return False
                
                if not unassigned:
                    # All assigned
                    # Record the edges used by this tree
                    for u in range(k):
                        for v in adj[u]:
                            if u < v:
                                hu = mapping[u]
                                hv = mapping[v]
                                if hu < hv:
                                    used_edges.add(hu * 20 + hv)
                                else:
                                    used_edges.add(hv * 20 + hu)
                    return True
                
                # Select next tree vertex to assign
                # Heuristic: Pick the unassigned vertex with the most assigned neighbors
                best_v = -1
                max_assigned_nbs = -1
                best_host_candidates = []
                
                # We could pre-filter unassigned vertices that have at least one assigned neighbor
                # If none have assigned neighbors, pick any (shouldn't happen after start)
                
                for tv in unassigned:
                    assigned_nbs = [nb for nb in adj[tv] if mapping[nb] != -1]
                    if len(assigned_nbs) > max_assigned_nbs:
                        max_assigned_nbs = len(assigned_nbs)
                        best_v = tv
                        
                if best_v == -1:
                    # No unassigned vertices? Should not happen
                    return False
                    
                # If no unassigned vertex has assigned neighbors (isolated components?), 
                # just pick the first one. But trees are connected, so this only happens 
                # if we start in the middle? No, we started with one.
                # If the tree is connected, there is always a frontier.
                
                # Generate candidates for best_v
                candidates = []
                for h in range(20):
                    if not used_hosts[h] and is_valid(h, best_v):
                        candidates.append(h)
                        
                if not candidates:
                    return False
                    
                # Shuffle candidates to randomize search
                rng.shuffle(candidates)
                
                for h in candidates:
                    node_count += 1
                    if time.time() > search_deadline:
                        return False
                        
                    # Assign
                    mapping[best_v] = h
                    used_hosts[h] = True
                    unassigned.remove(best_v)
                    
                    if backtrack():
                        return True
                    
                    # Unassign
                    mapping[best_v] = -1
                    used_hosts[h] = False
                    unassigned.append(best_v)
                    
                return False
            
            if backtrack():
                result[k - 1] = [mapping[i] for i in range(k)]
                total_edges += k - 1
            else:
                success = False
                break
                
        if success and total_edges > best_edges:
            best_edges = total_edges
            best_result = result[:]
            if best_edges == 190:
                break
                
    return best_result
