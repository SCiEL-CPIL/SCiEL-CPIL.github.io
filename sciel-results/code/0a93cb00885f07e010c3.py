"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
    deadline = start_time + 115.0
    
    # Generate the fixed trees
    trees = instance_trees(20, 611)
    
    # Pre-compute tree properties
    # tree_degrees[k_idx] is a list of degrees for vertices 0..k-1 in tree k
    tree_degrees = []
    tree_adj = []
    for k_idx in range(20):
        k = k_idx + 1
        edges = trees[k_idx]
        if k == 1:
            tree_degrees.append([0])
            tree_adj.append([])
            continue
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        tree_degrees.append(deg)
        tree_adj.append(adj)
    
    # We want to pack T1..T20 into K20.
    # Total edges = 190. K20 has 190 edges.
    # This is an exact decomposition.
    
    # Strategy:
    # Embed trees in decreasing order of size (T20, T19, ..., T1).
    # For each tree, we need to find a mapping of its vertices to host vertices 0..19
    # such that all tree edges map to unused host edges.
    
    # To make this deterministic and fast, we use a greedy heuristic with a twist:
    # We maintain the "remaining degree" of each host vertex in the unused edge graph.
    # Initially, every host vertex has degree 19 in K20.
    # When we place an edge (u, v) in the host, we reduce the remaining degree of u and v by 1.
    
    # For a tree T with k vertices and degrees d_0..d_{k-1},
    # if we map tree vertex i to host vertex h_i, then the sum of degrees of the mapped vertices
    # must be at least the sum of degrees in the tree? No, that's not a strict constraint per vertex.
    # However, a high-degree vertex in the tree should map to a host vertex with high remaining degree.
    
    # Algorithm:
    # 1. Sort trees by size descending.
    # 2. For each tree, sort its vertices by degree descending.
    # 3. Sort available host vertices by remaining degree descending.
    # 4. Try to match the highest degree tree vertex to the highest degree host vertex.
    # 5. This is a bipartite matching problem if we consider just degrees, but we also need
    #    the edges to exist and be unused.
    
    # Since K20 is dense, we can try a "greedy assignment" where we pick the best host vertex
    # for each tree vertex, prioritizing those with high remaining degree and many unused edges
    # to other already-placed tree vertices.
    
    # Let's implement a recursive backtracking with strong heuristics (MRV + Value Ordering).
    # But to be structurally different from the previous "random restart" approach,
    # we will use a **Deterministic Greedy with Limited Backtracking** approach.
    # We will try to solve it greedily. If it fails, we backtrack only a few steps.
    
    # Actually, a more robust structural change is to use **Simulated Annealing** or **Local Search**
    # on the set of mappings. But that's complex.
    
    # Let's stick to a very aggressive backtracking with **Forward Checking** and **Constraint Propagation**.
    # The previous program had weak pruning. We will implement strong pruning.
    
    # State:
    # used_edges: set of (u, v) with u < v
    # host_deg: array of length 20, remaining degree in unused graph
    
    # We will embed trees in order: 19, 18, ..., 0 (indices).
    
    order = list(range(19, -1, -1))
    
    best_result = [None] * 20
    best_score = 0
    
    def solve(order, rng):
        used_edges = set()
        host_deg = [19] * 20
        mappings = [None] * 20
        score = 0
        
        # Pre-compute for each tree, the list of vertices sorted by degree descending
        # and the edges incident to each vertex.
        
        def backtrack(idx):
            nonlocal used_edges, host_deg, mappings, score, best_result, best_score
            
            if time.time() > deadline:
                return False
            
            # Update best partial result
            if score > best_score:
                best_score = score
                best_result = [m[:] if m else None for m in mappings]
                
            if idx == len(order):
                return True
            
            k_idx = order[idx]
            k = k_idx + 1
            edges = trees[k_idx]
            t_adj = tree_adj[k_idx]
            t_deg = tree_degrees[k_idx]
            
            if k == 1:
                # Map to any vertex. Prefer one with high degree? Doesn't matter for edges.
                # Just pick 0 for simplicity, or random.
                mappings[k_idx] = [0]
                if backtrack(idx + 1):
                    return True
                mappings[k_idx] = None
                return False
            
            # We need to assign k distinct host vertices to the k tree vertices.
            # To reduce search space, we fix the mapping of the first tree vertex (highest degree)
            # to a specific set of candidate host vertices.
            
            # Sort tree vertices by degree descending
            t_verts_sorted = sorted(range(k), key=lambda v: -t_deg[v])
            
            # Candidate host vertices: all 0..19, sorted by host_deg descending
            # We will try to assign t_verts_sorted[0] to the top few host vertices.
            
            # Optimization: If the tree has a unique highest degree vertex,
            # and the host graph has a unique highest degree vertex, it's a strong candidate.
            
            # Let's try all possible host vertices for the first tree vertex,
            # ordered by host_deg descending.
            
            # To avoid symmetry breaking issues, we can just iterate through host vertices 0..19
            # sorted by current remaining degree.
            
            host_candidates = sorted(range(20), key=lambda h: -host_deg[h])
            
            # Limit candidates to top 10 to keep branching factor manageable?
            # For the first vertex of the largest tree, we might need to try many.
            # But with backtracking, if we pick a bad one, we unwind.
            
            for h0 in host_candidates:
                # Assign tree vertex t_verts_sorted[0] to host h0
                mapping = [-1] * k
                mapping[t_verts_sorted[0]] = h0
                
                # We need to assign the remaining k-1 tree vertices.
                # We can use a recursive helper that assigns the next tree vertex.
                
                # To speed up, we can check if it's possible to place the rest.
                # A simple necessary condition: The sum of remaining degrees of the host vertices
                # must be sufficient to support the degrees of the remaining tree vertices.
                # Sum of degrees in tree = 2 * (k-1).
                # Sum of degrees of mapped host vertices must be >= 2 * (k-1)? No,
                # the mapped host vertices will use some of their remaining degree for internal tree edges,
                # and the rest for external edges.
                
                # Let's just proceed with backtracking for the remaining vertices.
                
                def assign_next(t_idx):
                    if t_idx == len(t_verts_sorted):
                        # All vertices assigned. Check if all edges are valid.
                        # Actually, we should check edges as we assign to prune early.
                        # Let's restructure: assign vertex, check edges to already assigned neighbors.
                        
                        # If we reach here, we have a complete mapping.
                        # We need to verify all edges are unused and add them.
                        valid = True
                        used_here = []
                        for u, v in edges:
                            hu = mapping[u]
                            hv = mapping[v]
                            e_key = (min(hu, hv), max(hu, hv))
                            if e_key in used_edges:
                                valid = False
                                break
                            used_here.append(e_key)
                        
                        if valid:
                            # Apply
                            for e in used_here:
                                u, v = e
                                used_edges.add(e)
                                host_deg[u] -= 1
                                host_deg[v] -= 1
                            mappings[k_idx] = mapping
                            score += len(edges)
                            
                            if backtrack(idx + 1):
                                return True
                            
                            # Undo
                            for e in used_here:
                                u, v = e
                                used_edges.remove(e)
                                host_deg[u] += 1
                                host_deg[v] += 1
                            mappings[k_idx] = None
                            score -= len(edges)
                        
                        return False
                    
                    tv = t_verts_sorted[t_idx]
                    # Try all available host vertices for tv
                    # Heuristic: Sort available host vertices by:
                    # 1. Number of unused edges to already assigned tree neighbors (higher is better)
                    # 2. Remaining degree (higher is better)
                    
                    assigned_hosts = [mapping[v] for v in range(k) if mapping[v] != -1]
                    
                    candidates = []
                    for h in range(20):
                        if h in assigned_hosts:
                            continue
                        # Count unused edges from h to assigned_hosts that correspond to tree edges from tv
                        # Actually, we just need h to be connected to the tree neighbors of tv that are already placed.
                        # Let's calculate a score.
                        score_h = 0
                        # For each neighbor of tv in the tree, if that neighbor is already assigned to a host,
                        # check if the edge (h, host_neighbor) is unused.
                        for nb in t_adj[tv]:
                            if mapping[nb] != -1:
                                hn = mapping[nb]
                                e_key = (min(h, hn), max(h, hn))
                                if e_key not in used_edges:
                                    score_h += 1
                        # Add host_deg as a tie-breaker
                        candidates.append((score_h, host_deg[h], h))
                    
                    # Sort candidates: prioritize those that satisfy more tree edges, then high degree
                    candidates.sort(key=lambda x: (-x[0], -x[1]))
                    
                    # Limit candidates to top 20 to avoid explosion?
                    # If score_h is 0 for all, we might be stuck.
                    
                    for _, _, h in candidates[:20]: # Limit to 20 candidates
                        mapping[tv] = h
                        
                        # Check if any edge to an already assigned neighbor is used
                        # This is a forward checking step
                        valid_step = True
                        for nb in t_adj[tv]:
                            if mapping[nb] != -1:
                                hn = mapping[nb]
                                e_key = (min(h, hn), max(h, hn))
                                if e_key in used_edges:
                                    valid_step = False
                                    break
                        
                        if valid_step:
                            if assign_next(t_idx + 1):
                                return True
                        
                        mapping[tv] = -1
                    
                    return False
                
                if assign_next(1):
                    return True
                
                # Undo h0 assignment? It's local to this loop iteration.
                # The mapping is reset in the next iteration of the for loop?
                # No, mapping is a local variable in the outer scope of assign_next,
                # but we modify it. We need to ensure it's clean for the next h0.
                # Actually, mapping is defined in the outer scope of the for h0 loop.
                # We should reset it or handle it carefully.
                # Let's make mapping a local variable inside the for h0 loop.
                
                pass # The logic above has a scope issue. Let's refactor.
            
            return False
        
        # Refactored Backtracking
        def backtrack_refactored(idx):
            nonlocal used_edges, host_deg, mappings, score, best_result, best_score
            
            if time.time() > deadline:
                return False
            
            if score > best_score:
                best_score = score
                best_result = [m[:] if m else None for m in mappings]
                
            if idx == len(order):
                return True
            
            k_idx = order[idx]
            k = k_idx + 1
            edges = trees[k_idx]
            t_adj = tree_adj[k_idx]
            t_deg = tree_degrees[k_idx]
            
            if k == 1:
                mappings[k_idx] = [0]
                if backtrack_refactored(idx + 1):
                    return True
                mappings[k_idx] = None
                return False
            
            t_verts_sorted = sorted(range(k), key=lambda v: -t_deg[v])
            host_candidates = sorted(range(20), key=lambda h: -host_deg[h])
            
            for h0 in host_candidates:
                mapping = [-1] * k
                mapping[t_verts_sorted[0]] = h0
                assigned_hosts = {h0}
                
                def assign_next(t_idx):
                    if t_idx == len(t_verts_sorted):
                        # All assigned. Verify edges.
                        valid = True
                        used_here = []
                        for u, v in edges:
                            hu = mapping[u]
                            hv = mapping[v]
                            e_key = (min(hu, hv), max(hu, hv))
                            if e_key in used_edges:
                                valid = False
                                break
                            used_here.append(e_key)
                        
                        if valid:
                            for e in used_here:
                                u, v = e
                                used_edges.add(e)
                                host_deg[u] -= 1
                                host_deg[v] -= 1
                            mappings[k_idx] = mapping[:]
                            score += len(edges)
                            
                            if backtrack_refactored(idx + 1):
                                return True
                            
                            for e in used_here:
                                u, v = e
                                used_edges.remove(e)
                                host_deg[u] += 1
                                host_deg[v] += 1
                            mappings[k_idx] = None
                            score -= len(edges)
                        
                        return False
                    
                    tv = t_verts_sorted[t_idx]
                    
                    # Generate candidates
                    candidates = []
                    for h in range(20):
                        if h in assigned_hosts:
                            continue
                        score_h = 0
                        for nb in t_adj[tv]:
                            if mapping[nb] != -1:
                                hn = mapping[nb]
                                e_key = (min(h, hn), max(h, hn))
                                if e_key not in used_edges:
                                    score_h += 1
                        candidates.append((score_h, host_deg[h], h))
                    
                    candidates.sort(key=lambda x: (-x[0], -x[1]))
                    
                    for _, _, h in candidates[:20]:
                        mapping[tv] = h
                        assigned_hosts.add(h)
                        
                        valid_step = True
                        for nb in t_adj[tv]:
                            if mapping[nb] != -1:
                                hn = mapping[nb]
                                e_key = (min(h, hn), max(h, hn))
                                if e_key in used_edges:
                                    valid_step = False
                                    break
                        
                        if valid_step:
                            if assign_next(t_idx + 1):
                                return True
                        
                        mapping[tv] = -1
                        assigned_hosts.remove(h)
                    
                    return False
                
                if assign_next(1):
                    return True
            
            return False
        
        # Run with a fixed seed for determinism in the "random" parts?
        # The current approach is deterministic given the heuristics.
        # To add diversity, we can shuffle the host_candidates order slightly?
        # Or just run it once. If it fails, we might need randomization.
        # Let's add a randomization factor: shuffle the candidates if scores are tied.
        
        rng = Random(611)
        
        # Modify assign_next to use rng for tie-breaking?
        # For now, let's just run the deterministic version.
        # If it finds a solution, great.
        
        success = backtrack_refactored(0)
        
        if success:
            return [m[:] if m else None for m in mappings], score
        else:
            return best_result, best_score

    res, sc = solve(order, None)
    
    # Fallback if the deterministic backtracking didn't find a full solution
    # Try a simple greedy with random restarts as a backup, but keep the best found.
    if sc < 190:
        # Simple greedy fallback
        used_edges = set()
        host_deg = [19] * 20
        greedy_result = [None] * 20
        greedy_score = 0
        
        for k_idx in range(19, -1, -1):
            k = k_idx + 1
            edges = trees[k_idx]
            if k == 1:
                greedy_result[k_idx] = [0]
                continue
            
            found = False
            for _ in range(200):
                verts = [0] * k
                # Random permutation
                perm = list(range(20))
                Random(611 + _).shuffle(perm)
                verts = perm[:k]
                
                valid = True
                for u, v in edges:
                    hu = verts[u]
                    hv = verts[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        valid = False
                        break
                
                if valid:
                    for u, v in edges:
                        hu = verts[u]
                        hv = verts[v]
                        e_key = (min(hu, hv), max(hu, hv))
                        used_edges.add(e_key)
                        host_deg[hu] -= 1
                        host_deg[hv] -= 1
                    greedy_result[k_idx] = verts
                    greedy_score += len(edges)
                    found = True
                    break
        
        if greedy_score > sc:
            res = greedy_result
            sc = greedy_score

    # Final type check
    final_result = []
    for r in res:
        if r is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in r])
            
    return final_result

# EVOLVE-BLOCK-END
