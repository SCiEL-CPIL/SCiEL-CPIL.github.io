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
    # Safety margin: 150s limit, aim to finish by 140s
    deadline = start_time + 135.0
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Pre-compute all edges in K_n
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Build adjacency for each tree
    tree_adj = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Order: Largest first (most constrained)
    order = list(range(n - 1, -1, -1))
    
    def check_time():
        return time.time() < deadline

    def embed_tree_csp(tree_idx, available_edges, rng):
        """
        Embed tree tree_idx into K_n using available_edges.
        Uses CSP with MRV and LCV heuristics.
        Returns a list of length k+1 where mapping[i] is the host vertex for tree vertex i, or None.
        """
        k = tree_idx + 1
        if k == 1:
            return [0]
        
        adj = tree_adj[tree_idx]
        
        # We need to assign each tree vertex v in [0, k-1] to a host vertex g in [0, n-1].
        # Constraints:
        # 1. All g must be distinct.
        # 2. For each edge (u, v) in tree, (mapping[u], mapping[v]) must be in available_edges.
        
        # State:
        # assignment: list of -1 or host vertex
        # available_edges: set of (min, max) tuples
        
        # To make it faster, we can pre-calculate the "degree" of each host vertex in the available graph
        # and use that for heuristics.
        
        def get_available_neighbors(g):
            """Return set of host vertices h such that (g, h) is in available_edges."""
            nbrs = set()
            for h in range(n):
                if h == g:
                    continue
                e = (min(g, h), max(g, h))
                if e in available_edges:
                    nbrs.add(h)
            return nbrs

        # We will use a recursive backtracking search.
        # To avoid deep recursion issues and allow time checks, we'll implement it iteratively or with a depth limit.
        # Given k <= 20, recursion depth 20 is fine.
        
        mapping = [-1] * k
        used_hosts = set()
        
        # Pre-compute initial domains? 
        # Domain of v is all hosts h such that h is not used and ... 
        # Actually, the constraint is local: if v is assigned to h, then for every neighbor u of v, 
        # the edge (h, mapping[u]) must be available.
        
        # Heuristic: Order tree vertices by degree in the tree (highest first).
        # This forces high-degree nodes to be placed in high-degree regions of the available graph.
        tree_vertices = list(range(k))
        tree_vertices.sort(key=lambda v: len(adj[v]), reverse=True)
        
        # We need to track which tree vertices are assigned.
        assigned = [False] * k
        
        def dfs(idx):
            if not check_time():
                return False
            
            if idx == k:
                return True
            
            v = tree_vertices[idx]
            
            # Determine valid host vertices for v
            # A host h is valid if:
            # 1. h is not in used_hosts
            # 2. For every already assigned neighbor u of v, (h, mapping[u]) is in available_edges
            
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                valid = True
                for u in adj[v]:
                    if assigned[u]:
                        e = (min(h, mapping[u]), max(h, mapping[u]))
                        if e not in available_edges:
                            valid = False
                            break
                if valid:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            # Heuristic: Least Constraining Value (LCV)
            # Pick h that leaves the most options for unassigned neighbors of v.
            # Score(h) = sum over unassigned neighbors w of v of |{x in candidates_for_w | (h, x) in available}|
            # This is expensive to compute exactly. Approximation:
            # Pick h with the highest degree in the available graph restricted to unused hosts?
            # Or simply random shuffle to break ties and allow restarts to explore different paths.
            
            # Let's use a simple score: number of available edges from h to unused hosts.
            # This ensures we don't isolate the node.
            def score_host(h):
                cnt = 0
                for x in range(n):
                    if x in used_hosts:
                        continue
                    e = (min(h, x), max(h, x))
                    if e in available_edges:
                        cnt += 1
                return cnt
            
            candidates.sort(key=score_host, reverse=True)
            
            # To allow different search paths in restarts, we can shuffle the top candidates slightly?
            # Or just rely on the deterministic order + different initial random seeds for the whole process?
            # The current strategy is deterministic. Let's add a bit of randomness in tie-breaking or candidate order.
            # Actually, for CSP, deterministic is often better if heuristics are good.
            # But to escape local minima, randomization helps.
            # Let's shuffle candidates that have the same score? Or just pick randomly from top 3?
            
            # For now, let's stick to deterministic LCV. If it fails, we restart the whole packing.
            
            for h in candidates:
                mapping[v] = h
                used_hosts.add(h)
                assigned[v] = True
                
                # Tentatively remove edges used by v's assigned neighbors?
                # No, we only remove edges when the tree is fully embedded and verified.
                # Wait, if we don't remove edges, we might reuse an edge for another tree?
                # The `available_edges` set is passed by reference (or copy) to the function.
                # We must ensure that if we assign v to h, and u to g, the edge (g,h) is removed from available_edges
                # for the duration of this subtree search, so that no other tree vertex w can use (g,h)
                # unless w is also connected to both? No, tree edges are unique.
                # So, if we assign (v, h) and (u, g), we MUST remove (g, h) from available_edges
                # for the rest of the search for THIS tree, to ensure edge-disjointness within the tree.
                
                # Let's modify the state to track edges used within the current tree embedding.
                # This requires passing a copy of available_edges or modifying it and backtracking.
                
                # Backtracking approach:
                # 1. Save state of available_edges (or just the edges we removed).
                # 2. Remove edges (h, mapping[u]) for all assigned neighbors u of v.
                # 3. Recurse.
                # 4. Restore edges.
                
                removed_edges = []
                for u in adj[v]:
                    if assigned[u]:
                        e = (min(h, mapping[u]), max(h, mapping[u]))
                        if e in available_edges:
                            available_edges.remove(e)
                            removed_edges.append(e)
                        else:
                            # This shouldn't happen if we checked validity, but just in case
                            pass
                
                if dfs(idx + 1):
                    return True
                
                # Backtrack
                for e in removed_edges:
                    available_edges.add(e)
                
                mapping[v] = -1
                used_hosts.remove(h)
                assigned[v] = False
            
            return False
        
        # To allow different search paths, we can randomize the order of tree_vertices?
        # No, MRV/LCV depends on the structure.
        # Let's just run it. If it fails, the outer loop will restart.
        
        return mapping if dfs(0) else None

    # Main packing loop with restarts
    # We try multiple times. Each time, we might shuffle the order of trees slightly?
    # No, largest first is best.
    # We can randomize the tie-breaking in LCV or the initial candidate order.
    
    best_solution = [None] * n
    best_value = 0
    
    # We will do a few attempts.
    # Since the problem is deterministic, if one attempt fails, others might fail too unless we change search strategy.
    # However, the LCV heuristic is deterministic.
    # To make restarts useful, we should randomize the candidate order in LCV.
    
    for attempt in range(20):
        if not check_time():
            break
            
        available = set(all_edges)
        solution = [None] * n
        success = True
        
        # Create a random instance for tie-breaking
        rng = Random(42 + attempt)
        
        for t_idx in order:
            if not check_time():
                success = False
                break
                
            k = t_idx + 1
            if k == 1:
                solution[t_idx] = [0]
                continue
            
            # We need to modify embed_tree_csp to accept rng for randomization
            # Let's wrap it or pass rng.
            
            # For this attempt, we will use a slightly randomized LCV.
            # We'll pass a flag or just rely on the fact that if we change the "random seed" 
            # used inside the sort, we get different paths.
            
            # Let's create a local version of embed that uses rng.
            def embed_with_rng(tree_idx, avail, rng):
                k = tree_idx + 1
                if k == 1:
                    return [0]
                adj = tree_adj[tree_idx]
                mapping = [-1] * k
                used_hosts = set()
                assigned = [False] * k
                
                tree_vertices = list(range(k))
                tree_vertices.sort(key=lambda v: len(adj[v]), reverse=True)
                
                def dfs(idx):
                    if not check_time():
                        return False
                    if idx == k:
                        return True
                    
                    v = tree_vertices[idx]
                    candidates = []
                    for h in range(n):
                        if h in used_hosts:
                            continue
                        valid = True
                        for u in adj[v]:
                            if assigned[u]:
                                e = (min(h, mapping[u]), max(h, mapping[u]))
                                if e not in avail:
                                    valid = False
                                    break
                        if valid:
                            candidates.append(h)
                    
                    if not candidates:
                        return False
                    
                    # Score and shuffle
                    scored = []
                    for h in candidates:
                        cnt = 0
                        for x in range(n):
                            if x in used_hosts:
                                continue
                            e = (min(h, x), max(h, x))
                            if e in avail:
                                cnt += 1
                        scored.append((cnt, h))
                    
                    scored.sort(key=lambda x: x[0], reverse=True)
                    
                    # Randomize order of candidates with same score?
                    # Or just pick randomly from the top half?
                    # Let's shuffle the list of (score, host) and then sort by score, 
                    # but Python's sort is stable, so ties keep original order.
                    # To randomize ties, we can add a random key.
                    rng.shuffle(scored) # This shuffles the list before sorting? No.
                    
                    # Better:
                    # Add random tie-breaker
                    scored_with_rand = [(s, rng.random(), h) for s, h in scored]
                    scored_with_rand.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    
                    for _, _, h in scored_with_rand:
                        mapping[v] = h
                        used_hosts.add(h)
                        assigned[v] = True
                        
                        removed_edges = []
                        for u in adj[v]:
                            if assigned[u]:
                                e = (min(h, mapping[u]), max(h, mapping[u]))
                                if e in avail:
                                    avail.remove(e)
                                    removed_edges.append(e)
                        
                        if dfs(idx + 1):
                            return True
                        
                        for e in removed_edges:
                            avail.add(e)
                        
                        mapping[v] = -1
                        used_hosts.remove(h)
                        assigned[v] = False
                    
                    return False
                
                return mapping if dfs(0) else None

            mapping = embed_with_rng(t_idx, available, rng)
            
            if mapping is None:
                success = False
                break
            
            # Remove used edges from global available set
            for u, v in trees[t_idx]:
                e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                available.discard(e)
            
            solution[t_idx] = mapping
        
        if success:
            # Verify solution
            valid = True
            used_edges = set()
            for t_idx in range(n):
                if solution[t_idx] is None:
                    valid = False
                    break
                for u, v in trees[t_idx]:
                    e = (min(solution[t_idx][u], solution[t_idx][v]), max(solution[t_idx][u], solution[t_idx][v]))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
            
            if valid:
                # Calculate value
                val = sum(len(t) for t in trees)
                # Convert to list of lists of ints
                final_sol = []
                for t_idx in range(n):
                    if solution[t_idx] is not None:
                        final_sol.append([int(x) for x in solution[t_idx]])
                    else:
                        final_sol.append(None)
                return final_sol
        
        # If not successful, keep the best partial solution?
        # The task says "returning 20 entries... Entry k-1 is None or a list".
        # "The raw value is the total number of embedded tree edges".
        # So we should return the best partial solution found.
        
        val = 0
        for t_idx in range(n):
            if solution[t_idx] is not None:
                val += len(trees[t_idx])
        
        if val > best_value:
            best_value = val
            best_solution = [None if s is None else [int(x) for x in s] for s in solution]
    
    return best_solution
