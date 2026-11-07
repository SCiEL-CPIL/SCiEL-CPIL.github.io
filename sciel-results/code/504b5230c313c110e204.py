"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n):
    dist = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = [i]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
    return dist

def _get_diameter(dist):
    max_d = 0
    for i in range(len(dist)):
        for j in range(len(dist)):
            if dist[i][j] > max_d:
                max_d = dist[i][j]
    return max_d

def _is_triangle_free(adj, u, v, n):
    neighbors_u = set(adj[u])
    neighbors_v = set(adj[v])
    common = neighbors_u.intersection(neighbors_v)
    if common:
        return False
    return True

def _construct_exact():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    # 1. Initialize base graph
    adj = [[] for _ in range(n)]
    base_edges_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # 2. Identify candidate edges
    # An edge (u, v) is a candidate if:
    # - It is not in the base graph
    # - Adding it does not create a triangle in the base graph (or current state)
    # Note: Triangle freeness is a property of the final set. 
    # If (u,v) and (v,w) are added, and (u,w) is in base, we have a triangle.
    # So we must check triangle freeness dynamically during search.
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges_set:
                continue
            candidates.append((u, v))
            
    # Sort candidates? Maybe by some heuristic to find solutions faster?
    # For exact search, order doesn't affect correctness, but can affect speed if we find a solution early.
    # However, we are searching for the MINIMUM number. 
    # We iterate k = 0, 1, 2... and check if any combination of size k works.
    # The first valid combination found for a given k is optimal.
    
    # Pre-compute initial distances to check if base is already valid (unlikely for path)
    dist = _bfs_distances(adj, n)
    current_diam = _get_diameter(dist)
    if current_diam <= 4:
        return []
        
    # Search for minimum k
    # Max possible k? 
    # Triangle-free graph on 24 vertices.
    # Max edges in triangle-free graph is floor(n^2/4) = floor(576/4) = 144.
    # Base has 23 edges.
    # So max added edges is 144 - 23 = 121.
    # But diameter 4 constraint is very tight.
    
    for k in range(1, 15): # Start with small k, likely solution is small (e.g., 5-10)
        if time.time() > deadline:
            # If we run out of time, we need to return something valid.
            # This is a risk. We should have a fallback greedy that runs first?
            # Or we can just return the best found so far? 
            # But we haven't found any valid one yet in the exact loop if k is small.
            # Let's add a quick greedy fallback that runs if exact search takes too long?
            # Or better: run greedy first to get an upper bound, then exact search up to that bound?
            # The prompt says "Minimize". If we find a valid solution with 10 edges, we only need to check k=1..9.
            
            # Let's restructure: 
            # 1. Run a fast greedy to get a valid solution (upper bound U).
            # 2. Run exact search for k = 1 to U-1.
            # 3. If exact search finds a valid solution with k < U, return it.
            # 4. If not, return the greedy solution.
            # This guarantees a valid return even if exact search is slow.
            
            # Wait, the current function is _construct_exact. 
            # I should integrate the greedy into the main construct() or here.
            # Let's assume the greedy from the seed is available or I re-implement a simple one here.
            
            # For now, if deadline is hit, I'll raise an error or return a known valid set?
            # No, I must return a list.
            # Let's implement a quick greedy inside this function as a fallback.
            
            return _construct_greedy_fallback(n, adj, base_edges_set)

        # Generate all combinations of k candidates
        # Number of combinations C(M, k) where M ~ 24*23/2 - 23 = 253.
        # C(253, 1) = 253
        # C(253, 2) = ~32k
        # C(253, 3) = ~2.7M
        # C(253, 4) = ~180M -> Too slow for Python in 80s if k=4.
        
        # We need pruning.
        # Pruning idea: 
        # If we pick an edge, it must reduce the diameter or at least not increase it.
        # But more importantly, we can check triangle freeness incrementally.
        
        # Let's use a recursive backtracking search for the k edges instead of itertools.combinations
        # which generates all at once. Backtracking allows pruning invalid branches early.
        
        found = _search_for_k_edges(adj, base_edges_set, candidates, k, n, deadline)
        if found is not None:
            return found
            
    # If we reach here, we didn't find a solution with k < 15?
    # Or deadline hit?
    # Return greedy fallback
    return _construct_greedy_fallback(n, adj, base_edges_set)

def _search_for_k_edges(adj, base_edges_set, candidates, k, n, deadline):
    """
    Backtracking search to find k edges that make graph valid.
    Returns list of edges if found, None otherwise.
    """
    # Deep copy adj and base_edges_set? 
    # Modifying and reverting is faster.
    
    current_edges = []
    
    def backtrack(start_idx, edges_selected, adj_current, edges_set_current):
        if time.time() > deadline:
            return None
            
        if len(edges_selected) == k:
            # Check if valid
            dist = _bfs_distances(adj_current, n)
            if _get_diameter(dist) <= 4:
                # Verify triangle free (should be guaranteed by construction if we check on add)
                return edges_selected[:]
            return None
            
        # Heuristic: If remaining candidates + selected < k, fail (not possible with this loop structure)
        
        # Pruning: 
        # We need to pick (k - len(edges_selected)) edges from candidates[start_idx:]
        
        for i in range(start_idx, len(candidates)):
            u, v = candidates[i]
            
            # Check if adding (u, v) creates a triangle
            # Triangle if there exists w such that (u, w) and (v, w) are in current graph
            # Current graph edges are in adj_current
            # We can check: for each neighbor w of u, is w a neighbor of v?
            # If yes, triangle.
            
            is_tri = False
            for w in adj_current[u]:
                if w in adj_current[v]: # O(1) if adj lists are sets, but they are lists.
                    # Wait, adj_current is list of lists. 
                    # Converting to sets every time is expensive.
                    # Let's maintain a set of edges for O(1) lookup?
                    # Or just use sets for adjacency.
                    
                    # Let's switch adj_current to list of sets for efficiency in triangle check.
                    # But BFS needs lists. We can convert back or just use lists and accept O(deg) check.
                    # deg is small.
                    pass
                    
            # Actually, let's optimize the triangle check.
            # neighbors_u = set(adj_current[u])
            # neighbors_v = set(adj_current[v])
            # if neighbors_u.intersection(neighbors_v):
            #    continue
            
            # This creates sets every iteration. 
            # Alternative: maintain a global set of edges?
            # If (u,v) added, check if any common neighbor.
            
            # Let's try a different representation for the search state:
            # adj_sets: list of sets
            # This makes triangle check O(deg(u) + deg(v)) and add/remove O(1).
            # BFS can iterate over sets.
            
            # I'll refactor the search to use adj_sets.
            
            pass
            
        return None

    # Refactored search with adj_sets
    adj_sets = [set(neighs) for neighs in adj]
    # Note: adj passed in is the base graph.
    
    def backtrack(start_idx, edges_selected, adj_sets_current):
        if time.time() > deadline:
            return None
            
        if len(edges_selected) == k:
            # Convert sets to lists for BFS or just use sets in BFS
            # BFS:
            dist = [[-1] * n for _ in range(n)]
            for i in range(n):
                dist[i][i] = 0
                queue = [i]
                while queue:
                    u = queue.pop(0)
                    for v in adj_sets_current[u]:
                        if dist[i][v] == -1:
                            dist[i][v] = dist[i][u] + 1
                            queue.append(v)
                            
            max_d = 0
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > max_d:
                        max_d = dist[i][j]
                        
            if max_d <= 4:
                return edges_selected[:]
            return None
            
        # Pruning: 
        # If the current diameter is already <= 4, we don't need more edges? 
        # No, we are fixing k. We must pick exactly k.
        # But if we pick k edges and diameter is <= 4, it's valid.
        # If diameter > 4, we continue.
        
        # Can we prune if diameter is too large to be fixed by remaining edges?
        # Hard to estimate.
        
        for i in range(start_idx, len(candidates)):
            u, v = candidates[i]
            
            # Triangle check
            # Common neighbor?
            # If any w in adj_sets_current[u] is also in adj_sets_current[v], triangle.
            # Optimization: iterate over smaller set
            if len(adj_sets_current[u]) < len(adj_sets_current[v]):
                common = any(w in adj_sets_current[v] for w in adj_sets_current[u])
            else:
                common = any(w in adj_sets_current[u] for w in adj_sets_current[v])
                
            if common:
                continue
                
            # Add edge
            adj_sets_current[u].add(v)
            adj_sets_current[v].add(u)
            edges_selected.append((u, v))
            
            # Recurse
            res = backtrack(i + 1, edges_selected, adj_sets_current)
            if res is not None:
                return res
                
            # Backtrack
            edges_selected.pop()
            adj_sets_current[u].remove(v)
            adj_sets_current[v].remove(u)
            
        return None

    return backtrack(0, [], adj_sets)

def _construct_greedy_fallback(n, adj, base_edges_set):
    # Simple greedy to ensure we always return a valid solution
    # This is essentially the seed's logic but simplified for speed
    adj_copy = [list(neighs) for neighs in adj]
    added = []
    
    dist = _bfs_distances(adj_copy, n)
    current_diam = _get_diameter(dist)
    
    max_iters = 50
    iter_count = 0
    while current_diam > 4 and iter_count < max_iters:
        iter_count += 1
        max_d = -1
        best_u, best_v = -1, -1
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
                    best_u, best_v = i, j
        
        if best_u == -1:
            break
            
        if (best_u, best_v) in base_edges_set:
            continue
            
        # Check triangle
        is_tri = False
        for w in adj_copy[best_u]:
            if w in adj_copy[best_v]:
                is_tri = True
                break
                
        if not is_tri:
            adj_copy[best_u].append(best_v)
            adj_copy[best_v].append(best_u)
            base_edges_set.add((best_u, best_v))
            base_edges_set.add((best_v, best_u))
            added.append([best_u, best_v])
            dist = _bfs_distances(adj_copy, n)
            current_diam = _get_diameter(dist)
            
    return added

def _verify_solution(added_edges):
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    for u, v in added_edges:
        if (u, v) in base_edges:
            return False
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Check triangle free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
                    
    # Check connected and diameter
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return False
                
    diam = _get_diameter(dist)
    if diam > 4:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # 1. Get a valid solution quickly (Upper Bound)
    # We use a greedy approach to get a baseline valid solution.
    # This ensures we have something to return even if exact search fails/times out.
    
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # Simple Greedy to get initial valid solution
    greedy_sol = _construct_greedy_fallback(n, adj, base_edges_set)
    
    if not _verify_solution(greedy_sol):
        # If greedy fails (shouldn't happen), try a very dense valid construction?
        # Or just return empty? No, must be valid.
        # Let's assume greedy works for path graph.
        pass
        
    # 2. Try to find a better solution with Exact Search
    # We search for k = 1 to len(greedy_sol) - 1
    # If we find a valid solution with k edges, it's better.
    
    best_sol = greedy_sol
    best_k = len(greedy_sol)
    
    # Candidates for exact search
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges_set:
                candidates.append((u, v))
                
    # Re-initialize adj for search
    adj_search = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj_search[u].append(v)
        adj_search[v].append(u)
        
    # We need to pass the base adj to the search function
    # The search function creates its own adj_sets from the passed adj.
    
    for k in range(1, best_k):
        if time.time() > deadline:
            break
            
        # Run exact search for k
        # Note: _search_for_k_edges expects adj (list of lists) and base_edges_set
        # It creates adj_sets internally.
        
        # Optimization: If k is large, combinations explode.
        # But we stop at best_k - 1.
        # If greedy found 10 edges, we check 1..9.
        # C(253, 9) is huge. 
        # We rely on pruning (triangle check) and early termination (diameter check).
        # However, if no solution exists for small k, we might explore many branches.
        
        # To be safe, we can limit the search depth or time per k.
        # The deadline check inside backtrack handles this.
        
        res = _search_for_k_edges(adj_search, base_edges_set, candidates, k, n, deadline)
        if res is not None:
            # Convert tuples to lists for return
            best_sol = [list(e) for e in res]
            best_k = k
            # Since we search in increasing k, the first valid one is optimal.
            # We can break immediately!
            break
            
    # 3. Final verification
    if _verify_solution(best_sol):
        res = []
        for u, v in best_sol:
            res.append([int(u), int(v)])
        return res
    else:
        # Fallback: return greedy if it was valid, else empty (invalid)
        # But greedy should be valid.
        if _verify_solution(greedy_sol):
            res = []
            for u, v in greedy_sol:
                res.append([int(u), int(v)])
            return res
        else:
            return []

# EVOLVE-BLOCK-END
