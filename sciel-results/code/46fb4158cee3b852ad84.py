"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute degrees for heuristics
    degrees = [len(adj[i]) for i in range(n)]
    
    # --- Helper: Find a path from u to v avoiding forbidden vertices ---
    def find_path(u, v, forbidden):
        # BFS
        parent = [-1] * n
        parent[u] = u
        queue = deque([u])
        
        while queue:
            cur = queue.popleft()
            if cur == v:
                # Reconstruct path
                path = []
                c = v
                while c != u:
                    path.append(c)
                    c = parent[c]
                path.append(u)
                path.reverse()
                return path
            for nb in adj[cur]:
                if parent[nb] == -1 and nb not in forbidden:
                    parent[nb] = cur
                    queue.append(nb)
        return None

    # --- Helper: Check if a set of branches is realizable and return paths ---
    # Returns (True, paths_dict) or (False, None)
    def check_and_get_paths(B):
        if len(B) < 2:
            return False, None
        
        B_set = set(B)
        pairs = list(itertools.combinations(B_set, 2))
        
        # Heuristic: Sort pairs by "tightness" (shortest path length in G - B)
        # But this is dynamic. We will try multiple random permutations to overcome ordering issues.
        
        # Optimization: If the number of pairs is small, we can be more exhaustive with random restarts.
        # For k=8, C(8,2)=28 paths.
        
        num_restarts = 20
        for _ in range(num_restarts):
            used_interiors = set()
            paths = {}
            pairs_copy = list(pairs)
            random.shuffle(pairs_copy)
            
            success = True
            for u, v in pairs_copy:
                # Forbidden: all other branches + all used interiors
                forbidden = (B_set - {u, v}) | used_interiors
                
                path = find_path(u, v, forbidden)
                if not path:
                    success = False
                    break
                else:
                    paths[(u, v)] = path
                    for node in path[1:-1]:
                        used_interiors.add(node)
                        
            if success:
                return True, paths
        
        return False, None

    # --- Step 1: Find Max Clique (Bron-Kerbosch) ---
    max_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique
        if time.time() > deadline - 10.0: # Leave time for search
            return
        
        if len(P) == 0 and len(X) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        if len(P) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        union_p_x = P.union(X)
        if not union_p_x:
            return
        
        best_pivot = None
        max_uncovered = -1
        for u in union_p_x:
            uncovered = len(P.difference(adj[u]))
            if uncovered > max_uncovered:
                max_uncovered = uncovered
                best_pivot = u
        
        if best_pivot is None:
            return
        
        for v in list(P.difference(adj[best_pivot])):
            bron_kerbosch(R.union([v]), P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)
            
    all_vertices = set(range(n))
    bron_kerbosch(set(), all_vertices, set())
    
    if not max_clique:
        max_clique = [0, 1]
        
    current_B = set(max_clique)
    best_B = set(current_B)
    best_paths = None
    best_score = len(best_B)
    
    # Initial check
    valid, paths = check_and_get_paths(current_B)
    if valid:
        best_paths = paths
    else:
        # Fallback for clique: direct edges
        best_paths = {}
        for u, v in itertools.combinations(list(best_B), 2):
            best_paths[(u, v)] = [u, v]
        # Note: This fallback is only valid if they are adjacent.
        # But Bron-Kerbosch finds cliques, so they are adjacent.
        # However, check_and_get_paths might fail due to random shuffle luck? Unlikely for clique.
        # If it fails, we force direct edges.
        # Actually, if check_and_get_paths fails for a clique, something is wrong or time ran out.
        # Let's assume it works or use direct edges.

    # --- Step 2: Local Search / Greedy Expansion ---
    # We try to add vertices to the set.
    # Candidates are sorted by degree.
    
    # To avoid getting stuck, we also allow "swaps" (remove one, add one) if adding fails.
    
    # Sort all vertices by degree descending
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # We will iterate through potential additions.
    # Since n is small, we can be thorough.
    
    # Strategy:
    # 1. Try to add any vertex not in B.
    # 2. If we can't add any, try to swap a low-degree vertex in B with a high-degree vertex not in B.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Generate list of candidates to try adding
        # We prioritize high degree
        candidates = [v for v in vertices_by_degree if v not in current_B]
        
        added_any = False
        for v in candidates:
            if time.time() > deadline - 5.0:
                break
            new_B = current_B | {v}
            valid, paths = check_and_get_paths(new_B)
            if valid:
                current_B = new_B
                best_B = set(current_B)
                best_paths = paths
                best_score = len(current_B)
                improved = True
                added_any = True
                break # Restart loop with new B
        
        if not added_any:
            # Try swaps
            # Pick a random vertex from B to remove, and a high-degree vertex not in B to add
            if len(current_B) > 2:
                # Sort B by degree ascending (remove weakest)
                b_sorted = sorted(current_B, key=lambda x: degrees[x])
                # Candidates not in B, sorted by degree descending
                candidates_not_in_b = [v for v in vertices_by_degree if v not in current_B]
                
                # Try a few swaps
                for remove_v in b_sorted[:3]: # Try removing the 3 weakest
                    for add_v in candidates_not_in_b[:5]: # Try adding the 5 strongest
                        if time.time() > deadline - 5.0:
                            break
                        swap_B = (current_B - {remove_v}) | {add_v}
                        valid, paths = check_and_get_paths(swap_B)
                        if valid:
                            # Accept swap if it doesn't reduce size (it keeps size same)
                            # But we want to eventually grow.
                            # A swap might open up the possibility to add another vertex later.
                            # So we accept neutral swaps with some probability or if they improve "potential"?
                            # For simplicity, accept if valid.
                            current_B = swap_B
                            # Don't update best_score unless size increases, but update current state.
                            # We might have lost a "best" configuration if best was larger?
                            # No, swap keeps size same.
                            # But maybe the new configuration allows growth.
                            improved = True
                            break
                    if improved:
                        break
                        
    # --- Step 3: Final Polish / Verification ---
    # Ensure best_B is actually valid with the paths we have.
    # If best_paths is None or invalid, re-check.
    
    if best_paths is None:
        valid, paths = check_and_get_paths(best_B)
        if valid:
            best_paths = paths
        else:
            # Fallback to clique
            best_B = set(max_clique)
            best_paths = {}
            for u, v in itertools.combinations(list(best_B), 2):
                best_paths[(u, v)] = [u, v]
            
    # Construct output
    branches = [int(v) for v in best_B]
    # Sort branches to ensure deterministic order for pairs?
    # The problem says "unordered pair", but the list of paths must correspond to the branches.
    # Usually, we just need a list of paths.
    # Let's sort branches for consistency.
    branches.sort()
    
    paths_list = []
    for u, v in itertools.combinations(branches, 2):
        # Find the path
        key = (u, v)
        if key not in best_paths:
            key = (v, u)
        if key in best_paths:
            p = best_paths[key]
        else:
            # Should not happen
            p = [u, v]
        paths_list.append([int(x) for x in p])
        
    return {'branches': branches, 'paths': paths_list}

# EVOLVE-BLOCK-END
