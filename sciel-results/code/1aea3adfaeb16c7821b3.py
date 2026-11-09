"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _is_c4_free(adj, n):
    """Check if graph with adjacency list adj on n vertices is C4-free."""
    for i in range(n):
        for j in range(i + 1, n):
            common = 0
            for nb in adj[i]:
                if nb in adj[j]:
                    common += 1
                    if common > 1:
                        return False
    return True

def _has_independent_set_k(adj, n, k, max_sets=None):
    """Check if there is an independent set of size k using backtracking."""
    if max_sets is not None:
        found_count = [0]
    
    def backtrack(candidates, current_size):
        if max_sets is not None:
            if found_count[0] >= max_sets:
                return True
        if current_size == k:
            if max_sets is not None:
                found_count[0] += 1
            return True
        if len(candidates) < k - current_size:
            return False
        for idx, v in enumerate(candidates):
            new_candidates = candidates[idx+1:]
            filtered = [u for u in new_candidates if u not in adj[v]]
            if backtrack(filtered, current_size + 1):
                return True
        return False
    
    all_vertices = list(range(n))
    return backtrack(all_vertices, 0)

def _construct_paley_like(n, prime_p):
    """Construct a graph on n vertices using a quadratic residue construction."""
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Fallback: 10 disjoint triangles
    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Try the Hoffman-Singleton based construction first (deterministic, fast)
    # This should give n=39 if valid
    
    hs_edges = _try_hoffman_singleton()
    if hs_edges is not None:
        # Verify it
        n_hs = 39
        adj = [set() for _ in range(n_hs)]
        for u, v in hs_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        if _is_c4_free(adj, n_hs):
            # Check alpha <= 10
            if not _has_independent_set_k(adj, n_hs, 11):
                return {'n': n_hs, 'edges': hs_edges}
    
    # If Hoffman-Singleton didn't work, try the search approach
    best_result = fallback()
    
    random.seed(42)
    
    # Strategy: Build C4-free graphs by adding edges carefully,
    # then check independence number.
    
    for n in range(43, 30, -1):
        if time.time() > deadline - 5:
            break
        
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            adj = [set() for _ in range(n)]
            edges = []
            
            possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(possible_edges)
            
            for (i, j) in possible_edges:
                if time.time() > deadline - 2:
                    break
                
                common = adj[i] & adj[j]
                if len(common) > 1:
                    continue
                
                can_add = True
                for k in adj[i]:
                    if k == j:
                        continue
                    if adj[j] & adj[k]:
                        can_add = False
                        break
                if not can_add:
                    continue
                for k in adj[j]:
                    if k == i:
                        continue
                    if adj[i] & adj[k]:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
            
            if not _is_c4_free(adj, n):
                continue
            
            has_indep_11 = _has_independent_set_k(adj, n, 11)
            if not has_indep_11:
                best_result = {'n': n, 'edges': edges}
                break
        
        if best_result['n'] == n:
            break
    
    return best_result

def _try_hoffman_singleton():
    """Try to construct a valid 39-vertex subgraph from Hoffman-Singleton."""
    n_total = 50
    edges = []
    
    # 1. Cycle edges within each group i
    for i in range(5):
        for j in range(10):
            u = i * 10 + j
            v = i * 10 + ((j + 1) % 10)
            if u > v:
                u, v = v, u
            edges.append([u, v])
            
    # 2. Matching edges between group i and i+1
    offsets = [1, 2, 4, 8, 6]
    for i in range(5):
        next_i = (i + 1) % 5
        d = offsets[i]
        for j in range(10):
            u = i * 10 + j
            v = next_i * 10 + ((j + d) % 10)
            if u > v:
                u, v = v, u
            edges.append([u, v])
            
    n_target = 39
    valid_edges = []
    for e in edges:
        if e[0] < n_target and e[1] < n_target:
            valid_edges.append(e)
            
    return valid_edges

# EVOLVE-BLOCK-END
