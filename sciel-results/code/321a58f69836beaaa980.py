"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import time
import random
import itertools
import numpy as np

def construct():
    start = time.time()
    deadline = start + 145

    # Try to find a valid graph on n=20 using local search
    # If successful, try to extend to n=21
    
    best_graph = None
    best_n = 0

    # Phase 1: Local Search on n=20
    # Objective: Minimize number of regular induced subgraphs of size 6 and 7
    # We focus on size 6 first as it's the most common violation
    
    n_target = 20
    adj = None
    edges = None
    
    # Initialize random graph
    rng = random.Random(42)
    adj = np.zeros((n_target, n_target), dtype=np.uint8)
    for i in range(n_target):
        for j in range(i+1, n_target):
            if rng.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1

    # Precompute all subsets of size 6 and 7
    subsets_6 = list(itertools.combinations(range(n_target), 6))
    subsets_7 = list(itertools.combinations(range(n_target), 7))
    
    # Map subset to index for fast lookup? 
    # Or just maintain a list of violating subsets.
    
    def get_violations(adj, subsets):
        viols = []
        for s in subsets:
            idx = np.array(s)
            sub_adj = adj[np.ix_(idx, idx)]
            degs = sub_adj.sum(axis=1)
            if np.all(degs == degs[0]):
                viols.append(s)
        return viols

    # Initial check
    viols_6 = get_violations(adj, subsets_6)
    if not viols_6:
        viols_7 = get_violations(adj, subsets_7)
        if not viols_7:
             # Check larger sizes
             if _is_valid_full(adj, n_target):
                 best_graph = _adj_to_edges(adj, n_target)
                 best_n = n_target

    # Local Search
    if best_n < n_target:
        for step in range(5000):
            if time.time() > deadline - 20:
                break
            
            # Get current violations
            if viols_6:
                # Pick a random violation
                s = rng.choice(viols_6)
                # Flip a random edge in this subset
                u, v = rng.sample(s, 2)
                # Flip edge
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
                
                # Update violations
                # Only subsets containing both u and v are affected
                # We can just recompute viols_6 from scratch? 
                # 38k subsets * 0.001s = 0.04s. 5000 steps = 200s. Too slow.
                # We need incremental update.
                
                # Let's just recompute if the list is small?
                # Or use a faster check.
                
                # Alternative: Just recompute viols_6 using a fast bitset check.
                viols_6 = _get_violations_fast(adj, n_target, 6)
                if not viols_6:
                    viols_7 = _get_violations_fast(adj, n_target, 7)
                    if not viols_7:
                        if _is_valid_full(adj, n_target):
                            best_graph = _adj_to_edges(adj, n_target)
                            best_n = n_target
                            break
            else:
                # No size 6 violations, check 7
                viols_7 = _get_violations_fast(adj, n_target, 7)
                if not viols_7:
                     if _is_valid_full(adj, n_target):
                         best_graph = _adj_to_edges(adj, n_target)
                         best_n = n_target
                         break
                else:
                    s = rng.choice(viols_7)
                    u, v = rng.sample(s, 2)
                    adj[u, v] = 1 - adj[u, v]
                    adj[v, u] = adj[u, v]
                    viols_6 = _get_violations_fast(adj, n_target, 6)

    # Phase 2: Try to extend to 21
    if best_n == 20 and time.time() < deadline - 10:
        result = _try_extend_to_21(best_graph, deadline - 5)
        if result:
            return result

    if best_graph:
        return {'n': best_n, 'edges': best_graph}
    
    return {'n': 5, 'edges': []}

def _get_violations_fast(adj, n, size):
    """Fast check for regular induced subgraphs of specific size."""
    viols = []
    # Use bitsets for speed
    adj_bits = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i, j]:
                mask |= (1 << j)
        adj_bits[i] = mask
        
    for subset in itertools.combinations(range(n), size):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        # Check regularity
        degs = []
        for v in subset:
            deg = (adj_bits[v] & mask).bit_count()
            degs.append(deg)
        
        if all(d == degs[0] for d in degs):
            viols.append(subset)
            
    return viols

def _is_valid_full(adj, n):
    """Check all sizes >= 6."""
    for size in range(6, n + 1):
        if _get_violations_fast(adj, n, size):
            return False
    return True

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _try_extend_to_21(base_edges, deadline):
    """Try to extend n=20 graph to n=21."""
    n_old = 20
    n_new = 21
    adj_old = np.zeros((n_old, n_old), dtype=np.uint8)
    for u, v in base_edges:
        adj_old[u, v] = 1
        adj_old[v, u] = 1
        
    rng = random.Random(123)
    
    # Try random connections for new vertex
    for _ in range(1000):
        if time.time() > deadline:
            break
        new_edges_mask = 0
        for i in range(n_old):
            if rng.random() < 0.5:
                new_edges_mask |= (1 << i)
                
        # Check if adding vertex n_old with these edges creates violations
        # We only need to check subsets containing the new vertex
        # Subsets of size k containing new vertex correspond to subsets of size k-1 in old graph
        # But we need to check regularity of the induced subgraph on {new} U T
        
        valid = True
        # Check sizes 6 to 21
        for size in range(6, n_new + 1):
            t_size = size - 1
            if t_size < 0: continue
            if t_size > n_old: continue
            
            for T in itertools.combinations(range(n_old), t_size):
                # Build mask for T
                t_mask = 0
                for v in T:
                    t_mask |= (1 << v)
                
                # Degree of new vertex in T U {new}
                deg_new = (new_edges_mask & t_mask).bit_count()
                
                # Check if all v in T have degree deg_new in T U {new}
                ok = True
                for v in T:
                    # Degree of v in T (within old graph)
                    deg_v_t = 0
                    for u in T:
                        if u != v and adj_old[v, u]:
                            deg_v_t += 1
                    # Edge to new vertex?
                    if (new_edges_mask >> v) & 1:
                        deg_v_t += 1
                    
                    if deg_v_t != deg_new:
                        ok = False
                        break
                if not ok:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            # Construct full edge list
            edges = list(base_edges)
            for i in range(n_old):
                if (new_edges_mask >> i) & 1:
                    edges.append([n_old, i])
            return {'n': n_new, 'edges': edges}
            
    return None
