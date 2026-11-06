import time
import random

_START = time.time()
_DEADLINE = 140.0  # Safety margin under 150s

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check for K4 using bitmasks. Returns None if no K4 found, else the K4 tuple."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            # Common neighbors of i and j, greater than j
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            # Check if any two common neighbors are connected
            # Iterate through common neighbors
            c = common
            while c:
                lsb_a = c & -c
                a = lsb_a.bit_length() - 1
                c ^= lsb_a
                # Check if a is connected to any other common neighbor
                # Other common neighbors: common & ~((1 << (a+1)) - 1)
                rest = common & ~((1 << (a + 1)) - 1)
                if rest & adj[a]:
                    # Find the specific b
                    r = rest & adj[a]
                    if r:
                        lsb_b = r & -r
                        b = lsb_b.bit_length() - 1
                        return (i, j, a, b)
    return None

def _find_ind_6(adj, n):
    """Find an independent set of size 6. Returns list of vertices or None."""
    all_mask = (1 << n) - 1
    # comp[i] is bitmask of non-neighbors of i (excluding i)
    comp = [all_mask & ~(adj[i] | (1 << i)) for i in range(n)]
    
    found = [None]
    
    def backtrack(candidates, current):
        if found[0] is not None:
            return
        if len(current) == 6:
            found[0] = current[:]
            return
        if candidates.bit_count() < 6 - len(current):
            return
        
        # Heuristic: pick vertex with max degree in induced subgraph of candidates
        # to prune faster? Or min degree?
        # For independent set, we want vertices that are NOT connected.
        # In the complement graph (where edges exist between non-neighbors), we are looking for a clique.
        # So we are doing Bron-Kerbosch on the complement.
        # Pivot selection: choose u with max degree in P (candidates) in the complement graph.
        # Degree in complement = (candidates & comp[u]).bit_count()
        
        if candidates == 0:
            return
            
        # Simple heuristic: pick first vertex, or one with low "complement degree" (high original degree)
        # Actually, for finding ANY independent set of size 6, a greedy approach with backtracking is fine.
        
        # Select pivot u from candidates to split
        # We want to iterate over neighbors of u in the complement (i.e., non-neighbors of u in G)
        # Candidates to branch on: candidates & comp[u]
        # If we pick u, we must include it in the independent set? No, Bron-Kerbosch logic:
        # R = current independent set. P = candidates. X = excluded.
        # We iterate over v in P.
        # If we include v, new P = P & comp[v].
        
        # To speed up, we can use a pivot.
        # Let u be a vertex in P (or X) with max degree in P (in complement).
        # Then we only need to consider v in P such that v is NOT connected to u in complement.
        # i.e., v is connected to u in G.
        # So v in P & ~comp[u] = P & adj[u].
        
        # If P & ~comp[u] is empty, then u is connected to all in P in complement?
        # No, if P & ~comp[u] is empty, it means all v in P are connected to u in complement.
        # Then no clique can be formed by adding any v in P to R?
        
        # Let's stick to a simpler recursive search for n=35, it's small.
        # Optimization: Order vertices by degree.
        
        # Pick a vertex v from candidates
        # To maximize pruning, pick v with highest degree in the induced subgraph of candidates in the ORIGINAL graph?
        # High degree in G means few options for independent set.
        
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            # Degree of v in induced subgraph candidates
            deg = (candidates & adj[v]).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            c ^= lsb
            
        if best_v == -1:
            return
        
        v = best_v
        # Branch 1: Include v
        # New candidates must be non-neighbors of v
        new_cand = candidates & comp[v]
        backtrack(new_cand, current + [v])
        
        if found[0] is not None:
            return
            
        # Branch 2: Exclude v
        new_cand2 = candidates & ~(1 << v)
        backtrack(new_cand2, current)

    backtrack(all_mask, [])
    return found[0]

def _is_safe_edge(adj, n, u, v):
    """Check if adding edge (u,v) creates a K4."""
    if (adj[u] >> v) & 1:
        return True # Already exists
    
    # Common neighbors
    common = adj[u] & adj[v]
    # If any two common neighbors are connected, adding (u,v) creates a K4
    c = common
    while c:
        lsb_a = c & -c
        a = lsb_a.bit_length() - 1
        c ^= lsb_a
        rest = common & ~((1 << (a + 1)) - 1)
        if rest & adj[a]:
            return False
    return True

def _repair_graph(adj, n, rng, time_limit):
    """
    Iteratively repair graph to be K4-free and alpha < 6.
    Returns adj if successful, None if timeout.
    """
    start = time.time()
    adj = list(adj)
    
    while time.time() - start < time_limit:
        # 1. Check K4
        k4 = _has_k4(adj, n)
        if k4:
            i, j, a, b = k4
            # Remove one random edge from the K4
            edges = [(i, j), (i, a), (i, b), (j, a), (j, b), (a, b)]
            u, v = rng.choice(edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        # 2. Check Ind 6
        ind6 = _find_ind_6(adj, n)
        if ind6:
            # Find a safe edge to add within ind6
            # Try all pairs in ind6, pick a safe one
            safe_pairs = []
            for i in range(6):
                for j in range(i + 1, 6):
                    u, v = ind6[i], ind6[j]
                    if _is_safe_edge(adj, n, u, v):
                        safe_pairs.append((u, v))
            
            if safe_pairs:
                u, v = rng.choice(safe_pairs)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                # No safe edge found in this independent set.
                # This means the structure is rigid.
                # Try adding an edge to a random non-neighbor pair in the whole graph?
                # Or break the symmetry by removing an edge elsewhere?
                # For now, try to add any edge in ind6 and hope the K4 check fails?
                # No, that would break K4-free property.
                # If no safe edge, we are stuck.
                # Try removing a random edge from the graph to change structure?
                # This is a "kick".
                u = rng.randrange(n)
                v = rng.randrange(n)
                if u != v and (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # If no edges to remove, or random pick failed, break
                    break
            continue
        
        # 3. Valid!
        return adj
        
    return None

def _random_adj(n, p, rng):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    best_n = 0
    best_edges = []
    
    # We want to maximize n. Try 35 down to 20.
    # 35 is the known record.
    
    for n in range(35, 19, -1):
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Time allocation per n
        if n >= 35:
            time_slice = 40.0
        elif n >= 34:
            time_slice = 25.0
        elif n >= 33:
            time_slice = 15.0
        else:
            time_slice = 10.0
            
        if _elapsed() + time_slice > _DEADLINE - 2:
            time_slice = _DEADLINE - 2 - _elapsed()
            if time_slice < 1.0:
                break
                
        t_start = time.time()
        found = False
        restarts = 10 if n < 35 else 5
        
        for r in range(restarts):
            if time.time() - t_start > time_slice:
                break
            rem = time_slice - (time.time() - t_start)
            if rem < 0.5:
                break
            
            # Vary p slightly
            p = 0.45 + 0.02 * (r % 3)
            rng = random.Random(r * 1000 + n)
            adj_init = _random_adj(n, p, rng)
            
            adj_res = _repair_graph(adj_init, n, rng, rem)
            if adj_res is not None:
                # Verify
                if _has_k4(adj_res, n) is None and _find_ind_6(adj_res, n) is None:
                    best_n = n
                    best_edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj_res[i] >> j) & 1:
                                best_edges.append([i, j])
                    found = True
                    break
        
        if found:
            return {'n': best_n, 'edges': best_edges}
            
    # Fallback if n=20..35 failed (unlikely)
    if best_n == 0:
        n = 20
        rng = random.Random(42)
        adj_init = _random_adj(n, 0.5, rng)
        adj_res = _repair_graph(adj_init, n, rng, max(1.0, _DEADLINE - 5 - _elapsed()))
        if adj_res is not None:
            best_n = n
            best_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_res[i] >> j) & 1:
                        best_edges.append([i, j])
        else:
            # Absolute fallback: Empty graph on 5 vertices? No, needs no K4 (ok) and no Ind 6 (ok if n<6).
            # But we want max n.
            # If repair fails completely, return a valid small graph.
            # C5 is K4-free, alpha=2.
            n = 5
            best_edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
            best_n = n
            
    return {'n': best_n, 'edges': best_edges}
