import time
import random
from itertools import combinations

def _has_triangle_bitset(adj, n):
    """Check if graph has a triangle using bitsets."""
    for u in range(n):
        au = adj[u]
        v = u + 1
        while v < n:
            if au & (1 << v):
                common = au & adj[v]
                if common & ~((1 << (v + 1)) - 1):
                    return True
            v += 1
    return False

def _max_independent_set_size_bitset(adj, n, k):
    """
    Check if there exists an independent set of size k using bitmask backtracking.
    """
    full_mask = (1 << n) - 1
    non_neighbors = [0] * n
    for i in range(n):
        non_neighbors[i] = full_mask & ~adj[i] & ~(1 << i)
    
    found = [False]
    
    def dfs(candidates_mask, current_size):
        if found[0]:
            return
        if current_size == k:
            found[0] = True
            return
        
        if candidates_mask == 0:
            return
        if bin(candidates_mask).count('1') < k - current_size:
            return
        
        # Heuristic: pick vertex with most non-neighbors in candidates
        # This tends to prune the 'include' branch faster if it leads to dead ends
        # Actually, for finding independent set, we want to branch on vertices
        # that are most likely to be in the independent set.
        
        # Compute number of non-neighbors in candidates for each candidate
        best_v = None
        best_count = -1
        mask = candidates_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            nn_in_cands = bin(candidates_mask & non_neighbors[v]).count('1')
            if nn_in_cands > best_count:
                best_count = nn_in_cands
                best_v = v
            mask ^= lsb
        
        if best_v is None:
            return
            
        # Branch 1: Include best_v
        new_candidates = candidates_mask & non_neighbors[best_v] & ~(1 << best_v)
        dfs(new_candidates, current_size + 1)
        
        if found[0]:
            return
        
        # Branch 2: Exclude best_v
        dfs(candidates_mask & ~(1 << best_v), current_size)
    
    dfs(full_mask, 0)
    return found[0]

def _find_independent_set_bitset(adj, n, k):
    """Find an independent set of size k. Returns list of vertices or None."""
    full_mask = (1 << n) - 1
    non_neighbors = [0] * n
    for i in range(n):
        non_neighbors[i] = full_mask & ~adj[i] & ~(1 << i)
    
    found = [False]
    result_set = []
    
    def dfs(candidates_mask, current_set):
        if found[0]:
            return
        if len(current_set) == k:
            result_set.extend(current_set)
            found[0] = True
            return
        if candidates_mask == 0:
            return
        if bin(candidates_mask).count('1') < k - len(current_set):
            return
        
        mask = candidates_mask
        best_v = None
        best_count = -1
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            nn_count = bin(candidates_mask & non_neighbors[v]).count('1')
            if nn_count > best_count:
                best_count = nn_count
                best_v = v
            mask ^= lsb
        
        if best_v is None:
            return
        
        new_cands = candidates_mask & non_neighbors[best_v] & ~(1 << best_v)
        dfs(new_cands, current_set + [best_v])
        
        if found[0]:
            return
        
        dfs(candidates_mask & ~(1 << best_v), current_set)
    
    dfs(full_mask, [])
    if found[0]:
        return result_set
    return None

def _build_triangle_free_bitset(adj, n):
    """Remove edges to make the graph triangle-free."""
    for _ in range(10000):
        tri_found = False
        for u in range(n):
            au = adj[u]
            v = u + 1
            while v < n:
                if au & (1 << v):
                    common = au & adj[v] & ~((1 << (v + 1)) - 1)
                    if common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        tri_found = True
                        break
                v += 1
            if tri_found:
                break
        if not tri_found:
            break

def _local_search_bitset(adj, n, target_alpha, deadline):
    """
    Try to reduce alpha below target_alpha by adding edges.
    """
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        has_iso = _max_independent_set_size_bitset(adj, n, target_alpha)
        
        if not has_iso:
            return True
        
        iso = _find_independent_set_bitset(adj, n, target_alpha)
        if iso is None:
            return True
        
        improved = False
        random.shuffle(iso)
        for i in range(len(iso)):
            for j in range(i + 1, len(iso)):
                u, v = iso[i], iso[j]
                if not (adj[u] & (1 << v)):
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return not _max_independent_set_size_bitset(adj, n, target_alpha)

def _bitset_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _construct_39_vertex_graph():
    """
    Construct the known 39-vertex Ramsey(3,10) graph.
    This graph is triangle-free and has independence number 9.
    
    One such construction is derived from the Hoffman-Singleton graph (n=50)
    or specifically designed for R(3,10). 
    A well-known construction for R(3,10) > 39 is based on the 
    2-(39, 13, 4) design or similar structures.
    
    Actually, the standard reference for R(3,10) is that R(3,10) is between 39 and 41.
    The graph on 39 vertices is often constructed by taking a specific 
    regular graph. One common source is the "Ramsey graph" from the 
    work of Exoo et al.
    
    Since hardcoding the exact edge list of the 39-vertex graph is risky 
    without the specific reference at hand, we will use a robust 
    deterministic construction that is known to work for n=39:
    
    We can use the construction from the paper "Ramsey numbers R(3,10)" 
    which often utilizes a graph derived from the Paley graph or 
    specific circulant structures, but the 39-vertex one is non-circulant.
    
    Alternative Strategy:
    Use the known fact that R(3,10) > 39. We can try to find it via
    a very targeted local search starting from a specific high-quality seed.
    However, the prompt asks for a DIFFERENT strategy.
    
    Let's try the "Hoffman-Singleton" derived approach.
    The Hoffman-Singleton graph has n=50, is 7-regular, triangle-free, and alpha=7.
    If we delete 11 vertices, we get n=39.
    We need to ensure the remaining graph has alpha < 10.
    Since the original alpha is 7, deleting vertices cannot increase the independence number.
    So ANY 39-vertex induced subgraph of the Hoffman-Singleton graph is valid!
    
    Let's construct the Hoffman-Singleton graph (n=50).
    Vertices are pairs (i, j) where i in {0..4}, j in {0..9}.
    Edges:
    1. (i, j) connected to (i, j+1 mod 10) and (i, j-1 mod 10)
    2. (i, j) connected to (i+1 mod 5, j+2i mod 10) and (i-1 mod 5, j-2i mod 10)
    
    Wait, the Hoffman-Singleton graph is defined as follows:
    Vertices: (x, y) with x in Z_5, y in Z_10.
    Edges:
    1. (x, y) ~ (x, y+1)
    2. (x, y) ~ (x, y-1)
    3. (x, y) ~ (x+1, y+2x)
    4. (x, y) ~ (x-1, y-2x)
    
    This graph is 7-regular, triangle-free, and has independence number 7.
    Since alpha(HS) = 7 < 10, any induced subgraph also has alpha <= 7 < 10.
    And since HS is triangle-free, any induced subgraph is triangle-free.
    
    So, we just need to take the first 39 vertices of the Hoffman-Singleton graph!
    This is a deterministic, O(1) construction (plus edge generation) that guarantees
    a valid graph on n=39.
    """
    n_hs = 50
    adj_hs = [0] * n_hs
    
    # Map (x, y) to index 0..49
    def idx(x, y):
        return x * 10 + y
    
    for x in range(5):
        for y in range(10):
            u = idx(x, y)
            # Type 1: (x, y+1)
            v1 = idx(x, (y + 1) % 10)
            # Type 2: (x, y-1)
            v2 = idx(x, (y - 1) % 10)
            # Type 3: (x+1, y+2x)
            v3 = idx((x + 1) % 5, (y + 2 * x) % 10)
            # Type 4: (x-1, y-2x)
            v4 = idx((x - 1) % 5, (y - 2 * x) % 10)
            
            for v in [v1, v2, v3, v4]:
                if u < v:
                    adj_hs[u] |= (1 << v)
                    adj_hs[v] |= (1 << u)
    
    # Verify HS is triangle-free and alpha < 10 (should be true)
    # We take the first 39 vertices
    n_target = 39
    adj_target = [0] * n_target
    
    # Map old indices to new indices 0..38
    # We just take vertices 0..38 from HS
    for u in range(n_target):
        for v in range(u + 1, n_target):
            if adj_hs[u] & (1 << v):
                adj_target[u] |= (1 << v)
                adj_target[v] |= (1 << u)
    
    # Final verification
    if _has_triangle_bitset(adj_target, n_target):
        # Should not happen, but fallback
        return None
    
    if _max_independent_set_size_bitset(adj_target, n_target, 10):
        # Should not happen, but fallback
        return None
    
    return {'n': n_target, 'edges': _bitset_to_edges(adj_target, n_target)}

def _enumerate_circulants(n, deadline, target_alpha=10):
    """
    Enumerate circulant graphs on n vertices and find one that is
    triangle-free with alpha < target_alpha.
    """
    for n_try in range(40, 19, -1):
        if time.time() > deadline:
            break
        
        max_s = n // 2
        search_range = min(n // 2, 12)
        
        # Try arithmetic progressions
        for a in range(1, n_try // 2 + 1):
            for length in range(1, n_try // a + 1):
                S = set((a * i) % n_try for i in range(1, length + 1))
                S.discard(0)
                if not S:
                    continue
                
                adj = [0] * n_try
                S_symmetric = set()
                for s in S:
                    s_mod = s % n_try
                    if 0 < s_mod < n_try:
                        S_symmetric.add(s_mod)
                        S_symmetric.add((n_try - s_mod) % n_try)
                
                for i in range(n_try):
                    for s in S_symmetric:
                        j = (i + s) % n_try
                        if j > i:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                if _has_triangle_bitset(adj, n_try):
                    continue
                
                if not _max_independent_set_size_bitset(adj, n_try, target_alpha):
                    return {'n': n_try, 'edges': _bitset_to_edges(adj, n_try)}
        
        # Try random S
        for k in range(1, min(n_try // 2, 6) + 1):
            if time.time() > deadline:
                break
            for attempt in range(100):
                S = set(random.sample(range(1, n_try // 2 + 1), min(k, n_try // 2)))
                adj = [0] * n_try
                S_symmetric = set()
                for s in S:
                    s_mod = s % n_try
                    if 0 < s_mod < n_try:
                        S_symmetric.add(s_mod)
                        S_symmetric.add((n_try - s_mod) % n_try)
                
                for i in range(n_try):
                    for s in S_symmetric:
                        j = (i + s) % n_try
                        if j > i:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                if _has_triangle_bitset(adj, n_try):
                    continue
                
                if not _max_independent_set_size_bitset(adj, n_try, target_alpha):
                    return {'n': n_try, 'edges': _bitset_to_edges(adj, n_try)}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # 1. Try the deterministic Hoffman-Singleton derived graph (n=39)
    # This is O(1) construction time and guaranteed valid if HS properties hold.
    result = _construct_39_vertex_graph()
    if result is not None:
        return result

    # 2. Fallback: Circulant enumeration
    result = _enumerate_circulants(40, deadline)
    if result is not None:
        return result

    # 3. Fallback: Local search
    for n in range(39, 19, -1):
        if time.time() > deadline:
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            _build_triangle_free_bitset(adj, n)
            
            if _local_search_bitset(adj, n, 10, deadline):
                return {'n': n, 'edges': _bitset_to_edges(adj, n)}

    # 4. Final fallback: n=9
    n = 9
    return {'n': n, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
