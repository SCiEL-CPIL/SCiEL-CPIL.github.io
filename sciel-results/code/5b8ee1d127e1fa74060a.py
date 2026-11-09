import time
import random

def _is_triangle_free(n, adj):
    """Check triangle-freeness using bitmasks."""
    for i in range(n):
        a = adj[i]
        while a:
            lsb = a & (-a)
            a ^= lsb
            j = lsb.bit_length() - 1
            # If j is neighbor of i, check if they share a common neighbor
            # common = adj[j] & adj[i]
            # If common has any bit set, there is a triangle.
            # We can just check if (adj[j] & adj[i]) != 0
            # Note: adj[i] includes j, adj[j] includes i.
            # So (adj[j] & adj[i]) will always have at least bits i and j set?
            # No, adj[i] has bit j set. adj[j] has bit i set.
            # So (adj[j] & adj[i]) has bit i and bit j set?
            # Bit i in adj[j] is 1. Bit i in adj[i] is 0 (no self loop).
            # So bit i is 0 in intersection.
            # Bit j in adj[i] is 1. Bit j in adj[j] is 0.
            # So bit j is 0 in intersection.
            # So if intersection is non-zero, it's a triangle.
            if adj[j] & adj[i]:
                return False
    return True

def _has_independent_set_size(n, adj, target=10):
    """Returns True if an independent set of size >= target exists."""
    # Branch and bound
    # State: (cand_mask, current_size)
    # cand_mask: bitmask of vertices still available to be added to the independent set
    # We want to see if we can pick 'target - current_size' more vertices from cand_mask
    # such that they are mutually non-adjacent.
    
    # Pre-compute non-neighbors for faster access?
    # non_adj[i] = ~adj[i] & ((1 << n) - 1)
    # When we pick a vertex v, the new candidate set is cand_mask & non_adj[v]
    
    non_adj = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        non_adj[i] = all_mask & ~adj[i]
        non_adj[i] &= ~(1 << i) # Remove self

    def bnb(cand, sz):
        if sz >= target:
            return True
        if cand == 0:
            return False
        # Upper bound: if sz + popcount(cand) < target, return False
        # popcount is fast in Python 3.8+
        if sz + cand.bit_count() < target:
            return False
            
        # Heuristic: pick a vertex with high degree in the subgraph induced by cand?
        # Or low degree?
        # For independent set, we want vertices that are "far" from others.
        # Low degree in the *original* graph might be good.
        # Let's just pick the first available vertex to keep it simple, or use a greedy bound.
        
        # Greedy upper bound on independent set size in cand:
        # This is expensive. Let's use a simpler bound or just branch.
        # To speed up, pick a vertex v from cand.
        # Branch 1: Include v. New cand = cand & non_adj[v]. sz + 1.
        # Branch 2: Exclude v. New cand = cand & ~(1 << v). sz.
        
        # Pick vertex with highest degree in cand to prune "Exclude" branch faster?
        # If we exclude a high-degree vertex, we don't lose many candidates?
        # Actually, if we INCLUDE a high-degree vertex, we remove many candidates.
        # So we should try INCLUDE first for high-degree vertices?
        # Or EXCLUDE first for low-degree vertices?
        
        # Let's pick the vertex with the highest degree in the induced subgraph of cand.
        best_v = -1
        best_deg = -1
        m = cand
        while m:
            lsb = m & (-m)
            m ^= lsb
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        v_bit = 1 << best_v
        
        # Try Include
        if bnb(cand & non_adj[best_v], sz + 1):
            return True
        # Try Exclude
        if bnb(cand & ~v_bit, sz):
            return True
        return False

    return bnb(all_mask, 0)

def _edges_from_adj(n, adj):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _construct_graph(n, deadline):
    """
    Attempts to construct a triangle-free graph on n vertices with alpha < 10.
    Uses local search.
    """
    if time.time() > deadline:
        return None
        
    # Try multiple random restarts
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        random.seed(attempt * 12345 + 67890)
        
        # Initialize random graph
        adj = [0] * n
        # p=0.2 might be too dense, p=0.1 sparse.
        # We need enough edges to reduce alpha, but not so many as to form triangles.
        # Let's start sparse and add edges.
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Enforce triangle-free
        # While there is a triangle, remove an edge
        # We can do this efficiently by checking all pairs
        changed = True
        while changed:
            changed = False
            for i in range(n):
                a = adj[i]
                while a:
                    lsb = a & (-a)
                    a ^= lsb
                    j = lsb.bit_length() - 1
                    if adj[i] & (1 << j):
                        common = adj[i] & adj[j]
                        if common:
                            # Triangle found: i, j, k where k is in common
                            k_bit = common & (-common)
                            k = k_bit.bit_length() - 1
                            # Remove edge (i, j)
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
                            break
                if changed:
                    break
                    
        if not _is_triangle_free(n, adj):
            continue
            
        # Check alpha
        if not _has_independent_set_size(n, adj, 10):
            return {'n': n, 'edges': _edges_from_adj(n, adj)}
            
        # Local Search: Try to add edges to reduce alpha
        # Find an independent set of size 10
        # We can reuse the B&B logic to find one, or just use a greedy heuristic.
        # Let's use a greedy heuristic to find a large independent set.
        def find_large_is(adj, n, target):
            # Greedy: pick vertex with min degree in remaining set
            cand = (1 << n) - 1
            is_set = []
            while cand and len(is_set) < target:
                # Find vertex with min degree in cand
                best_v = -1
                best_deg = n + 1
                m = cand
                while m:
                    lsb = m & (-m)
                    m ^= lsb
                    v = lsb.bit_length() - 1
                    deg = (adj[v] & cand).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                if best_v == -1:
                    break
                is_set.append(best_v)
                cand &= ~(1 << best_v)
                cand &= ~adj[best_v]
            return is_set

        improved = True
        while improved and time.time() < deadline:
            improved = False
            is_set = find_large_is(adj, n, 10)
            if len(is_set) < 10:
                # Should not happen if _has_independent_set_size returned True
                # but maybe the greedy didn't find one?
                # If greedy fails to find 10, we should check with B&B.
                if not _has_independent_set_size(n, adj, 10):
                    return {'n': n, 'edges': _edges_from_adj(n, adj)}
                else:
                    break # Stuck, restart or give up
            
            # Try to add edges within this independent set
            # For each pair (u, v) in is_set, if not adjacent, check if adding (u,v) is safe
            added_any = False
            for u_idx in range(len(is_set)):
                u = is_set[u_idx]
                for v_idx in range(u_idx + 1, len(is_set)):
                    v = is_set[v_idx]
                    if adj[u] & (1 << v):
                        continue # Already adjacent
                    # Check if adding (u, v) creates a triangle
                    # Triangle exists if N(u) and N(v) intersect
                    if adj[u] & adj[v]:
                        continue
                    # Safe to add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added_any = True
                    # Note: adding an edge might break the independence of the set,
                    # but we are trying to fix the graph.
                    
            if added_any:
                improved = True
                # Re-check if we solved it
                if not _has_independent_set_size(n, adj, 10):
                    return {'n': n, 'edges': _edges_from_adj(n, adj)}
            else:
                # Could not add any edge in this independent set
                # Try a different independent set?
                # For now, break and restart
                break

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin
    
    # Try n=40 first, then 39
    for n in [40, 39]:
        result = _construct_graph(n, deadline)
        if result:
            return result
            
    # Fallback: Andrásfai graph And(9) on 26 vertices
    # Z_26, edges if (j-i) mod 26 in {1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 0?}
    # And(3) is defined on 3n vertices. For n=9, 27 vertices?
    # Wait, And(k) is on 3k vertices?
    # And(3) is on 9 vertices? No, And(3) is the graph on 9 vertices?
    # The Andrásfai graph And(3) has 9 vertices?
    # Let's check the seed program's fallback:
    # n = 26, edges if (j - i) % 26 % 3 == 1.
    # This is a circulant graph.
    # Let's just use that.
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}
