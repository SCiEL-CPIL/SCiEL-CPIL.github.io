import time
import random

def _greedy_independent_set(adj, n):
    """
    Finds a large independent set using a greedy heuristic.
    Returns the set of vertices.
    """
    available = (1 << n) - 1
    indep_set = 0
    while available:
        # Pick vertex with minimum degree in the induced subgraph?
        # For speed, just pick the lowest vertex or a random one.
        # Let's pick the one with the fewest neighbors in 'available' to maximize future choices.
        # This is O(n^2) per step, total O(n^3). For n=40, this is fine.
        
        best_v = -1
        min_deg = n
        temp_avail = available
        while temp_avail:
            v_bit = temp_avail & -temp_avail
            v = v_bit.bit_length() - 1
            # Count neighbors in available
            deg = bin(adj[v] & available).count('1')
            if deg < min_deg:
                min_deg = deg
                best_v = v
            temp_avail &= temp_avail - 1
            
        if best_v == -1:
            break
            
        indep_set |= (1 << best_v)
        # Remove best_v and its neighbors from available
        available &= ~(adj[best_v] | (1 << best_v))
        
    return indep_set

def _has_independent_set_10(adj_mask, n):
    """
    Exact check for independent set of size 10.
    """
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(available_mask, size):
        if size >= 10:
            return True
        if available_mask == 0:
            return False
        
        if popcount(available_mask) < 10 - size:
            return False
            
        v_bit = available_mask & -available_mask
        v = v_bit.bit_length() - 1
        
        new_avail = available_mask & ~(adj_mask[v] | v_bit)
        if dfs(new_avail, size + 1):
            return True
            
        new_avail = available_mask & ~v_bit
        if dfs(new_avail, size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def _random_maximal_triangle_free(n, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
        
    adj = [0] * n
    edges_to_add = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_add)
    
    for u, v in edges_to_add:
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try to find a graph for n from 40 down to 28
    for n in range(40, 27, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        attempts = 0
        max_attempts = 100  # More attempts per n
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline - 5:
                break
                
            rng = random.Random(attempts * 1000 + n)
            adj = _random_maximal_triangle_free(n, seed=attempts * 7 + n)
            
            # Local search
            search_iters = 0
            max_search_iters = 5000
            last_check = 0
            
            while search_iters < max_search_iters:
                search_iters += 1
                
                if time.time() > deadline - 5:
                    break
                
                # Periodic check
                if search_iters - last_check > 50:
                    last_check = search_iters
                    # Greedy check
                    indep_mask = _greedy_independent_set(adj, n)
                    if bin(indep_mask).count('1') < 10:
                        # Potential success, run exact check
                        if not _has_independent_set_10(adj, n):
                            # Found a valid graph!
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            return {'n': n, 'edges': edges}
                    
                # Move: Try to add an edge
                # Pick random u, v
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                
                if adj[u] & (1 << v):
                    # Edge exists, maybe remove it?
                    # For now, let's just try to add non-edges
                    # If we want to explore more, we can remove edges too.
                    # Let's stick to adding edges first to reduce alpha.
                    # But if we get stuck, we need to remove.
                    # Let's do 80% add, 20% remove.
                    if rng.random() < 0.2:
                        # Remove edge
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    # else do nothing, try again
                else:
                    # Try to add edge (u, v)
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        # Can't add directly.
                        # Try to "fix" by removing a common neighbor edge?
                        # Pick a random common neighbor w
                        w_bit = common & -common
                        w = w_bit.bit_length() - 1
                        
                        # Remove (u, w) or (v, w)?
                        # Let's remove (u, w) and then add (u, v)
                        # Check if (u, v) is still safe after removing (u, w)?
                        # Removing (u, w) might not help if there are other common neighbors.
                        # This is complex. Let's just skip this move.
                        # Or, try to add edge between two vertices in a large independent set.
                        
                        # Heuristic: If we are stuck, find a large independent set and try to add edges within it.
                        if search_iters % 100 == 0:
                            indep_mask = _greedy_independent_set(adj, n)
                            # Get list of vertices in indep_mask
                            verts = []
                            temp = indep_mask
                            while temp:
                                b = temp & -temp
                                verts.append(b.bit_length() - 1)
                                temp &= temp - 1
                                
                            if len(verts) >= 2:
                                # Pick two random vertices from the independent set
                                i1, i2 = rng.sample(verts, 2)
                                if not (adj[i1] & (1 << i2)):
                                    common2 = adj[i1] & adj[i2]
                                    if common2 == 0:
                                        adj[i1] |= (1 << i2)
                                        adj[i2] |= (1 << i1)

        # If no graph found for this n, try smaller n

    # Fallback: return a small valid graph if we didn't find a big one
    # We should have found at least n=27 or similar.
    # If we failed completely, return a trivial one?
    # The loop goes down to 28. If we fail all, we return n=1?
    # Better to ensure we at least return the best we found.
    # But the loop breaks if found. If not found, it continues to next n.
    # If it finishes all n, it falls through.
    
    # Let's just return a known small graph as fallback
    # Andrasfai graph And(3) on 9 vertices? Or just a cycle C5 blown up?
    # For safety, return n=5, C5
    return {'n': 5, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]}
