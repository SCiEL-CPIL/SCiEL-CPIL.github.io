import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 18
    
    # Bitmask utilities
    def popcount(x):
        return bin(x).count('1')
        
    def get_subsets_of_size(k):
        # Generator for subsets of range(n) of size k as bitmasks
        # This is a bit slow in pure python, but n=18 is small
        # We can use Gosper's hack or just combinations
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= 1 << v
            yield mask

    def is_k4_free(adj):
        # Check if graph is K4-free
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    # Check if C contains an edge
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if adj[w] & C:
                            return False
                        temp &= temp - 1
        return True

    def get_triangles(adj):
        tris = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if w > v: # Avoid duplicates
                            if adj[v] & (1 << w):
                                # Triangle u, v, w
                                mask = (1 << u) | (1 << v) | (1 << w)
                                tris.append(mask)
                        temp &= temp - 1
        return tris

    def max_triangle_free_size(adj, triangles):
        # Find largest k such that there is a triangle-free subset of size k
        # We check k from n down to 1
        # Optimization: if k > len(triangles) + something? No.
        # Just check all k
        for k in range(n, 0, -1):
            # Check if any subset of size k is triangle-free
            found = False
            for mask in get_subsets_of_size(k):
                is_tf = True
                for t in triangles:
                    if (mask & t) == t:
                        is_tf = False
                        break
                if is_tf:
                    found = True
                    break
            if found:
                return k
        return 0

    def add_edge_safe(adj, u, v):
        # Check if adding (u,v) creates a K4
        if adj[u] & (1 << v):
            return False # Already exists
        C = adj[u] & adj[v]
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            if adj[w] & C:
                return False
            temp &= temp - 1
        return True

    def try_improve(adj, triangles, best_val):
        # Try to add an edge to reduce best_val
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if add_edge_safe(adj, u, v):
                        # Tentatively add
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                        # Update triangles
                        new_tris = get_triangles(adj)
                        
                        # Check if max triangle free size decreased
                        # We only need to check if a triangle-free subset of size best_val still exists
                        # If it does, the max size is >= best_val, so no improvement (or same)
                        # If it doesn't, the max size is < best_val, so improvement
                        exists = False
                        for mask in get_subsets_of_size(best_val):
                            is_tf = True
                            for t in new_tris:
                                if (mask & t) == t:
                                    is_tf = False
                                    break
                            if is_tf:
                                exists = True
                                break
                                
                        if not exists:
                            # Improvement found
                            triangles = new_tris
                            return True, adj, triangles, best_val - 1 # New best val is at most best_val - 1
                            # Actually we should recalculate exact max, but for greedy, knowing it dropped is enough
                            # To be safe, let's recalculate in the main loop or here
                            # Let's just return the new state and let the caller recalculate if needed
                            # Or we can find the new max here
                            new_best = max_triangle_free_size(adj, new_tris)
                            return True, adj, new_tris, new_best
                        else:
                            # No improvement, revert
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
        return False, adj, triangles, best_val

    # Initialize
    best_adj = None
    best_triangles = None
    best_val = 18 # Max possible
    
    # Strategy 1: Start with K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    triangles = get_triangles(adj)
    val = max_triangle_free_size(adj, triangles)
    
    if val < best_val:
        best_val = val
        best_adj = adj[:]
        best_triangles = triangles[:]
        
    # Local Search
    current_adj = adj[:]
    current_triangles = triangles[:]
    current_val = val
    
    while time.time() < deadline:
        improved, current_adj, current_triangles, current_val = try_improve(current_adj, current_triangles, current_val)
        if improved:
            if current_val < best_val:
                best_val = current_val
                best_adj = current_adj[:]
                best_triangles = current_triangles[:]
        else:
            # Stuck, try a random restart or perturb
            # Perturb: remove a random edge
            edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if current_adj[u] & (1 << v):
                        edges.append((u, v))
            if edges:
                u, v = random.choice(edges)
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                current_triangles = get_triangles(current_adj)
                current_val = max_triangle_free_size(current_adj, current_triangles)
            else:
                # Random restart
                break

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
