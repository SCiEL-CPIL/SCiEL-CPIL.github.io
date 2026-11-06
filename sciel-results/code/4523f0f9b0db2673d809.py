import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 18
    full_mask = (1 << n) - 1
    
    # Helper to create triangle mask from u, v, w
    def tri_mask(u, v, w):
        return (1 << u) | (1 << v) | (1 << w)

    def get_initial_triangles(adj):
        tris = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if w > v:
                            if adj[v] & (1 << w):
                                tris.add(tri_mask(u, v, w))
                        temp &= temp - 1
        return tris

    def can_add_edge_no_k4(adj, u, v):
        if adj[u] & (1 << v):
            return False
        C = adj[u] & adj[v]
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            # If w is connected to any other vertex in C, we have a K4
            # Specifically, if adj[w] intersects C (excluding u and v, but u,v are not in C usually? 
            # C is common neighbors of u and v. If w in C is connected to x in C, then u,v,w,x form K4?
            # No. u-v is the edge. u-w and v-w exist. If w-x exists, do we have K4?
            # The K4 would be {u, v, w, x}.
            # Edges: (u,v) exists. (u,w) exists. (v,w) exists.
            # We need (w,x), (u,x), (v,x).
            # x is in C, so (u,x) and (v,x) exist.
            # So if any w in C has an edge to another x in C, we form a K4.
            if adj[w] & C:
                return False
            temp &= temp - 1
        return True

    def add_edge(adj, u, v, triangles):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        C = adj[u] & adj[v]
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            if w > v:
                triangles.add(tri_mask(u, v, w))
            elif w < v:
                triangles.add(tri_mask(u, w, v)) # u < w < v? No, u < v is fixed.
                # Triangles are unordered sets of vertices. 
                # My tri_mask doesn't care about order.
                # But I need to ensure I add the correct triangle.
                # The triangle is {u, v, w}.
                # It's fine to just add tri_mask(u, v, w).
                # But I must avoid duplicates. Since I use a set, duplicates are handled.
                pass
            temp &= temp - 1
        # Re-do logic: just iterate all w in C and add tri_mask(u, v, w)
        # The set will handle duplicates.
        
    def remove_edge(adj, u, v, triangles):
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        # Find triangles containing u and v
        # These are formed by common neighbors of u and v (before removal)
        # But we already removed the edge. We need the common neighbors BEFORE removal.
        # So we should calculate C before modifying adj.
        # Let's swap logic: calculate C, then modify adj, then remove tris.
        
    # Revised incremental approach
    def try_add_edge(adj, u, v, triangles):
        if adj[u] & (1 << v):
            return False
        if not can_add_edge_no_k4(adj, u, v):
            return False
        
        # Calculate new triangles
        C = adj[u] & adj[v]
        new_tris = []
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            new_tris.append(tri_mask(u, v, w))
            temp &= temp - 1
            
        # Apply change
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        for t in new_tris:
            triangles.add(t)
            
        return True

    def try_remove_edge(adj, u, v, triangles):
        if not (adj[u] & (1 << v)):
            return False
            
        # Calculate triangles to remove
        C = adj[u] & adj[v]
        tris_to_remove = []
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            tris_to_remove.append(tri_mask(u, v, w))
            temp &= temp - 1
            
        # Apply change
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        for t in tris_to_remove:
            triangles.discard(t)
            
        return True

    def max_triangle_free_size(adj, triangles):
        # Check from high k down
        # Optimization: If we have a triangle-free set of size k, we return k.
        # We can prune if k is too large?
        # For n=18, max k is 18.
        # We can iterate k from 18 down to 1.
        # To speed up, we can first check if the whole graph is triangle-free.
        # If not, check 17, etc.
        
        # Heuristic: The max triangle-free set is at least n - (number of triangles)? No.
        
        # Efficient check:
        # A set S is triangle-free if it contains no triangle from `triangles`.
        # We want max |S|.
        
        # Let's use a simple loop for now, but with early exit.
        for k in range(n, 0, -1):
            # Generate subsets of size k
            # To speed up, we can use a recursive generator or Gosper's hack.
            # But for n=18, combinations is okay if we break early.
            found = False
            for mask in _subsets_of_size(k, n):
                is_tf = True
                for t in triangles:
                    if (mask & t) == t:
                        is_tf = False
                        break
                if is_tf:
                    return k
        return 0

    def _subsets_of_size(k, n):
        if k == 0:
            yield 0
            return
        if k == n:
            yield (1 << n) - 1
            return
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= 1 << v
            yield mask

    def local_search(adj, triangles, current_val, rng, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding edges
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if not (adj[u] & (1 << v)):
                        if can_add_edge_no_k4(adj, u, v):
                            # Tentative add
                            # We need to update triangles and check value
                            # To avoid modifying and reverting complex state, we can simulate
                            # But incremental update is fast.
                            # Let's do it in place and revert if bad.
                            
                            # Save state? No, just do it.
                            # Calculate new triangles list
                            C = adj[u] & adj[v]
                            new_tris = []
                            temp = C
                            while temp:
                                w = (temp & -temp).bit_length() - 1
                                new_tris.append(tri_mask(u, v, w))
                                temp &= temp - 1
                            
                            # Apply
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            for t in new_tris:
                                triangles.add(t)
                                
                            new_val = max_triangle_free_size(adj, triangles)
                            
                            if new_val < current_val:
                                current_val = new_val
                                improved = True
                            else:
                                # Revert
                                for t in new_tris:
                                    triangles.discard(t)
                                adj[u] ^= (1 << v)
                                adj[v] ^= (1 << u)
            
            # Try removing edges
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if adj[u] & (1 << v):
                        # Calculate triangles to remove
                        C = adj[u] & adj[v]
                        tris_to_remove = []
                        temp = C
                        while temp:
                            w = (temp & -temp).bit_length() - 1
                            tris_to_remove.append(tri_mask(u, v, w))
                            temp &= temp - 1
                        
                        # Apply
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        for t in tris_to_remove:
                            triangles.discard(t)
                        
                        new_val = max_triangle_free_size(adj, triangles)
                        
                        if new_val < current_val:
                            current_val = new_val
                            improved = True
                        else:
                            # Revert
                            for t in tris_to_remove:
                                triangles.add(t)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            
        return adj, triangles, current_val

    best_val = n
    best_adj = None
    best_triangles = None
    
    rng = random.Random(42)
    
    # Strategy 1: Tripartite K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    triangles = get_initial_triangles(adj)
    val = max_triangle_free_size(adj, triangles)
    
    if val < best_val:
        best_val = val
        best_adj = adj[:]
        best_triangles = triangles.copy()
        
    # Strategy 2: Random Greedy + Local Search
    # We will run multiple restarts.
    
    for i in range(20): # Reduced restarts to allow deeper local search
        if time.time() > deadline - 10:
            break
            
        # Random init
        adj = [0] * n
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(possible_edges)
        for u, v in possible_edges:
            if can_add_edge_no_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        triangles = get_initial_triangles(adj)
        val = max_triangle_free_size(adj, triangles)
        
        if val < best_val:
            best_val = val
            best_adj = adj[:]
            best_triangles = triangles.copy()
            
        # Local Search
        remaining = deadline - time.time()
        if remaining > 5:
            time_limit = time.time() + min(remaining - 5, 20)
            adj, triangles, val = local_search(adj, triangles, val, rng, time_limit)
            
            if val < best_val:
                best_val = val
                best_adj = adj[:]
                best_triangles = triangles.copy()
                
        if best_val <= 8:
            break

    # Final polish on best found
    if time.time() < deadline - 5:
        time_limit = deadline - 2
        best_adj, best_triangles, best_val = local_search(best_adj, best_triangles, best_val, rng, time_limit)
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
