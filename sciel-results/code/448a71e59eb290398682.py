import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    
    # Helper: Check triangle-free
    def is_triangle_free(adj):
        for i in range(n):
            # adj[i] is bitmask of neighbors
            # For each neighbor j, check if adj[i] & adj[j] != 0
            # Actually, if (i,j) is edge, then adj[i] & adj[j] must be 0?
            # No, adj[i] contains j. adj[j] contains i.
            # Common neighbors are adj[i] & adj[j].
            # If there is a common neighbor k, then (i,j,k) is a triangle.
            # So we need (adj[i] & adj[j]) == 0 for all edges (i,j).
            # Wait, adj[i] includes j. adj[j] includes i.
            # So adj[i] & adj[j] will always have at least... no.
            # If k is common neighbor, k is in adj[i] and k is in adj[j].
            # So we check if (adj[i] & adj[j]) == 0.
            # But i is in adj[j] and j is in adj[i].
            # So adj[i] & adj[j] will contain... nothing?
            # i is not in adj[i] (no self loops). j is not in adj[j].
            # So if (i,j) is an edge, i is in adj[j], j is in adj[i].
            # The intersection adj[i] & adj[j] contains vertices k such that k is neighbor of i AND k is neighbor of j.
            # So yes, if (adj[i] & adj[j]) != 0, there is a triangle.
            pass
        # Optimization: Only check edges.
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                mask -= lsb
                if (adj[i] & adj[j]) != 0:
                    return False
        return True

    # Helper: Check 3-colorable
    def is_3_colorable(adj):
        # Backtracking
        colors = [-1] * n
        
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try colors 0, 1, 2
            for c in range(3):
                # Check if any neighbor has color c
                ok = True
                mask = adj[v]
                while mask:
                    lsb = mask & -mask
                    u = lsb.bit_length() - 1
                    mask -= lsb
                    if colors[u] == c:
                        ok = False
                        break
                if ok:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    # Generate Clebsch
    def get_clebsch():
        vectors = [i for i in range(32) if bin(i).count('1') % 2 == 0]
        adj = [0] * n
        for u_idx in range(n):
            for v_idx in range(u_idx + 1, n):
                xor = vectors[u_idx] ^ vectors[v_idx]
                if bin(xor).count('1') == 4:
                    adj[u_idx] |= (1 << v_idx)
                    adj[v_idx] |= (1 << u_idx)
        return adj

    best_adj = get_clebsch()
    best_count = count_edges(best_adj)
    
    # Local Search
    # Strategy: Remove one edge, add as many as possible
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        current_adj = best_adj[:]
        
        # Pick a random edge to remove
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if current_adj[i] & (1 << j):
                    edges.append((i, j))
        
        if not edges:
            continue
            
        u, v = random.choice(edges)
        
        # Remove (u, v)
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        
        # Try to add edges
        # Generate list of non-edges
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not (current_adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        improved = False
        for i, j in non_edges:
            # Check if adding (i, j) creates a triangle
            # Triangle if (adj[i] & adj[j]) != 0
            if (current_adj[i] & current_adj[j]) == 0:
                # Tentatively add
                current_adj[i] |= (1 << j)
                current_adj[j] |= (1 << i)
                improved = True
                # Check if graph is still triangle-free?
                # We checked the new edge. But does it create triangles with other edges?
                # No, adding an edge (i,j) only creates triangles involving (i,j).
                # So if (adj[i] & adj[j]) == 0 before adding, it's safe.
        
        if not improved:
            continue
            
        new_count = count_edges(current_adj)
        
        if new_count > best_count:
            # Check if 3-colorable
            if not is_3_colorable(current_adj):
                best_adj = current_adj
                best_count = new_count
            else:
                # It's 3-colorable, so not a valid candidate for "not 3-colorable"
                # But maybe we can remove another edge to make it non-3-colorable?
                # For now, skip.
                pass
        elif new_count == best_count:
            # Maybe swap for better structure?
            # If current best is 3-colorable (it shouldn't be), we'd want to fix it.
            # But we only update if not 3-colorable.
            pass

    return [ [u, v] for u in range(n) for v in range(u+1, n) if best_adj[u] & (1 << v) ]
