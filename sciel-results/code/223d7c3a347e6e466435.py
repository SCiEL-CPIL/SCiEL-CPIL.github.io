import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    n = 18
    # Initialize with T(18,3)
    # Parts: 0-5, 6-11, 12-17
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    # Build adjacency masks
    N = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    N[u] |= (1 << v)
                    N[v] |= (1 << u)
    
    def is_k4_free(N):
        # Check all 4-subsets
        # 18 choose 4 is 3060, very fast
        for a in range(n):
            for b in range(a + 1, n):
                if not (N[a] >> b) & 1: continue
                for c in range(b + 1, n):
                    if not (N[b] >> c) & 1: continue
                    if not (N[a] >> c) & 1: continue
                    # a,b,c form a triangle. Check if any d forms K4
                    # d must be neighbor of a, b, c
                    common = N[a] & N[b] & N[c]
                    # Remove a,b,c
                    common &= ~(1 << a)
                    common &= ~(1 << b)
                    common &= ~(1 << c)
                    if common:
                        return False
        return True

    def max_triangle_free_size(N):
        # Find max size of subset with no triangles
        # Try k from n down to 1
        # We can use a recursive search or brute force for small k
        # Given n=18, brute force for k=18..10 is feasible if optimized
        # But let's use a simple branch and bound
        
        best = 0
        # We want to find a set S such that no triangle in S
        # Triangle exists if u,v,w in S and u~v, v~w, w~u
        
        # Helper to check if a set (mask) is triangle free
        def is_tf(mask):
            # Iterate over set bits
            m = mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                m ^= lsb
                # Check neighbors of i in mask
                neighbors = N[i] & mask
                # If any two neighbors are connected, triangle
                # Iterate over neighbors
                n_m = neighbors
                while n_m:
                    lsb2 = n_m & -n_m
                    j = lsb2.bit_length() - 1
                    n_m ^= lsb2
                    if (N[j] & neighbors) & ~lsb2:
                        # j and another neighbor of i are connected
                        # Wait, (N[j] & neighbors) contains i? No, i is not in neighbors.
                        # It contains other neighbors of i.
                        # If N[j] intersects neighbors (excluding j), then triangle.
                        # neighbors includes j.
                        # So N[j] & (neighbors ^ (1<<j))
                        if (N[j] & (neighbors ^ (1 << j))):
                            return False
            return True

        # Branch and Bound
        # State: available mask, current set mask
        # We want to maximize size of current set
        
        def search(avail, curr, curr_size):
            nonlocal best
            if curr_size + avail.bit_count() <= best:
                return
            if avail == 0:
                if curr_size > best:
                    best = curr_size
                return
            
            # Pick a vertex
            # Heuristic: pick vertex with most neighbors in avail? Or least?
            # To find large triangle-free sets, picking vertices that "conflict" less might be good?
            # Actually, we just need to find ANY set of size > best.
            # So we can just pick the first available vertex.
            
            lsb = avail & -avail
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            # If we include v, we cannot include any pair of neighbors of v that are connected.
            # This is complex.
            # Simpler: Just remove v from avail and add to curr.
            # But we must ensure no triangle is formed.
            # A triangle is formed if we add v, and v has two neighbors u,w in curr that are connected.
            # So, if (N[v] & curr) has an edge, we cannot add v?
            # No, if u,w in curr and u~w and u~v and w~v, then triangle.
            # So, if (N[v] & curr) is not an independent set, we cannot add v?
            # Actually, if (N[v] & curr) contains an edge, then adding v creates a triangle.
            
            neighbors_in_curr = N[v] & curr
            # Check if neighbors_in_curr is independent
            valid = True
            nmc = neighbors_in_curr
            while nmc:
                lsb_c = nmc & -nmc
                u = lsb_c.bit_length() - 1
                nmc ^= lsb_c
                if (N[u] & neighbors_in_curr) & ~lsb_c:
                    valid = False
                    break
            
            if valid:
                search(avail ^ lsb, curr | lsb, curr_size + 1)
            
            # Branch 2: Exclude v
            search(avail ^ lsb, curr, curr_size)

        search((1 << n) - 1, 0, 0)
        return best

    best_score = max_triangle_free_size(N)
    best_N = N[:]
    
    def get_edges(N):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (N[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Local Search
    while time.time() < deadline:
        # Randomly try to add an edge
        # Pick random non-edge (u,v)
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v: continue
        if (N[u] >> v) & 1: continue
        
        # Check if adding (u,v) creates K4
        # Common neighbors C = N[u] & N[v]
        C = N[u] & N[v]
        # C must be independent
        valid = True
        cm = C
        while cm:
            lsb = cm & -cm
            w = lsb.bit_length() - 1
            cm ^= lsb
            if (N[w] & C) & ~lsb:
                valid = False
                break
        
        if valid:
            # Add edge
            N[u] |= (1 << v)
            N[v] |= (1 << u)
            
            # Check K4 free (just in case, though the check above should suffice for the new K4s involving u,v)
            # Actually, adding (u,v) can only create K4s involving u and v.
            # The check "C is independent" ensures no K4 with u,v.
            # So it is safe.
            
            new_score = max_triangle_free_size(N)
            if new_score < best_score:
                best_score = new_score
                best_N = N[:]
            else:
                # Revert
                N[u] &= ~(1 << v)
                N[v] &= ~(1 << u)
                
        # Occasionally remove an edge to escape local optima
        if random.random() < 0.1:
            # Pick a random edge to remove
            # Find a random edge
            # Iterate through all edges, pick one randomly
            all_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (N[i] >> j) & 1:
                        all_edges.append((i, j))
            if all_edges:
                u, v = random.choice(all_edges)
                N[u] &= ~(1 << v)
                N[v] &= ~(1 << u)
                new_score = max_triangle_free_size(N)
                if new_score < best_score:
                    best_score = new_score
                    best_N = N[:]
                else:
                    # Revert
                    N[u] |= (1 << v)
                    N[v] |= (1 << u)

    return get_edges(best_N)
