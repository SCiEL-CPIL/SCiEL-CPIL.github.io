import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin before 120s limit
    
    n = 40
    
    # Bitset adjacency for fast operations
    adj = [0] * n
    
    def bit_count(x):
        return bin(x).count('1')
    
    def neighbors(u):
        return [i for i in range(n) if (adj[u] >> i) & 1]
    
    def would_create_c6(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A new C6 containing (u,v) would be u-w-z-y-x-v-u,
        which requires a path of length 5 from u to v in the current graph.
        Path: u - w - z - y - x - v
        w in N(u), x in N(v).
        Need path w-z-y-x of length 3.
        """
        Nu_mask = adj[u]
        Nv_mask = adj[v]
        
        # Iterate over set bits in Nu_mask
        # To minimize work, iterate over the smaller neighbor set first
        if bin(Nu_mask).count('1') > bin(Nv_mask).count('1'):
            Nu_mask, Nv_mask = Nv_mask, Nu_mask
            u, v = v, u
        
        # Extract neighbors from mask
        Nu = []
        mask = Nu_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            Nu.append(idx)
            mask ^= lsb
            
        Nv = []
        mask = Nv_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            Nv.append(idx)
            mask ^= lsb

        for w in Nu:
            for x in Nv:
                if w == x:
                    continue
                # Check if there's a path of length 3 from w to x
                # i.e., exists z such that w~z and z~y and y~x for some y
                # Simpler: for each z in N(w), check if N(z) intersects N(x)
                # But we must ensure the path is simple (no repeated vertices)
                # w, z, y, x must be distinct.
                
                # Iterate neighbors of w
                mask_w = adj[w]
                while mask_w:
                    lsb = mask_w & -mask_w
                    z = lsb.bit_length() - 1
                    mask_w ^= lsb
                    
                    if z == x:
                        continue
                    # Check common neighbors of z and x
                    # We need a common neighbor y such that y != w and y != x
                    # Note: y cannot be x because x is not in N(x) (no self-loops)
                    # We only need to check y != w.
                    common = adj[z] & adj[x]
                    if common:
                        # Check if the intersection contains any vertex other than w
                        if (common >> w) & 1:
                            common_without_w = common & ~(1 << w)
                            if common_without_w:
                                return True
                        else:
                            return True
        return False

    def local_search(adj, edges, deadline):
        """
        Perform edge swaps to try to increase edge count.
        Remove an edge (a,b) and try to add two new edges (a,c), (b,d) or similar.
        Simpler: try to remove an edge and add a different edge if it fits? No, that keeps count same.
        We want to increase count.
        
        Actually, standard local search for MaxClique-like problems:
        1. Try to add any missing edge. (Already done in greedy)
        2. If stuck, try a swap: Remove edge e1, add edge e2, e3? 
           If we remove 1 and add 2, count increases by 1.
           This is complex to check for C6.
        
        Alternative: Simulated Annealing?
        Or just more greedy restarts with different orderings.
        
        Let's stick to greedy restarts but optimize the inner loop.
        """
        pass

    # Initial graph: 8 disjoint K5s
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    
    best_edges = edges[:]
    best_count = len(edges)
    
    # Try multiple random restarts with different orders
    for restart in range(1000):  # Increase restarts as we are faster
        if time.time() > deadline:
            break
        
        # Restore initial state
        adj = [0] * n
        for k in range(8):
            base = k * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    u = base + i
                    v = base + j
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        
        # Candidate edges: all non-edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    candidates.append((u, v))
        
        # Shuffle candidates to try different orders
        random.shuffle(candidates)
        
        # Greedy addition
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Check if still a non-edge
            if not ((adj[u] >> v) & 1):
                # Check if adding this edge would create a C6
                if not would_create_c6(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
            # If we found a new best, try to polish it with more specific local searches?
            # For now, just record it.
            
    return best_edges
