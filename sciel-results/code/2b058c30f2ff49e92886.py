import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def check_valid(adj, n):
        """
        Check if the graph defined by adj masks on n vertices
        has no regular induced subgraph of size >= 6.
        """
        for k in range(6, n + 1):
            mask = (1 << k) - 1
            limit = 1 << n
            
            while mask < limit:
                m = mask
                first_vertex = True
                d0 = 0
                is_regular = True
                
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    d = (adj[v] & mask).bit_count()
                    
                    if first_vertex:
                        d0 = d
                        first_vertex = False
                    else:
                        if d != d0:
                            is_regular = False
                            break
                    
                    m ^= lsb
                
                if is_regular:
                    return False, mask
                
                c = mask & -mask
                r = mask + c
                mask = (((r ^ mask) >> 2) // c) | r

        return True, 0

    def find_violation(adj, n, max_k=8):
        """Find a regular induced subgraph of size 6 to max_k. Returns mask or 0."""
        for k in range(6, min(max_k, n + 1)):
            mask = (1 << k) - 1
            limit = 1 << n
            
            while mask < limit:
                m = mask
                first_vertex = True
                d0 = 0
                is_regular = True
                
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    d = (adj[v] & mask).bit_count()
                    
                    if first_vertex:
                        d0 = d
                        first_vertex = False
                    else:
                        if d != d0:
                            is_regular = False
                            break
                    
                    m ^= lsb
                
                if is_regular:
                    return mask
                
                c = mask & -mask
                r = mask + c
                mask = (((r ^ mask) >> 2) // c) | r
        
        return 0

    def flip_edge(adj, n, u, v):
        """Flip the edge between u and v."""
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def local_search(adj, n, max_iters=5000):
        """
        Local search: repeatedly find violations and flip edges to break them.
        Returns True if no violations found.
        """
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                return False
            
            mask = find_violation(adj, n)
            if mask == 0:
                return True
            
            # Get vertices in the violating subset
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Strategy: flip an edge that changes degrees to break regularity
            # For a regular subgraph, all vertices have the same degree within it.
            # Flipping an edge between two vertices in the subset changes both their degrees by 1.
            # Flipping an edge between a vertex in subset and one outside changes only one degree.
            
            best_flip = None
            best_improvement = -1
            
            # Try flipping edges within the subset
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v = verts[i], verts[j]
                    # Simulate flip
                    if adj[u] & (1 << v):
                        # Removing edge: both lose 1 degree in subset
                        du = (adj[u] & mask).bit_count() - 1
                        dv = (adj[v] & mask).bit_count() - 1
                    else:
                        # Adding edge: both gain 1 degree in subset
                        du = (adj[u] & mask).bit_count() + 1
                        dv = (adj[v] & mask).bit_count() + 1
                    
                    # Check if this breaks regularity
                    # Recompute all degrees after flip
                    is_still_regular = True
                    target_d = -1
                    for w in verts:
                        if w == u or w == v:
                            d = (adj[w] & mask).bit_count()
                            if w == u:
                                d = du
                            else:
                                d = dv
                        else:
                            d = (adj[w] & mask).bit_count()
                        
                        if target_d == -1:
                            target_d = d
                        elif d != target_d:
                            is_still_regular = False
                            break
                    
                    if not is_still_regular:
                        # Good flip - it breaks this particular violation
                        # Prefer flips that create more diversity
                        improvement = 1
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_flip = (u, v)
            
            # If no internal flip works, try flipping edges between subset and outside
            if best_flip is None:
                outside_mask = ((1 << n) - 1) ^ mask
                for u in verts:
                    om = outside_mask
                    while om:
                        lsb = om & -om
                        v = lsb.bit_length() - 1
                        if adj[u] & (1 << v):
                            du = (adj[u] & mask).bit_count() - 1
                        else:
                            du = (adj[u] & mask).bit_count() + 1
                        
                        # Check if this breaks regularity
                        is_still_regular = True
                        target_d = -1
                        for w in verts:
                            if w == u:
                                d = du
                            else:
                                d = (adj[w] & mask).bit_count()
                            
                            if target_d == -1:
                                target_d = d
                            elif d != target_d:
                                is_still_regular = False
                                break
                        
                        if not is_still_regular:
                            best_flip = (u, v)
                            break
                        om ^= lsb
                    if best_flip is not None:
                        break
            
            if best_flip is None:
                # Random flip as fallback
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    best_flip = (u, v)
                else:
                    continue
            
            flip_edge(adj, n, best_flip[0], best_flip[1])
        
        # Final check
        valid, _ = check_valid(adj, n)
        return valid

    def generate_random_graph(n, p):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Try to find a graph for n=22, 21, 20, 19...
    for n in range(22, 5, -1):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        attempts = 30 if n >= 21 else 50
        
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Start with a random graph
            p = 0.5 if attempt % 2 == 0 else 0.4
            adj = generate_random_graph(n, p)
            
            # Local search
            if local_search(adj, n):
                edges = adj_to_edges(adj, n)
                # Final verification
                valid, _ = check_valid(adj, n)
                if valid:
                    return {'n': n, 'edges': edges}
        
        # If we didn't find one for this n, try next smaller n

    # Fallback: n=5
    return {'n': 5, 'edges': []}
