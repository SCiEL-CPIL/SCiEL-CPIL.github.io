import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Target n=46, min_degree = 7
    # If that fails, try n=45, min_degree = 6, etc.
    
    for n in [46, 45, 44, 43, 42, 41, 40, 39]:
        min_deg_required = n - 39
        if min_deg_required < 0:
            min_deg_required = 0
            
        # Try to find a valid graph for this n
        found = False
        attempts = 0
        while time.time() < deadline and not found:
            attempts += 1
            # Random initialization
            # Expected degree ~ min_deg_required
            p = min_deg_required / (n - 1) if n > 1 else 0
            
            # Use bitsets for adjacency
            # adj[i] is an integer representing the set of neighbors of i
            adj = [0] * n
            edges = []
            
            # Generate random graph
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
            
            # Helper to check if adding (u, v) creates a C4
            # A C4 is created if there is a path u-w-x-v in current graph
            # i.e., exists w in N(u), x in N(v) such that (w, x) is an edge
            def creates_c4(u, v, adj):
                # N(u) and N(v) are bitsets
                nu = adj[u]
                nv = adj[v]
                
                # Iterate through neighbors of u
                # We need to check if any neighbor of a neighbor of u is in N(v)
                # Let's iterate w in N(u)
                w_mask = nu
                while w_mask:
                    # Get lowest set bit
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    
                    # Check if N(w) intersects N(v)
                    # If yes, there is an edge (w, x) with x in N(v)
                    # This forms u-w-x-v
                    if adj[w] & nv:
                        return True
                return False

            # Helper to find a C4 in the graph
            # Returns (u, v, w, x) or None
            def find_c4(adj, n):
                # For each pair of non-adjacent vertices (u, v), check if they have a common neighbor? No, that's C3.
                # C4: u-w-x-v-u.
                # We can iterate over edges (w, x) and check if N(w) \ {x} and N(x) \ {w} have a common neighbor?
                # No, that finds C3s sharing an edge?
                
                # Better: Iterate over all pairs of vertices (u, v).
                # If u and v are not adjacent, check if they have a common neighbor? No.
                
                # Let's iterate over all pairs of edges (a,b) and (c,d) that are disjoint?
                # Too slow.
                
                # Efficient way:
                # For each vertex u, look at its neighbors N(u).
                # For each pair of neighbors w, x in N(u), check if N(w) and N(x) share a common neighbor v != u.
                # If so, u-w-v-x-u is a C4.
                
                for u in range(n):
                    nu = adj[u]
                    # Get list of neighbors
                    neighbors = []
                    mask = nu
                    while mask:
                        lsb = mask & -mask
                        w = lsb.bit_length() - 1
                        neighbors.append(w)
                        mask ^= lsb
                    
                    # Check pairs in neighbors
                    len_n = len(neighbors)
                    for i in range(len_n):
                        w = neighbors[i]
                        for j in range(i + 1, len_n):
                            x = neighbors[j]
                            # Check if N(w) and N(x) intersect at a vertex v != u
                            # N(w) & N(x) & ~(1 << u)
                            common = adj[w] & adj[x] & ~(1 << u)
                            if common:
                                # Found C4: u-w-v-x-u
                                # v is any bit in common
                                lsb = common & -common
                                v = lsb.bit_length() - 1
                                return (u, w, v, x)
                return None

            # Main repair loop
            valid = True
            while valid and time.time() < deadline:
                # Check for C4
                c4 = find_c4(adj, n)
                if c4:
                    # Remove one edge from the C4
                    # Edges are (u,w), (w,v), (v,x), (x,u)
                    u, w, v, x = c4
                    # Pick a random edge to remove
                    # e.g., (u, w)
                    # Check if removing it is safe? Yes, removing edges never creates C4s.
                    # But we must maintain min degree.
                    # If deg(u) <= min_deg_required or deg(w) <= min_deg_required, we might violate min degree.
                    # Let's pick an edge where both endpoints have degree > min_deg_required if possible.
                    candidates = []
                    for a, b in [(u, w), (w, v), (v, x), (x, u)]:
                        deg_a = bin(adj[a]).count('1')
                        deg_b = bin(adj[b]).count('1')
                        if deg_a > min_deg_required and deg_b > min_deg_required:
                            candidates.append((a, b))
                    
                    if candidates:
                        a, b = random.choice(candidates)
                        # Remove edge (a, b)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        # Update edges list? We can rebuild at the end.
                    else:
                        # Must remove an edge, even if it drops degree below required
                        # We will try to fix degrees later
                        a, b = (u, w)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                    continue # Re-check C4s immediately

                # No C4 found, check min degree
                min_deg = min(bin(adj[i]).count('1') for i in range(n))
                if min_deg >= min_deg_required:
                    # Success!
                    # Reconstruct edges list
                    final_edges = []
                    for u in range(n):
                        mask = adj[u]
                        while mask:
                            lsb = mask & -mask
                            v = lsb.bit_length() - 1
                            if u < v:
                                final_edges.append([u, v])
                            mask ^= lsb
                    return {'n': n, 'edges': final_edges}
                
                # Need to increase degrees
                # Find vertex with min degree
                v = min(range(n), key=lambda i: bin(adj[i]).count('1'))
                deg_v = bin(adj[v]).count('1')
                
                if deg_v >= min_deg_required:
                    # This shouldn't happen if min_deg < required
                    break
                
                # Try to add an edge from v to a non-neighbor u
                # such that creates_c4(v, u) is False
                # and deg(u) > min_deg_required (to be safe, though adding edge increases deg)
                # Actually, adding edge increases deg(v) and deg(u).
                
                non_neighbors = []
                for u in range(n):
                    if u == v:
                        continue
                    if not (adj[v] & (1 << u)):
                        non_neighbors.append(u)
                
                random.shuffle(non_neighbors)
                added = False
                for u in non_neighbors:
                    if not creates_c4(v, u, adj):
                        # Add edge (v, u)
                        adj[v] |= (1 << u)
                        adj[u] |= (1 << v)
                        added = True
                        break
                
                if not added:
                    # Stuck. Cannot add any edge to v without creating C4.
                    # Try to remove an edge elsewhere to "relax" the graph?
                    # Or just restart.
                    break

            if time.time() > deadline - 10:
                break

        if found:
            break

    # Fallback: return trivial solution
    return {'n': 39, 'edges': []}

