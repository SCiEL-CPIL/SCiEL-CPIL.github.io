import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # State: list of integers representing adjacency bitsets
    # adj[i] is an integer with bit j set if edge (i,j) exists
    
    def is_k33_free(adj):
        """Check if graph is K3,3 free using bitsets."""
        for u in range(n):
            for v in range(u+1, n):
                uv_common = adj[u] & adj[v]
                for w in range(v+1, n):
                    # Common neighbors of u, v, w
                    count = (uv_common & adj[w]).bit_count()
                    if count >= 3:
                        return False
        return True

    def count_edges(adj):
        # Sum of bits in all adj[i] / 2
        return sum(a.bit_count() for a in adj) // 2

    def repair(adj):
        """Remove edges to make graph K3,3 free."""
        changed = True
        while changed:
            changed = False
            # Find a violating triple
            for u in range(n):
                for v in range(u+1, n):
                    uv_common = adj[u] & adj[v]
                    for w in range(v+1, n):
                        common = uv_common & adj[w]
                        if common.bit_count() >= 3:
                            # Pick a random neighbor from common to remove edge to
                            # We need to remove an edge incident to u, v, or w and a node in common
                            # Let's remove (u, x) where x is in common
                            # Find first set bit in common
                            x = (common & -common).bit_length() - 1
                            # Clear bit x in adj[u] and adj[x]
                            mask = ~(1 << x)
                            adj[u] &= mask
                            adj[x] &= mask
                            changed = True
                            break
                    if changed: break
                if changed: break
        return adj

    def try_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return False
        # Add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check validity
        # Only need to check triples involving u or v?
        # No, adding (u,v) can only create a K3,3 that includes (u,v).
        # So we only need to check if there is a K3,3 containing (u,v).
        # This is faster than full check.
        # But full check with bitsets is ~20^3/6 * 1 op = ~1300 ops.
        # Let's just do full check for safety and simplicity, it's fast enough.
        if is_k33_free(adj):
            return True
        else:
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    best_adj = None
    best_count = 0
    
    # Initialize with seed graph
    seed_adj = [0] * n
    for a in range(2):
        for b in range(2, 20):
            seed_adj[a] |= (1 << b)
            seed_adj[b] |= (1 << a)
    best_adj = seed_adj[:]
    best_count = count_edges(seed_adj)
    
    # Random restarts
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        # Generate random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.4:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        adj = repair(adj)
        curr_count = count_edges(adj)
        
        # Local Search / Greedy Add
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add random edges
            # Create list of non-edges
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        non_edges.append((i, j))
            
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if try_add_edge(adj, u, v):
                    curr_count += 1
                    improved = True
                    if curr_count > best_count:
                        best_count = curr_count
                        best_adj = adj[:]
        
        # Simulated Annealing / Swaps
        T = 1.0
        for step in range(200):
            if time.time() > deadline:
                break
            if T < 0.01:
                break
            
            # Pick a random edge to remove
            existing_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        existing_edges.append((i, j))
            
            if not existing_edges:
                break
            
            u1, v1 = random.choice(existing_edges)
            
            # Pick a random non-edge to add
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        non_edges.append((i, j))
            
            if not non_edges:
                break
            
            u2, v2 = random.choice(non_edges)
            
            # Try swap
            # Remove (u1, v1)
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            # Add (u2, v2)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            if is_k33_free(adj):
                # Accept
                pass
            else:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
            
            T *= 0.95
            
            # After some swaps, try to add edges again
            if step % 20 == 0:
                improved2 = True
                while improved2 and time.time() < deadline:
                    improved2 = False
                    non_edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if not (adj[i] & (1 << j)):
                                non_edges.append((i, j))
                    random.shuffle(non_edges)
                    for u, v in non_edges:
                        if time.time() > deadline:
                            break
                        if try_add_edge(adj, u, v):
                            curr_count += 1
                            improved2 = True
                            if curr_count > best_count:
                                best_count = curr_count
                                best_adj = adj[:]

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
