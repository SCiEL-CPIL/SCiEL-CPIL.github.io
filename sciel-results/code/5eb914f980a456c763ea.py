import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s kill

    n = 40

    # Precompute all possible edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6.
        A new C6 must use the new edge (u, v).
        So we need a path of length 5 from u to v in the current graph:
        u - a - b - c - d - v, with a, b, c, d distinct and not u, v.
        
        Using bitmasks for adjacency:
        adj[i] is an integer where bit j is 1 if (i, j) is an edge.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        # a in N(u) \ {v}
        # d in N(v) \ {u}
        # a != d
        
        # Iterate over a in N(u)
        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            # Iterate over d in N(v) \ {u, a}
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                # We need b in N(a) \ {u, v, d}
                # and c in N(d) \ {u, v, a}
                # such that b != c and (b, c) is an edge.
                
                Na = adj[a]
                Nd = adj[d]
                
                # Mask for b: neighbors of a, excluding u, v, d
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                
                # Mask for c: neighbors of d, excluding u, v, a
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Check if there exists b in b_mask such that adj[b] intersects c_mask
                # Also need b != c. Since adj[b] doesn't have bit b, if c=b, adj[b]&c_mask won't have bit b set.
                # So we just check if for any b in b_mask, adj[b] & c_mask != 0.
                
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False

    def run_greedy(seed_val):
        random.seed(seed_val)
        
        # Initialize adjacency bitmasks
        adj = [0] * n
        edges = set()
        
        # Start with a sparse seed to avoid trivial C6s?
        # Strategy: Start with 8 K5s (known good base)
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add((u, v))
        
        # Generate random order of remaining edges
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges

    def local_search(edges_set, seed_val):
        """
        Try to improve the edge set by swapping or adding edges.
        Returns a new edge set if improved, otherwise the original.
        """
        random.seed(seed_val)
        
        # Build adj bitmasks
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        current_edges = set(edges_set)
        
        # List of edges to try removing
        removable = list(current_edges)
        random.shuffle(removable)
        
        # List of non-edges to try adding
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        # Try to swap: remove one edge, add two non-edges? Or just add one non-edge if possible?
        # Since we are maximizing, simple addition is best if possible.
        # But if we are stuck, maybe we need to remove an edge to add two others?
        # Let's try a simple "add if possible" pass first, then a "swap" pass.
        
        # Pass 1: Add any possible non-edge
        improved = False
        for u, v in non_edges:
            if time.time() > deadline:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
                improved = True
        
        if improved:
            return current_edges
            
        # Pass 2: Try swaps (remove 1, add 2)
        # This is expensive, so only do it if we have time.
        # We look for a pair of non-edges (e1, e2) and an edge e such that
        # G - e + e1 + e2 is C6-free.
        
        # Heuristic: Try removing an edge from a random clique or random edge,
        # then try to add back two non-edges.
        
        # To keep it fast, we'll just try a few random swaps.
        num_swaps_to_try = 50
        for _ in range(num_swaps_to_try):
            if time.time() > deadline:
                break
                
            # Pick a random edge to remove
            if not removable:
                break
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            # Save state
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            
            # Try to add two non-edges
            added = 0
            # Re-shuffle non_edges slightly or use a local slice
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added == 2:
                # We gained 1 edge
                return current_edges
            else:
                # Revert
                adj = old_adj
                current_edges = old_edges

        return current_edges

    best_edges = set()
    best_count = 0
    
    # Run multiple greedy restarts
    seed_counter = 42
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        # Run a greedy construction
        edges = run_greedy(seed_counter)
        
        # Run local search on the result
        if time.time() < deadline - 2:
            edges = local_search(edges, seed_counter + 1000)
            
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = set(edges)
            
        seed_counter += 1
        
        # If we hit 105, we might want to try harder to beat it, but 105 is the known max.
        # Actually, the task says "best known graphs have 105 edges".
        # If we find 105, we can stop early? No, maybe we can find 106? Unlikely but possible.
        # Let's keep searching until time is up.
        
    # Convert back to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
