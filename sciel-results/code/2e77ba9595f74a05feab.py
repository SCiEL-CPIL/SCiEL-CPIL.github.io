import random
import time
import numpy as np

def construct():
    N = 40
    # 150s limit, use 145s for safety
    deadline = time.time() + 145.0
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def time_left():
        return deadline - time.time()

    def popcount(x):
        return x.bit_count()

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        A C6 containing (u, v) looks like: u-a-b-c-d-v.
        This requires:
        1. a in N(u) \ {v}
        2. d in N(v) \ {u}
        3. a != d
        4. Path a-b-c-d of length 3 in G \ {u, v}
        
        We check for existence of a, d such that there is a path of length 3 between them avoiding u, v.
        Path a-b-c-d implies:
        - b in N(a) \ {u, v, d}
        - c in N(d) \ {u, v, a}
        - b connected to c? No, b-c is the middle edge.
        Actually, path a-b-c-d means:
        - b is neighbor of a
        - c is neighbor of d
        - b is neighbor of c
        - All distinct from u, v.
        
        So we need:
        Exists a in N(u)\{v}, d in N(v)\{u}, a!=d
        Such that exists b in N(a)\{u,v,d}, c in N(d)\{u,v,a}
        With (b,c) in E and b!=c and b!=a and c!=d and b!=v and c!=u etc.
        
        Optimized check:
        Iterate a in N(u)\{v}
        Iterate d in N(v)\{u} \ {a}
        Let Na = N(a) \ {u, v, d}
        Let Nd = N(d) \ {u, v, a}
        We need an edge (b, c) with b in Na, c in Nd, b != c.
        This is equivalent to checking if any b in Na has a neighbor in Nd \ {b}.
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        # Iterate over the smaller set to reduce outer loop
        if popcount(Nu) > popcount(Nv):
            Nu, Nv = Nv, Nu
            # Note: The logic is symmetric for a and d? 
            # Cycle is u-a-b-c-d-v. 
            # If we swap u and v, it becomes v-d-c-b-a-u. 
            # The check is symmetric.
            pass

        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            # a is in N(u). Check if a is also in N(v)? 
            # If a == d, it's not a valid C6 (would be a triangle or shorter). 
            # We require a != d.
            
            Na_base = adj[a] & ~(1 << u) & ~(1 << v)
            
            # d must be in N(v) and d != a
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                # Na: neighbors of a, excluding u, v, d
                Na = Na_base & ~(1 << d)
                if Na == 0:
                    continue
                
                # Nd: neighbors of d, excluding u, v, a
                Nd = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if Nd == 0:
                    continue
                
                # Check if there is an edge (b, c) with b in Na, c in Nd, b != c
                # Iterate b in Na
                mask_b = Na
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    # b must be connected to some c in Nd
                    # c cannot be b
                    if adj[b] & Nd & ~(1 << b):
                        return True
        return False

    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        
        adj = [0] * N
        edge_set = set()
        edges = []
        
        # Randomize edge order
        current_edges = all_edges[:]
        random.shuffle(current_edges)
        
        for u, v in current_edges:
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                edges.append([u, v])
                
        return adj, edge_set, edges

    def local_search(adj, edge_set, edges, max_iters=500):
        """
        Try to improve the graph by adding edges or swapping.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters and time_left() > 2.0:
            improved = False
            iters += 1
            
            # 1. Try pure addition
            # Pick random non-edges
            for _ in range(50):
                if time_left() < 2.0: break
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v: continue
                if adj[u] & (1 << v): continue
                
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try 1-swap: remove one edge, try to add two? Or just add one?
            # Standard 1-swap: Remove (ru, rv), try to add (u, v).
            # This keeps edge count same. To increase, we need a 2-swap or find a "free" add after removal.
            # Let's try: Remove (ru, rv). Then try to add (u, v). If successful, count is same.
            # Then try to add ANOTHER edge (w, z). If successful, count increases by 1.
            
            if len(edges) == 0:
                break
                
            idx = random.randint(0, len(edges) - 1)
            ru, rv = edges[idx]
            key = (ru, rv)
            
            # Remove
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
            edge_set.discard(key)
            edges.pop(idx)
            
            # Try to add a new edge
            added_new = False
            for _ in range(100):
                if time_left() < 2.0: break
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v: continue
                if adj[u] & (1 << v): continue
                
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
                    added_new = True
                    break
            
            if not added_new:
                # Restore original
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                edge_set.add(key)
                edges.append([ru, rv])
                continue
            
            # Now we have swapped (ru,rv) for (u,v). Edge count is same.
            # Try to add one more edge to get net +1
            for _ in range(50):
                if time_left() < 2.0: break
                w = random.randint(0, N - 1)
                z = random.randint(0, N - 1)
                if w == z: continue
                if adj[w] & (1 << z): continue
                
                if not would_create_c6(adj, w, z):
                    adj[w] |= (1 << z)
                    adj[z] |= (1 << w)
                    edge_set.add((w, z))
                    edges.append([w, z])
                    improved = True
                    break
            
            if not improved:
                # We swapped but didn't gain. 
                # Should we keep the swap? It might be a better local minimum.
                # Let's keep it if it doesn't reduce count (it doesn't).
                # But we didn't "improve" in terms of count. 
                # To avoid cycling, we might want to revert if no gain.
                # However, the new structure might allow future gains.
                # Let's revert to be safe and strict on "improvement" for count.
                # If we don't revert, we might oscillate.
                # Let's revert.
                w, z = edges[-1]
                key2 = (w, z)
                adj[w] &= ~(1 << z)
                adj[z] &= ~(1 << w)
                edge_set.discard(key2)
                edges.pop()
                
                # Revert the swap too?
                # If we revert the add, we are back to state after swap.
                # If we revert the swap, we are back to start.
                # Let's revert the swap to ensure we only accept if we gain.
                u_swap, v_swap = edges[-1]
                key3 = (u_swap, v_swap)
                adj[u_swap] &= ~(1 << v_swap)
                adj[v_swap] &= ~(1 << u_swap)
                edge_set.discard(key3)
                edges.pop()
                
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                edge_set.add(key)
                edges.append([ru, rv])

        return adj, edge_set, edges

    # Main loop
    start_time = time.time()
    restarts = 0
    while time_left() > 5.0:
        restarts += 1
        # Use different seeds for variety
        adj, edge_set, edges = greedy_construct(seed=restarts * 101)
        count = len(edges)
        
        # Polish with local search
        if time_left() > 10.0:
            adj, edge_set, edges = local_search(adj, edge_set, edges, max_iters=200)
            count = len(edges)
        
        if count > best_count:
            best_count = count
            best_edges = edges[:]
            
        # If we hit the theoretical max or known high bound, we might break early?
        # 105 is known. 106 is the target.
        if best_count >= 106:
            # Try a few more restarts to see if we can get 107?
            # But 150s limit is tight.
            pass

    return best_edges

