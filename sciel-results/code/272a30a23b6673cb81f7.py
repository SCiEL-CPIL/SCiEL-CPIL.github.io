import time
import random
import sys

def construct():
    n = 40
    deadline = 115.0
    start_time = time.time()
    
    # Bitset representation of adjacency
    adj = [0] * n
    
    # A2[i] is an integer where bit j is 1 if there is a path of length 2 from i to j
    # A2[i] = OR of adj[k] for all k in adj[i]
    A2 = [0] * n
    
    def update_A2():
        for i in range(n):
            mask = 0
            a = adj[i]
            while a:
                lsb = a & -a
                k = lsb.bit_length() - 1
                mask |= adj[k]
                a ^= lsb
            A2[i] = mask

    def check_c6_add(u, v):
        """
        Returns True if adding (u,v) creates a C6.
        A C6 is formed if there is a path of length 3 between u and v in the current graph.
        Path u-x-y-z-v.
        This is equivalent to: exists x in N(u) such that exists y in N(x) such that y in N(v)?
        No, that's length 3 from u to v?
        Let's trace: u-x (1), x-y (2), y-z (3), z-v (4). That's length 4.
        Wait. C6 is u-v-...-u.
        If we add edge (u,v), a C6 exists if there is a path of length 5 between u and v.
        Path: u-x1-x2-x3-x4-v.
        
        Let's re-evaluate the check in the seed.
        Seed check:
        for x in N(u):
            for w in N(v):
                if adj[x] & A2[w]: return True
        
        Let's trace this logic.
        x is neighbor of u. w is neighbor of v.
        adj[x] & A2[w] != 0 means there exists a common neighbor between x and some neighbor of w?
        A2[w] is the set of nodes at distance 2 from w.
        So adj[x] & A2[w] != 0 means there is a node z such that z is neighbor of x AND z is distance 2 from w.
        z is neighbor of x: u-x-z (length 2 from u).
        z is distance 2 from w: w-y-z (length 2 from w).
        So we have path u-x-z-y-w-v.
        Length: u-x (1), x-z (2), z-y (3), y-w (4), w-v (5).
        Total length 5. Adding (u,v) closes the cycle of length 6.
        Yes, the seed's check is correct for C6.
        
        However, checking C4 is simpler and often a stronger constraint for density.
        A C4 is formed if there is a path of length 3 between u and v.
        Path u-x-y-v.
        Check: exists x in N(u) such that x in A2[v]?
        A2[v] is nodes at distance 2 from v.
        If x in N(u) and x in A2[v], then there is a path v-y-x (length 2) and u-x (length 1).
        Total path u-x-y-v has length 3. Adding (u,v) creates C4.
        """
        # Check C4 first (faster)
        # If adding (u,v) creates C4, it's invalid.
        # C4 check: N(u) & A2[v] != 0
        if adj[u] & A2[v]:
            return True
        if adj[v] & A2[u]:
            return True
            
        # Check C6
        a_u = adj[u]
        a_v = adj[v]
        
        x_mask = a_u
        while x_mask:
            lsb_x = x_mask & -x_mask
            x = lsb_x.bit_length() - 1
            x_mask ^= lsb_x
            
            w_mask = a_v
            while w_mask:
                lsb_w = w_mask & -w_mask
                w = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                
                # Check if there is a path of length 3 between x and w
                # This is equivalent to N(x) & A2[w] != 0
                if adj[x] & A2[w]:
                    return True
        
        return False

    # 1. Initial Graph: Start with a C4-free structure to ensure safety.
    # A disjoint union of K5s is C4-free and C6-free.
    # 8 * K5 = 40 vertices.
    # Edges: 8 * 10 = 80 edges.
    
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    update_A2()
    
    best_edges = sum(bin(a).count("1") for a in adj) // 2
    best_adj = adj[:]
    
    # 2. Greedy Add with Backtracking/Restart
    # We will try to add edges greedily.
    # To avoid getting stuck in a bad local optimum, we will use a randomized greedy approach
    # with multiple restarts if we hit a dead end or time limit.
    
    def try_greedy_add(adj_state, A2_state):
        """
        Tries to add edges greedily. Returns the new edge count.
        """
        edges = sum(bin(a).count("1") for a in adj_state) // 2
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Calculate "score" for each potential edge to prioritize.
            # Score: Number of constraints it satisfies? 
            # Or simply: Try all, pick one at random from valid ones?
            # Better: Pick the edge that is "most likely" to be valid or "least restrictive".
            # Actually, just iterating and adding the first valid one found is fast.
            # To break symmetry, we shuffle the order of vertices.
            
            valid_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_state[u] >> v) & 1:
                        continue
                    if not check_c6_add_state(u, v, adj_state, A2_state):
                        valid_edges.append((u, v))
                        
            if not valid_edges:
                break
                
            # Pick a random valid edge to add
            # This randomization helps escape specific local traps
            u, v = random.choice(valid_edges)
            
            # Add edge
            adj_state[u] |= (1 << v)
            adj_state[v] |= (1 << u)
            
            # Update A2
            # Efficient update:
            # A2[u] |= adj_state[v]
            # A2[v] |= adj_state[u]
            # For all x in N(u): A2[x] |= adj_state[v]
            # For all y in N(v): A2[y] |= adj_state[u]
            # Note: We must be careful with order if we update in place.
            # Since we are adding an edge, the new A2 values are supersets of old ones.
            # We can just do the full recompute for simplicity if it's fast enough,
            # or implement the delta update.
            
            # Delta update for A2:
            # New adj[u] includes v. New adj[v] includes u.
            # A2[u] = OR(adj[k] for k in N(u)). N(u) gains v.
            # So A2[u] |= adj[v] (old adj[v] + u? No, adj[v] is current state which includes u? 
            # Wait, if we update adj first, adj[v] includes u.
            # Let's save old neighbors.
            
            # Simpler: Just recompute A2 for all nodes.
            # O(N * D) ~ 40 * 10 = 400 operations. Very fast.
            for i in range(n):
                mask = 0
                a = adj_state[i]
                while a:
                    lsb = a & -a
                    k = lsb.bit_length() - 1
                    mask |= adj_state[k]
                    a ^= lsb
                A2_state[i] = mask
                
            edges += 1
            changed = True
            
        return edges

    def check_c6_add_state(u, v, adj_s, A2_s):
        # Check C4
        if adj_s[u] & A2_s[v]:
            return True
        if adj_s[v] & A2_s[u]:
            return True
            
        # Check C6
        a_u = adj_s[u]
        a_v = adj_s[v]
        
        x_mask = a_u
        while x_mask:
            lsb_x = x_mask & -x_mask
            x = lsb_x.bit_length() - 1
            x_mask ^= lsb_x
            
            w_mask = a_v
            while w_mask:
                lsb_w = w_mask & -w_mask
                w = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                
                if adj_s[x] & A2_s[w]:
                    return True
        return False

    # Main Loop: Multiple Greedy Runs with Perturbation
    # We start from the C4-free seed.
    # We run greedy add until stuck.
    # Then we perturb (remove a few edges) and try again.
    
    current_adj = adj[:]
    current_A2 = A2[:]
    
    # Initial greedy run
    edges = try_greedy_add(current_adj, current_A2)
    if edges > best_edges:
        best_edges = edges
        best_adj = current_adj[:]
        
    # Perturbation Loop
    while time.time() < deadline - 2:
        # Perturb: Remove 3-5 random edges
        edges_list = []
        for i in range(n):
            a = current_adj[i]
            while a:
                lsb = a & -a
                j = lsb.bit_length() - 1
                if i < j:
                    edges_list.append((i, j))
                a ^= lsb
                
        if not edges_list:
            break
            
        num_remove = random.randint(3, 5)
        to_remove = random.sample(edges_list, min(num_remove, len(edges_list)))
        for u, v in to_remove:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
        # Update A2 after removal
        for i in range(n):
            mask = 0
            a = current_adj[i]
            while a:
                lsb = a & -a
                k = lsb.bit_length() - 1
                mask |= current_adj[k]
                a ^= lsb
            current_A2[i] = mask
            
        # Try greedy add again
        new_edges = try_greedy_add(current_adj, current_A2)
        
        if new_edges > best_edges:
            best_edges = new_edges
            best_adj = current_adj[:]
            
        # If we didn't improve, we stay at the perturbed state for next iteration
        # (This is effectively ILS)
        
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        a = best_adj[u]
        while a:
            lsb = a & -a
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            a ^= lsb
            
    return edges
