import time
import random
import sys

def construct():
    deadline = time.time() + 110
    n = 40
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)
    
    # Precompute C6 constraints for each potential edge
    # For edge (u,v), a C6 is u-a-b-c-d-v where all vertices distinct
    # We need to precompute which sets of other edges form a C6 with (u,v)
    
    # For each edge index e = (u,v), find all 6-cycles that include (u,v)
    # A 6-cycle through (u,v) is determined by choosing a in N(u)\{v}, d in N(v)\{u},
    # then b in N(a)\{u,v,d}, c in N(d)\{u,v,a,b} such that b-c is an edge.
    # But we need to enumerate all possible C6s through (u,v) as sets of 6 edges.
    
    # Actually, for Max-SAT / WalkSAT, we need: for each potential C6, a clause
    # saying "not all 6 edges are present". 
    # A C6 is a specific 6-cycle: v0-v1-v2-v3-v4-v5-v0.
    # The clause is: NOT(e01 AND e12 AND e23 AND e34 AND e45 AND e50)
    
    # Enumerate all 6-cycles in K40. That's a lot, but we can be smart.
    # Number of 6-cycles in K40: C(40,6) * 5! / 2 / 6... actually for a fixed set of 6 vertices,
    # there are (6-1)!/2 = 60 distinct cycles. C(40,6) = 3,838,380. So ~230M cycles. Too many.
    
    # Alternative approach: Use a different search strategy.
    # Let's use simulated annealing / stochastic local search on the edge set.
    # State: which edges are present (bitmask or boolean array of 780 bits)
    # Objective: maximize number of edges
    # Constraint: no C6 present
    
    # We can use a penalty-based approach:
    # fitness = num_edges - penalty * num_C6s
    # Then do random flips and accept based on fitness change.
    
    # But counting C6s is expensive. Let's use a different approach.
    
    # Actually, let's use a smarter greedy + local search approach:
    # 1. Build a good initial solution using multiple random greedy passes
    # 2. Do 1-swap local search: remove one edge, try to add multiple others
    # 3. Use simulated annealing to escape local optima
    
    # Fast C6 check for adding edge (u,v) to graph with adjacency list adj:
    def has_c6_fast(adj, u, v):
        """Check if adding edge (u,v) creates a C6."""
        # C6 through (u,v): u-a-b-c-d-v
        # a in N(u), d in N(v), b in N(a), c in N(d), b-c is edge
        # all distinct: u,v,a,b,c,d all different
        nu = adj[u]
        nv = adj[v]
        
        # For each a in N(u) \ {v}
        a_iter = nu & ~(1 << v)
        while a_iter:
            a_bit = a_iter & -a_iter
            a = a_bit.bit_length() - 1
            a_iter ^= a_bit
            
            # For each d in N(v) \ {u, a}
            d_iter = nv & ~(1 << u) & ~(1 << a)
            while d_iter:
                d_bit = d_iter & -d_iter
                d = d_bit.bit_length() - 1
                d_iter ^= d_bit
                
                # b in N(a) \ {u, v, d}
                # c in N(d) \ {u, v, a}
                # b-c is an edge, b != c
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Check if any b in b_mask is connected to any c in c_mask
                b_iter = b_mask
                while b_iter:
                    b_bit = b_iter & -b_iter
                    b = b_bit.bit_length() - 1
                    b_iter ^= b_bit
                    if (adj[b] & c_mask) != 0:
                        return True
        
        return False
    
    def count_c6s(adj):
        """Count number of C6s in the graph. Expensive but useful for annealing."""
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    # Check if removing (u,v) would make it C6-free locally
                    # i.e., count C6s that include edge (u,v)
                    # But this double counts. Let's just count all C6s.
                    pass
        # This is too slow. Let's skip and use a different approach.
        return 0
    
    # Simulated annealing on edge set
    def simulated_annealing(initial_adj, deadline):
        """
        Start from initial_adj, try to improve by:
        - Adding edges that don't create C6
        - Removing edges to enable adding more
        """
        adj = initial_adj[:]
        edge_count = sum(bin(x).count('1') for x in adj) // 2
        best_adj = adj[:]
        best_count = edge_count
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < deadline:
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Random move
            move_type = random.random()
            
            if move_type < 0.7:
                # Try to add an edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    continue
                
                if not has_c6_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
                    if edge_count > best_count:
                        best_count = edge_count
                        best_adj = adj[:]
                else:
                    # Edge would create C6. With probability T, accept anyway (then we'd have C6, bad)
                    # Actually we can't accept invalid states. So just skip.
                    pass
            elif move_type < 0.9:
                # Try to remove an edge and add another
                # Pick a random edge to remove
                edges_in_graph = []
                for u in range(n):
                    v_iter = adj[u] & (~((1 << (u+1)) - 1))  # only v > u
                    while v_iter:
                        v_bit = v_iter & -v_iter
                        v = v_bit.bit_length() - 1
                        v_iter ^= v_bit
                        edges_in_graph.append((u, v))
                
                if not edges_in_graph:
                    continue
                
                ru, rv = random.choice(edges_in_graph)
                
                # Remove edge (ru, rv)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                edge_count -= 1
                
                # Try to add a random edge
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                if (adj[u2] >> v2) & 1:
                    # Edge already present, just keep the removal
                    pass
                else:
                    if not has_c6_fast(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        edge_count += 1
                    else:
                        # Can't add, so put back the removed edge
                        adj[ru] |= (1 << rv)
                        adj[rv] |= (1 << ru)
                        edge_count += 1
                        continue
                
                if edge_count > best_count:
                    best_count = edge_count
                    best_adj = adj[:]
            else:
                # Try 1-swap: remove one edge, try to add two
                edges_in_graph = []
                for u in range(n):
                    v_iter = adj[u] & (~((1 << (u+1)) - 1))
                    while v_iter:
                        v_bit = v_iter & -v_iter
                        v = v_bit.bit_length() - 1
                        v_iter ^= v_bit
                        edges_in_graph.append((u, v))
                
                if not edges_in_graph:
                    continue
                
                ru, rv = random.choice(edges_in_graph)
                
                # Remove edge (ru, rv)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                edge_count -= 1
                
                added = 0
                for _ in range(2):
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 == v2:
                        continue
                    if u2 > v2:
                        u2, v2 = v2, u2
                    if (adj[u2] >> v2) & 1:
                        continue
                    if not has_c6_fast(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        edge_count += 1
                        added += 1
                
                if edge_count > best_count:
                    best_count = edge_count
                    best_adj = adj[:]
                elif added == 0:
                    # Put back the removed edge
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    edge_count += 1
        
        return best_adj, best_count
    
    # Greedy build with random ordering
    def greedy_build(adj, order, deadline):
        added = 0
        for u, v in order:
            if time.time() > deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if not has_c6_fast(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added
    
    # Start with a good base: 8 disjoint K5s
    base_adj = [0] * n
    for k in range(8):
        start = k * 5
        for a in range(5):
            for b in range(a + 1, 5):
                u = start + a
                v = start + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
    
    best_adj = base_adj[:]
    best_count = sum(bin(x).count('1') for x in best_adj) // 2
    
    random.seed(42)
    
    # Multiple greedy passes with different orderings
    while time.time() < deadline - 20:
        adj = base_adj[:]
        order = all_pairs[:]
        random.shuffle(order)
        order = [(u,v) for u,v in order if not ((adj[u] >> v) & 1)]
        
        added = greedy_build(adj, order, deadline - 20)
        current_count = sum(bin(x).count('1') for x in adj) // 2
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
    
    # Simulated annealing from best
    if time.time() < deadline - 5:
        anneal_deadline = deadline - 2
        best_adj, best_count = simulated_annealing(best_adj, anneal_deadline)
    
    # Final greedy polish
    if time.time() < deadline - 1:
        adj = best_adj[:]
        order = all_pairs[:]
        random.shuffle(order)
        order = [(u,v) for u,v in order if not ((adj[u] >> v) & 1)]
        greedy_build(adj, order, deadline - 0.5)
        current_count = sum(bin(x).count('1') for x in adj) // 2
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
    
    # Convert to edge list
    edges = []
    for u in range(n):
        v_iter = best_adj[u] & (~((1 << (u+1)) - 1))
        while v_iter:
            v_bit = v_iter & -v_iter
            v = v_bit.bit_length() - 1
            v_iter ^= v_bit
            edges.append([u, v])
    
    return edges
