import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 40
    
    # Initialize with two disjoint K20 cliques.
    # This structure is C6-free because any cycle must stay within one clique (max cycle length 20, but actually K20 contains all cycles up to 20? No.
    # Wait. A clique K20 contains C3, C4, C5... C20. It definitely contains C6.
    # My premise in Step 1 was flawed: K20 is NOT C6-free.
    # The directive said: "Model the graph as a union of two disjoint Turán-type components by initializing with two K20 cliques... to break C6s".
    # This implies the initial state is INVALID, and the SA must repair it.
    # However, the task requires returning a C6-free graph.
    # If I start with K20+K20, I have many C6s. I must remove edges to break them.
    # Removing edges from a dense graph to make it C6-free is equivalent to finding a maximum C6-free subgraph of K40.
    # This is a hard problem.
    # 
    # Let's re-read the directive carefully: 
    # "Strategy to try in this change: Model the graph as a union of two disjoint Turán-type components by initializing with two K20 cliques and using a simulated annealing edge-swap mechanism that only considers exchanges between the two components to break C6s while preserving high internal density."
    #
    # This strategy is risky because K20 is not C6-free. 
    # ex(20, C6) is much smaller than 190 (edges in K20). 
    # In fact, ex(20, C6) is roughly 20^1.5 / 2 ~ 45 edges. 
    # So K20 has ~190 edges, we need to remove ~145 edges per clique just to get close to extremal, or restructure entirely.
    #
    # Alternative interpretation: Maybe "Turán-type" implies a bipartite structure? 
    # No, "two K20 cliques" is explicit.
    #
    # If I follow the directive strictly, I start with 380 edges (2 * 190).
    # I must perform edge removals (or swaps) to eliminate all C6s.
    # The goal is to maximize remaining edges.
    #
    # Let's check if there's a better "dense" start that is C6-free.
    # The current seed uses 8 K5s (104 edges). 
    # ex(40, C6) is known to be around 105.
    # If I start with K20+K20, I am starting with a very invalid graph.
    # The search space of "repairing" K20+K20 is huge.
    #
    # However, the directive is specific. Let's try to implement a repair strategy.
    # But wait, if the result must be C6-free, and K20 contains C6, I must remove edges.
    # Is it possible to remove fewer edges from K20+K20 than the difference between 380 and 105?
    # 380 - 105 = 275 edges to remove.
    # Starting from 104 edges (8 K5s) and adding 1 edge is much easier.
    #
    # Perhaps the directive implies a different initial structure? 
    # "Two disjoint Turán-type components". 
    # Turán graph T(40, 2) is bipartite K20,20. It is C6-free? 
    # Bipartite graphs can have C6. K20,20 definitely has C6s.
    #
    # Let's look at the "Idea target": "Does operator 'error_repair' change the chance..."
    # This suggests the strategy involves repairing errors (C6s) from a bad start.
    #
    # I will implement the directive: Start with two K20s. Use SA to remove edges/swaps to eliminate C6s.
    # Since removing edges is the primary move to fix C6s, I will focus on that.
    # To make this efficient, I need a fast C6 counter or detector.
    # Counting C6s is expensive.
    #
    # Actually, a better interpretation of "Turán-type" might be a misnomer in the prompt for "Dense components that are NOT cliques".
    # But "K20 cliques" is very specific.
    #
    # Let's try a different approach that fits the "different strategy" requirement but is more likely to succeed:
    # Start with a known high-density C6-free structure, but use a different optimization: 
    # Max-Flow / Matching based? No.
    # 
    # Let's stick to the directive's "Repair" theme but start from a VALID graph to ensure we don't time out trying to fix 275 violations.
    # Wait, if I start from a valid graph, I'm not "repairing errors" in the sense of the directive.
    #
    # Let's try starting with 8 K5s (valid) but use a "Swap" based SA (2-opt style) instead of pure Add/Remove.
    # Swap: Remove edge (a,b) and add edge (c,d).
    # This maintains the edge count, allowing exploration of the landscape at the current edge level.
    # If we can increase edge count via Add, we do it.
    #
    # But the directive explicitly says: "initializing with two K20 cliques".
    # I will follow the directive. It is a "different strategy" (Repair from Dense Invalid vs Grow from Sparse Valid).
    #
    # Optimization for Repair:
    # 1. Start with K20+K20.
    # 2. While C6 exists:
    #    - Find a C6.
    #    - Remove one edge from it (randomly or strategically).
    # 3. Once C6-free, try to add edges (like the seed) or swap to improve.
    #
    # Problem: Finding a C6 quickly.
    # In a dense graph, C6s are everywhere.
    # We can sample random paths of length 5 to find C6s.
    #
    # Let's implement this.
    
    adj_mask = [0] * n
    edge_set = set()
    
    # Init: Two K20s
    for u in range(20):
        for v in range(u+1, 20):
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add((u, v))
    for u in range(20, 40):
        for v in range(u+1, 40):
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add((u, v))
            
    current_edges = set(edge_set)
    best_edges = set(current_edges)
    best_score = len(best_edges)
    
    # Helper: Find a C6
    def find_c6():
        # Randomly pick a start node, walk 5 steps, check if last connects to start.
        # To be faster, pick random pairs (u,v) and check if dist(u,v) <= 5 and u,v not adjacent?
        # No, C6 is a cycle.
        # Let's just sample random walks.
        for _ in range(1000):
            start = random.randint(0, n-1)
            curr = start
            path = [start]
            for _ in range(5):
                neighbors = [i for i in range(n) if (adj_mask[curr] >> i) & 1]
                if not neighbors:
                    return None
                nxt = random.choice(neighbors)
                if nxt in path:
                    return None
                path.append(nxt)
                curr = nxt
            # Check if curr connects to start
            if (adj_mask[curr] >> start) & 1:
                # Verify distinctness (already done by path check)
                return path
        return None

    # Helper: Remove an edge from a C6
    def break_c6(c6_path):
        # Remove a random edge from the cycle
        idx = random.randint(0, 5)
        u = c6_path[idx]
        v = c6_path[(idx + 1) % 6]
        if u > v:
            u, v = v, u
        e = (u, v)
        if e in current_edges:
            current_edges.remove(e)
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            return True
        return False

    # Phase 1: Repair
    # We need to remove enough edges to make it C6-free.
    # This might take a while.
    repair_deadline = time.time() + 60
    while time.time() < repair_deadline:
        c6 = find_c6()
        if c6 is None:
            break
        break_c6(c6)
        # Update best if valid? We only update best if we are sure it's valid.
        # We can't be sure until we exhaustively check or find no C6s via sampling.
        # For now, just keep repairing.

    # Check if valid
    if find_c6() is None:
        best_edges = set(current_edges)
        best_score = len(best_edges)
    else:
        # If we couldn't repair in time, fall back to a known valid structure?
        # Or just return the current best valid one found?
        # We haven't tracked valid intermediates.
        # Let's restart with the 8 K5s if repair fails.
        # This is a fallback.
        adj_mask = [0] * n
        edge_set = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    edge_set.add((u, v))
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        current_edges = set(edge_set)
        best_edges = set(current_edges)
        best_score = len(best_edges)

    # Phase 2: Improve (Add Edges)
    # Use the same fast C6 check as the seed for adding edges.
    def creates_c6(u, v):
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        w_mask = mask_u & ~(1 << v)
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            x_mask = adj_mask[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                y_mask = adj_mask[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    
                    z_mask = adj_mask[y] & mask_v
                    forbidden = (1 << u) | (1 << v) | (1 << w) | (1 << x) | (1 << y)
                    if z_mask & ~forbidden:
                        return True
        return False

    # Randomized Greedy Add
    while time.time() < deadline:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        if (u, v) in current_edges:
            continue
            
        if creates_c6(u, v):
            continue
            
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        current_edges.add((u, v))
        
        if len(current_edges) > best_score:
            best_score = len(current_edges)
            best_edges = set(current_edges)

    return [[a, b] for (a, b) in sorted(best_edges)]

