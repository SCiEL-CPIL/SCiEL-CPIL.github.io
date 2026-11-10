import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    q = 5
    # Build PG(2, q) points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    edges = []
    adj = [0] * n  # bitmask adjacency
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_4cycles_bm(adj_list):
        """
        Count 4-cycles correctly.
        For each unordered pair {i,j}, let k = |N(i) ∩ N(j)|.
        Each 4-cycle has exactly 2 pairs of opposite vertices.
        Total 4-cycles = (1/2) * sum over all pairs {i,j} of C(k, 2)
        """
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                common = adj_list[i] & adj_list[j]
                k = popcount(common)
                if k >= 2:
                    count += k*(k-1)//2
        return count // 2
    
    # Convert to sets for the search
    adj_sets = [set() for _ in range(n)]
    for e in edges:
        u, v = e
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    base_4cycles = count_4cycles_bm(adj)
    
    # Find the best non-edge to add (minimizes new 4-cycles)
    best_edge = None
    best_cost = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            if j in adj_sets[i]:
                continue
            
            # When we add edge (i,j), new 4-cycles are created.
            # Using the formula: total_4cycles = (1/2) * sum C(|N(u)∩N(v)|, 2) over all pairs
            # Adding edge (i,j) changes N(i) and N(j).
            # For pair (i,j): k increases by... well, N(i) gains j, N(j) gains i.
            #   common neighbors of i,j: was |N(i)∩N(j)|, now |N(i)∩N(j)| (same, since j∉N(i) and i∉N(j) before)
            #   Wait, actually j is not in N(i) and i is not in N(j) before adding.
            #   After adding: N(i) has j, N(j) has i.
            #   N(i)∩N(j) after = (N(i)∩N(j)) before ∪ possibly new elements?
            #   No, the common neighbors don't change because we're only adding i to N(j) and j to N(i).
            #   So for pair (i,j): k stays the same.
            #   
            # For pair (i, a) where a is a neighbor of j: 
            #   N(i) gains j, so N(i)∩N(a) might gain elements if j∈N(a).
            #   Actually N(i)∩N(a) after = N(i)∩N(a) before ∪ {j} if j∈N(a) and j∉N(i) before
            #   Since j∉N(i) before, if j∈N(a), then j is newly in the intersection.
            #   So k increases by 1 if j∈N(a).
            #
            # For pair (j, a) where a is a neighbor of i:
            #   Similarly, k increases by 1 if i∈N(a).
            #
            # For pair (a, b) where a∈N(i), b∈N(j):
            #   N(a) unchanged, N(b) unchanged. No change.
            #
            # Actually, let me just compute the difference directly.
            # new_4cycles = count_4cycles_after - count_4cycles_before
            # This is expensive to compute exactly, so let's use an approximation or direct computation.
            
            # Direct computation: temporarily add and count
            # But that's O(n^2) per candidate, and we have ~380 non-edges, so ~380*465 ≈ 176K operations
            # That's fine.
            
            # Actually, let's use a smarter formula.
            # The change in 4-cycle count when adding edge (i,j):
            # Δ = (1/2) * [sum over all pairs of (C(k'_pair, 2) - C(k_pair, 2))]
            # where k'_pair is the new common neighbor count.
            
            # Only pairs involving i or j change.
            # Pairs involving i: (i, v) for all v != i
            # Pairs involving j: (j, v) for all v != j
            # But pair (i,j) is counted once in each, so we need to be careful.
            
            # Let's just do it directly for correctness.
            old_count = base_4cycles
            
            # Temporarily add
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            new_count = count_4cycles_bm(adj)
            
            # Revert
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
            
            cost = new_count - old_count
            
            if cost < best_cost:
                best_cost = cost
                best_edge = [i, j]
            
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break
    
    if best_edge is not None:
        u, v = best_edge
        edges.append(best_edge)
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Now do local search: try edge swaps to reduce 4-cycles
    current_count = count_4cycles_bm(adj)
    
    # Build list of edges for easier manipulation
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Local search: try edge swaps
    improved = True
    max_iterations = 100
    iteration = 0
    random.seed(42)
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        current_count = count_4cycles_bm(adj)
        
        best_swap = None
        best_net = 0  # we want negative net change (decrease in 4-cycles)
        
        edge_list = list(edge_set)
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) not in edge_set:
                    non_edges.append((i,j))
        
        if not non_edges:
            break
        
        # Try random swaps
        num_tries = min(len(edge_list) * 15, 800)
        
        for _ in range(num_tries):
            if time.time() > deadline - 30:
                break
            
            e1 = random.choice(edge_list)
            e2 = random.choice(non_edges)
            
            u1, v1 = e1
            u2, v2 = e2
            
            # Temporarily apply swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            new_count = count_4cycles_bm(adj)
            net = new_count - current_count
            
            # Revert
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)
            
            if net < best_net:
                best_net = net
                best_swap = (e1, e2)
        
        if best_swap is not None:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            
            # Apply swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edge_set.remove(e1)
            
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edge_set.add(e2)
            
            improved = True
        
        if time.time() > deadline - 30:
            break
    
    # Final validation
    final_edges = []
    seen = set()
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            final_edges.append([e[0], e[1]])
            seen.add(key)
    
    # Ensure exactly 91 distinct edges
    if len(final_edges) > 91:
        final_edges = final_edges[:91]
    elif len(final_edges) < 91:
        # Add more edges with minimal 4-cycle increase
        # Rebuild adj from final_edges
        adj_tmp = [0] * n
        for e in final_edges:
            u, v = e
            adj_tmp[u] |= (1 << v)
            adj_tmp[v] |= (1 << u)
        
        candidates = []
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) not in seen:
                    candidates.append((i,j))
        
        # Sort by cost
        for i, j in candidates:
            if len(final_edges) >= 91:
                break
            adj_tmp[i] |= (1 << j)
            adj_tmp[j] |= (1 << i)
            new_count = count_4cycles_bm(adj_tmp)
            adj_tmp[i] &= ~(1 << j)
            adj_tmp[j] &= ~(1 << i)
            
            final_edges.append([i, j])
            seen.add((i,j))
            adj_tmp[i] |= (1 << j)
            adj_tmp[j] |= (1 << i)
        
        # If still not enough, just add arbitrary
        for i in range(n):
            if len(final_edges) >= 91:
                break
            for j in range(i+1, n):
                if len(final_edges) >= 91:
                    break
                if (i,j) not in seen:
                    final_edges.append([i,j])
                    seen.add((i,j))
    
    return final_edges[:91]
