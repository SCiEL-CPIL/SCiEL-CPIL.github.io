import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    q = 5
    n = 31

    # 1. Generate 31 points of PG(2,5)
    # This specific ordering is used to ensure consistency with previous successful runs if needed,
    # but the geometry is what matters.
    points = []
    # Standard enumeration:
    # (1, x, y) for x,y in 0..4 (25 points)
    # (0, 1, z) for z in 0..4 (5 points)
    # (0, 0, 1) (1 point)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    # Verify count
    assert len(points) == n

    # 2. Construct Polarity Graph
    # Edge if dot product is 0 mod q
    adj = [0] * n  # Bitmasks for adjacency
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q

    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Count initial edges
    edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
    # For q=5, the polarity graph of PG(2,5) has 90 edges.
    # If not 90, we handle it, but it should be 90.
    
    # 3. Find the best non-edge to add to reach 91 edges
    # We want to minimize the increase in 4-cycles.
    # Adding edge (u, v) creates new 4-cycles of the form u-a-b-v-u 
    # where a is a neighbor of u, b is a neighbor of v, a != b, a != v, b != u, and a is neighbor of b.
    # Actually, the number of new 4-cycles created by adding edge (u,v) is exactly the number of
    # paths of length 2 from u to v in the existing graph? No.
    # A 4-cycle is u-a-b-v-u. 
    # If we add (u,v), any existing path u-a-b-v creates a 4-cycle.
    # The number of such paths is the number of common neighbors of u and v? 
    # Wait. If a and b are common neighbors, u-a-b-v is a path of length 3? No.
    # Path u-a-b-v means edges (u,a), (a,b), (b,v).
    # If a and b are common neighbors of u and v, then (u,a), (a,v), (v,b), (b,u) exist.
    # This forms u-a-v-b-u which is a 4-cycle.
    # So, adding (u,v) creates C(k, 2) new 4-cycles where k is the number of common neighbors?
    # Let's verify.
    # Existing graph G. Add (u,v).
    # New 4-cycles must use the new edge (u,v).
    # So the cycle is u-v-b-a-u.
    # Edges (u,v) [new], (v,b) [old], (b,a) [old], (a,u) [old].
    # So b is a neighbor of v, a is a neighbor of u, and (a,b) is an edge.
    # Also a != v and b != u.
    # Since u and v were not connected, a cannot be v (v is not neighbor of u yet) and b cannot be u.
    # So we need to count pairs (a, b) such that a in N(u), b in N(v), and (a,b) in E.
    # This is not just C(k,2). It depends on the edges between N(u) and N(v).
    
    # Let's compute this cost for each non-edge.
    best_edge = None
    best_cost = float('inf')
    
    # Precompute neighbor lists for faster intersection if needed, but bitmasks are fast.
    # To count pairs (a in N(u), b in N(v)) with edge (a,b):
    # We can iterate a in N(u), and check neighbors of a in N(v).
    
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                non_edges.append((i, j))
                
    for u, v in non_edges:
        cost = 0
        # Iterate neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & (-mask_u)
            a = lsb.bit_length() - 1
            mask_u ^= lsb
            
            # Check if a is connected to any neighbor of v
            # Neighbors of v is adj[v]
            # Common neighbors of a and v
            common_av = adj[a] & adj[v]
            # But we must exclude u? No, u is not a neighbor of v yet.
            # We must exclude v? a is neighbor of u. If a==v, then u-v is edge, which is false.
            # So a != v.
            # We count edges (a, b) where b in N(v).
            # So cost += bit_count(adj[a] & adj[v])
            cost += common_av.bit_count()
            
        if cost < best_cost:
            best_cost = cost
            best_edge = (u, v)
            
    # Add the best edge
    if best_edge:
        u, v = best_edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Now we have 91 edges.
    # 4. Local Search to minimize 4-cycles
    # We will perform swaps: remove an existing edge (u,v) and add a non-edge (x,y).
    # We want to minimize the total number of 4-cycles.
    
    def count_4cycles(adj_list):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_list[i] & (1 << j):
                    common = adj_list[i] & adj_list[j]
                    c = common.bit_count()
                    if c >= 2:
                        total += c * (c - 1) // 2
        return total // 2

    current_count = count_4cycles(adj)
    
    # Precompute edge list and non-edge list
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edge_set.add((i, j))
                
    non_edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                non_edge_set.add((i, j))

    # Function to calculate delta in 4-cycles when swapping (u,v) out and (x,y) in.
    # This is complex to do analytically fast.
    # Given the small n=31, we can just recalculate if we have time, but that's O(n^2) per swap.
    # Number of swaps to check per iteration: ~91 * ~389 ~ 35000.
    # Recalculation is ~ 31*31/2 * bit_count ~ 500 ops.
    # 35000 * 500 = 17.5M ops per iteration. Might be slow in Python.
    
    # Better approach:
    # Delta = C4(new) - C4(old)
    # C4 is sum over pairs (i,j) of C(common(i,j), 2) / 2.
    
    # Let's try a simpler heuristic local search first:
    # Try to remove an edge that is part of many 4-cycles and add a non-edge that creates few.
    
    # To speed up, we can estimate the cost of removing (u,v) and adding (x,y)
    # by counting how many 4-cycles use (u,v) and how many use (x,y).
    # But interactions matter.
    
    # Let's use a randomized local search with recalculation, but limit the number of candidates.
    
    improved = True
    iterations = 0
    max_iterations = 500 # Safety
    
    while improved and time.time() < deadline - 5 and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Sample a subset of edges to remove and non-edges to add to save time
        # Or iterate through all if fast enough.
        
        # Let's try to be smarter:
        # For each edge (u,v), calculate its "load" = number of 4-cycles it participates in.
        # Load(u,v) = sum over a in N(u), b in N(v) of [a~b and a!=b] ... wait.
        # A 4-cycle u-a-b-v-u uses edge (u,v).
        # So for a fixed edge (u,v), the number of 4-cycles containing it is the number of
        # paths of length 3 from u to v in G - (u,v)? No, in G, the other 3 edges form a path u-a-b-v.
        # So it's the number of pairs (a,b) with a in N(u), b in N(v), a~b.
        # This is exactly the cost we calculated for adding (u,v) earlier!
        
        # So, removing (u,v) reduces C4 count by load(u,v).
        # Adding (x,y) increases C4 count by load(x,y) calculated on the graph WITHOUT (u,v)?
        # Approximately.
        
        # Let's calculate load for all current edges.
        loads = {}
        for u, v in edge_set:
            load = 0
            mask_u = adj[u] & ~(1 << v)
            while mask_u:
                lsb = mask_u & (-mask_u)
                a = lsb.bit_length() - 1
                mask_u ^= lsb
                # Count neighbors of a that are also neighbors of v
                load += (adj[a] & adj[v]).bit_count()
            loads[(u, v)] = load
            
        # Find edge with max load
        if not loads:
            break
        max_load_edge = max(loads, key=loads.get)
        
        # Try to find a non-edge (x,y) that has low "potential load"
        # We can approximate the load of adding (x,y) by the cost calculated on the current graph.
        # But if we remove (u,v), the graph changes.
        
        # Simplest effective strategy:
        # Pick the edge with highest load to remove.
        # Pick the non-edge with lowest load to add.
        # Check if the swap improves the count.
        
        # Calculate loads for non-edges? That's expensive (389 non-edges * ~10 neighbors * bit_count).
        # 389 * 10 = 4000 ops. Very fast.
        
        non_edge_loads = {}
        for x, y in non_edge_set:
            load = 0
            mask_x = adj[x]
            while mask_x:
                lsb = mask_x & (-mask_x)
                a = lsb.bit_length() - 1
                mask_x ^= lsb
                load += (adj[a] & adj[y]).bit_count()
            non_edge_loads[(x, y)] = load
            
        if not non_edge_loads:
            break
            
        min_load_non_edge = min(non_edge_loads, key=non_edge_loads.get)
        
        # Attempt swap: remove max_load_edge, add min_load_non_edge
        u, v = max_load_edge
        x, y = min_load_non_edge
        
        # Apply swap
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[x] |= (1 << y)
        adj[y] |= (1 << x)
        
        new_count = count_4cycles(adj)
        
        if new_count < current_count:
            # Accept
            current_count = new_count
            edge_set.discard((u, v))
            edge_set.add((x, y))
            non_edge_set.discard((x, y))
            non_edge_set.add((u, v))
            improved = True
        else:
            # Revert
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            adj[x] &= ~(1 << y)
            adj[y] &= ~(1 << x)
            # Try a few random swaps if greedy failed?
            # For now, just break or try next best?
            # To avoid getting stuck, we can try the next best non-edge for the same edge.
            # But for simplicity and speed, we just break if the best greedy swap fails.
            # Or we can try a random swap.
            
            # Let's try a few random swaps to escape local minima
            random.shuffle(list(edge_set))
            random.shuffle(list(non_edge_set))
            
            found_improvement = False
            for e in list(edge_set)[:5]: # Try 5 random edges
                for ne in list(non_edge_set)[:5]: # Try 5 random non-edges
                    u2, v2 = e
                    x2, y2 = ne
                    
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    adj[x2] |= (1 << y2)
                    adj[y2] |= (1 << x2)
                    
                    c = count_4cycles(adj)
                    
                    if c < current_count:
                        current_count = c
                        edge_set.discard(e)
                        edge_set.add(ne)
                        non_edge_set.discard(ne)
                        non_edge_set.add(e)
                        improved = True
                        found_improvement = True
                        break
                    else:
                        # Revert
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        adj[x2] &= ~(1 << y2)
                        adj[y2] &= ~(1 << x2)
                        
                    if time.time() > deadline - 5:
                        break
                if found_improvement or time.time() > deadline - 5:
                    break
            
            if not found_improvement:
                break

    # Construct final edge list
    final_edges = [list(e) for e in edge_set]
    
    # Ensure we return exactly 91 edges
    # If something went wrong and we don't have 91, pad or trim
    if len(final_edges) < 91:
        # Add arbitrary non-edges
        for i in range(n):
            for j in range(i + 1, n):
                if len(final_edges) >= 91:
                    break
                if (i, j) not in edge_set:
                    final_edges.append([i, j])
                    edge_set.add((i, j))
            if len(final_edges) >= 91:
                break
    elif len(final_edges) > 91:
        final_edges = final_edges[:91]
        
    return final_edges
