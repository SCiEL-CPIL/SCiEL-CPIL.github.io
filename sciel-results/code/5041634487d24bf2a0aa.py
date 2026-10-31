import time
import sys
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy:
    1. Use a random lift of the Petersen graph (base girth 5) to create a 3-regular graph.
       A random lift of size m (multiplier) has n = 10*m vertices.
       The girth of a random lift is typically O(log n).
       For n=936, m=93.6 is not an integer. 
       Wait, Petersen has 10 vertices. A lift of size m has 10m vertices.
       936 is not divisible by 10. 
       The best known graph has 936 vertices.
       Maybe we should target n close to 936 that is divisible by 10? 
       Or use a different base?
       
       The directive says: "Use a random lift of the Petersen graph... to construct a 3-regular graph with a guaranteed girth of at least 14".
       If we use m=94, n=940.
       If we use m=93, n=930.
       Let's try n=940 (m=94) or n=930 (m=93).
       Actually, the "guaranteed girth of at least 14" claim for a random lift is a bit strong without specific parameters, but random lifts often have high girth.
       Let's assume we can generate a graph with girth >= 14 and then fix it to 16.
       
       However, if n=936 is the target, and we can't hit it exactly with Petersen lifts, we might have to use a different base or accept n close to 936.
       The score is seed/value. If seed is ~2800 (implied by score 3.0 for 936), then:
       If we produce 940, score = 2808/940 = 2.98 (slightly worse).
       If we produce 930, score = 2808/930 = 3.02 (slightly better).
       
       Let's try to generate a graph of size 930 or 940 using Petersen lifts and see if we can reach girth 16.
       
       Alternative: Use a different base graph?
       The directive specifically asks for Petersen lift.
       
       Let's implement:
       1. Generate random lift of Petersen with m=93 (n=930) and m=94 (n=940).
       2. Check girth.
       3. If girth < g, perform local search (2-swaps) to increase girth.
       4. Return the best graph found.
       
       Note: 2-swaps on a graph with girth 14 to reach 16 is feasible if the number of 14-cycles is small.
       
       Let's refine the "guaranteed girth" part. 
       Actually, a random lift of Petersen with m=93 might have girth ~ 10-12. 
       The directive says "guaranteed girth of at least 14". This might refer to a specific construction or a high-probability event for larger m? 
       Or maybe I should use a larger base? No, Petersen is the smallest.
       
       Let's just implement the lift and the local search.
       If the lift has girth 14, we need to break 14-cycles.
       
       Local Search Algorithm:
       1. Find a shortest cycle C.
       2. If len(C) >= g, done.
       3. Pick an edge e in C.
       4. Find another edge f such that swapping e and f maintains 3-regularity and breaks C.
          Swapping edges (u,v) and (x,y) creates (u,x) and (v,y) or (u,y) and (v,x).
          We need to ensure no multiple edges or loops.
          We prefer swaps that break the shortest cycle.
       5. Repeat until girth >= g or timeout.
       
       To be safe, I will try multiple random lifts and pick the one with the highest initial girth, then refine.
       
       Since 936 is not a multiple of 10, I will try n=930 and n=940.
       If I can find a valid graph at 930, score improves slightly over 936 (if seed is fixed to 936-equivalent).
       Wait, the "seed" in the score calculation is the value of the *seed program*.
       The prompt says "1.0 equals the seed program".
       If the seed program produced 2808 (as inferred), then:
       Current program (score 3.0) produces 936? 
       If current program produces 936, score = 2808/936 = 3.0.
       If I produce 930, score = 2808/930 = 3.02.
       If I produce 940, score = 2808/940 = 2.98.
       
       So 930 is better.
       
       Let's implement the Petersen lift for m=93 (n=930).
       
       Construction of Random Lift:
       Base graph G0 (Petersen) has vertices 0..9.
       Lift has vertices (v, i) where v in 0..9, i in 0..m-1.
       For each edge (u, v) in G0, and for each i in 0..m-1, we add an edge between (u, i) and (v, (i + sigma_e(i)) % m) for some random permutation sigma_e?
       Actually, a random lift is defined by random permutations sigma_e for each edge e.
       Edge (u,v) in base connects (u, i) to (v, sigma_{uv}(i)).
       To ensure the graph is undirected, we need sigma_{vu} = sigma_{uv}^{-1}.
       
       Steps:
       1. Define Petersen edges.
       2. For each edge, generate a random permutation of 0..m-1.
       3. Construct the lift.
       4. Verify girth.
       5. If girth < g, run local search.
       
       Local Search Details:
       - Maintain adjacency list.
       - Function `find_shortest_cycle()`: BFS from each node to find shortest cycle length.
       - Function `perform_swap()`: 
         - Get the shortest cycle C.
         - Pick a random edge (u,v) in C.
         - Try to swap it with a random edge (x,y) not in C (or anywhere) such that:
           - x != u, x != v, y != u, y != v
           - (u,x) and (v,y) are not existing edges
           - (u,y) and (v,x) are not existing edges
           - The swap breaks the cycle C (i.e., the new edges don't reconnect the path in C).
           - Actually, any swap of an edge in C with an edge not in C will break C, provided the new edges don't create a shorter cycle elsewhere.
           - We just need to ensure validity (simple graph).
       - Accept the swap if it doesn't create a cycle shorter than the current shortest cycle length? 
         Or just accept any valid swap? 
         Greedy: Accept if it breaks the current shortest cycle.
       
       Given time constraints (120s), and n=930, E=1395.
       BFS is fast.
       Swaps are O(1) update.
       
       Let's code this.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    if k != 3:
        # Fallback for other k, though task specifies k=3
        # We return a dummy or raise, but we assume k=3
        pass

    # Petersen Graph Definition
    # Vertices 0-9
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5 (indices 5,6,7,8,9)
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def verify_girth(n, adj, g):
        # adj is list of lists
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # BFS from each node to find shortest cycle
        # We can optimize by stopping if we find a cycle < g
        # But we need to know if ANY cycle < g exists.
        
        # Efficient girth check:
        # For each node, BFS up to depth g//2.
        # If we encounter a visited node at depth d1 and current depth d2, cycle length is d1+d2.
        # If d1+d2 < g, return False.
        
        max_depth = g // 2
        for start in range(n):
            visited_depth = [-1] * n
            visited_depth[start] = 0
            queue = deque()
            for v in adj[start]:
                queue.append((v, start, 1))
                visited_depth[v] = 1
            
            while queue:
                u, prev, depth = queue.popleft()
                if depth == max_depth:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if visited_depth[v] == -1:
                        visited_depth[v] = depth + 1
                        queue.append((v, u, depth + 1))
                    else:
                        # Found a cycle
                        # Cycle length is depth + 1 + visited_depth[v]
                        # Wait, this logic is tricky with standard BFS.
                        # Better: Store parent pointers or just use the non-backtracking walk logic from the seed?
                        # The seed used:
                        # state = (v, u)
                        # if state in visited_depth:
                        #    if depth - visited_depth[state] < g: return False
                        # This is for non-backtracking walks.
                        
                        # Let's use the seed's verification logic as it was likely correct.
                        pass
        
        # Re-implementing seed's verification for consistency
        for start in range(n):
            visited_depth = {}
            queue = deque()
            for v in adj[start]:
                queue.append((v, start, 1))
                visited_depth[(v, start)] = 1
            
            while queue:
                u, prev, depth = queue.popleft()
                if depth == max_depth:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start:
                        if depth + 1 < g:
                            return False
                        continue
                    state = (v, u)
                    if state in visited_depth:
                        if depth - visited_depth[state] < g:
                            return False
                    else:
                        visited_depth[state] = depth
                        queue.append((v, u, depth + 1))
        return True

    def generate_lift(m):
        """Generates a random lift of Petersen graph with multiplier m."""
        n = 10 * m
        # adj[u] = list of neighbors
        adj = [[] for _ in range(n)]
        
        # For each edge in Petersen, assign a random permutation
        # Edge (u, v) in base -> connects (u, i) to (v, perm[i])
        # We need to handle the inverse for the return edge.
        
        # Map base edges to indices
        base_edges = petersen_edges
        perms = {}
        for u, v in base_edges:
            p = list(range(m))
            random.shuffle(p)
            perms[(u, v)] = p
            # Inverse permutation for (v, u)
            inv_p = [0] * m
            for i, val in enumerate(p):
                inv_p[val] = i
            perms[(v, u)] = inv_p
        
        for u, v in base_edges:
            p = perms[(u, v)]
            for i in range(m):
                u_lift = u * m + i
                v_lift = v * m + p[i]
                adj[u_lift].append(v_lift)
                adj[v_lift].append(u_lift)
                
        return n, adj

    def find_shortest_cycle(adj, n):
        """Returns the length of the shortest cycle, or -1 if none."""
        # Use the non-backtracking walk logic to find min cycle length
        min_len = float('inf')
        # We only need to check up to current best or g
        # For efficiency, we can just check if girth < g.
        # But for local search, we need to know the actual girth or at least find a short cycle.
        
        # Let's just check for cycles of length < g
        # If we find one, we return its length (or just a flag).
        # To pick a cycle to break, we need the specific cycle.
        
        # Simpler: Just check if girth >= g.
        # If not, find one short cycle.
        
        # We'll use a BFS that stops at depth g//2.
        # If it finds a cycle of length L < g, return L and the cycle.
        
        for start in range(n):
            visited_depth = {}
            parent = {} # (v, prev) -> (prev, prev_prev) ?
            # Actually, storing the path is expensive.
            # We just need to know IF a cycle < g exists.
            # For the swap, we can just pick a random edge and swap?
            # No, we need to target short cycles.
            
            # Let's modify the verification to return the cycle if found.
            queue = deque()
            for v in adj[start]:
                queue.append((v, start, 1, [start, v]))
                visited_depth[(v, start)] = 1
            
            while queue:
                u, prev, depth, path = queue.popleft()
                if depth >= g // 2:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start:
                        cycle_len = depth + 1
                        if cycle_len < g:
                            return cycle_len, path + [start]
                        continue
                    state = (v, u)
                    if state in visited_depth:
                        cycle_len = depth + 1 - visited_depth[state]
                        # This cycle might not be the one involving the path stored in queue
                        # But if cycle_len < g, we have a short cycle.
                        # We don't have the exact path easily here without storing more.
                        # However, if we just want to break A short cycle, we can pick any edge from the current path `path`?
                        # No, the cycle detected here is formed by the walk from start to (v,u) and the previous visit to (v,u).
                        # The path `path` is the walk from start to u.
                        # The previous visit to (v,u) was at depth d0.
                        # The cycle is not necessarily in `path`.
                        
                        # To be robust, let's just return True/False for "has short cycle".
                        # And for the swap, we will just try random swaps and check if girth increased?
                        # That's too slow.
                        
                        # Alternative: Use the cycle found by the BFS from `start` that closes at `start`.
                        # The BFS above finds cycles that pass through `start`.
                        # If we find a cycle of length L < g passing through `start`, we have the path.
                        # Let's rely on that.
                        
                        # If no cycle through start is found < g, we move to next start.
                        # This might miss cycles not passing through start?
                        # No, every cycle passes through every node in it.
                        # So if we iterate all starts, we will find the shortest cycle.
                        
                        # But we stop at depth g//2.
                        # So we only find cycles of length <= g.
                        # If we find one < g, we return it.
                        
                        # If we don't find one < g, we continue.
                        # If we finish all starts, girth >= g.
                        
                        # So the function `verify_girth` is sufficient to check.
                        # For the swap, we need the cycle.
                        # Let's modify the BFS to store the path for the first short cycle found.
                        
                        if cycle_len < g:
                             # We don't have the path for this specific cycle easily.
                             # But we can just return a flag and let the swap be random?
                             # No, random swaps might not break the specific cycle.
                             
                             # Let's stick to cycles passing through `start`.
                             # The `v == start` case gives us the path.
                             # So we only return cycles that pass through `start`.
                             # This is fine, as we iterate all starts.
                             
                             # We already handled `v == start` above.
                             # So if we reach here, we found a cycle NOT passing through start?
                             # No, the state (v, u) was visited before.
                             # The previous visit was (v, u) at depth d0.
                             # The path from start to u is `path`.
                             # The path from start to the previous (v, u) is unknown.
                             # So we can't easily reconstruct the cycle.
                             
                             # Therefore, we ONLY use the `v == start` case for cycle extraction.
                             # This means we only find cycles that pass through `start`.
                             # Since we iterate all starts, we will find the global shortest cycle.
                             
                             # So, if `v == start` and `depth + 1 < g`, we return the cycle.
                             # Otherwise, we ignore the `state in visited_depth` case for cycle extraction,
                             # but we can still use it for the verification (return False).
                             
                             # For the local search, we only need to find ONE short cycle to break.
                             # So we can just return the first one found via `v == start`.
                             
                             # If we don't find one via `v == start`, but find one via `state in visited_depth`,
                             # we know girth < g, but we don't have the path.
                             # We can then just pick a random edge and swap?
                             # Or we can restart the BFS with a different start?
                             
                             # To keep it simple:
                             # We will modify the BFS to ONLY look for cycles through `start`.
                             # If we find one, we return it.
                             # If we don't, we move to next start.
                             # If we finish all starts without finding one < g, girth >= g.
                             
                             # This is correct.
                             
                             pass
                    else:
                        visited_depth[state] = depth
                        queue.append((v, u, depth + 1, path + [v]))
                        
        return -1, []

    def perform_swap(adj, n, edge1, edge2):
        """
        Swaps edges (u1, v1) and (u2, v2) to (u1, u2) and (v1, v2) or (u1, v2) and (v1, u2).
        Returns True if swap was performed and valid.
        """
        u1, v1 = edge1
        u2, v2 = edge2
        
        # Check validity
        # No loops
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return False
            
        # Check no multiple edges
        if v2 in adj[u1] or u2 in adj[v1]:
            return False
        if u2 in adj[u1] or v2 in adj[v1]: # Wait, adj is symmetric
            pass
            
        # Let's check properly
        if u1 in adj[u2] or v1 in adj[v2]: # These are the edges we are removing
            pass
            
        # New edges: (u1, u2) and (v1, v2)
        if u2 in adj[u1]: # Already connected? No, we are removing (u1, v1) and (u2, v2).
             # But if u1 and u2 are already connected, we create a multi-edge.
             return False
        if v2 in adj[v1]:
             return False
             
        # Perform swap
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        adj[u1].append(u2)
        adj[u2].append(u1)
        adj[v1].append(v2)
        adj[v2].append(v1)
        
        return True

    # Main Logic
    best_n = float('inf')
    best_adj = None
    
    # Try m=93 (n=930) and m=94 (n=940)
    for m in [93, 94]:
        if time.time() > deadline - 20:
            break
            
        n, adj = generate_lift(m)
        
        # Check initial girth
        if not verify_girth(n, adj, g):
            # Local Search
            improvements = 0
            for _ in range(5000): # Max iterations
                if time.time() > deadline - 5:
                    break
                    
                # Find a short cycle
                # We need a function that returns the cycle
                # Let's reuse the logic from verify_girth but return the cycle
                
                cycle_found = False
                short_cycle = None
                for start in range(n):
                    visited_depth = {}
                    queue = deque()
                    for v in adj[start]:
                        queue.append((v, start, 1, [start, v]))
                        visited_depth[(v, start)] = 1
                    
                    while queue:
                        u, prev, depth, path = queue.popleft()
                        if depth >= g // 2:
                            continue
                        for v in adj[u]:
                            if v == prev:
                                continue
                            if v == start:
                                cycle_len = depth + 1
                                if cycle_len < g:
                                    short_cycle = path + [start]
                                    cycle_found = True
                                    break
                            else:
                                state = (v, u)
                                if state in visited_depth:
                                    # Ignore for path extraction, but if we want to be thorough:
                                    # We only use cycles through start.
                                    pass
                                else:
                                    visited_depth[state] = depth
                                    queue.append((v, u, depth + 1, path + [v]))
                            
                    if cycle_found:
                        break
                
                if not cycle_found:
                    # Girth >= g
                    break
                
                # We have a short cycle `short_cycle`
                # Pick a random edge in the cycle
                # The cycle is a list of nodes [v0, v1, ..., vk] where v0=start, vk=start
                # Edges are (v0, v1), (v1, v2), ..., (v_{k-1}, vk)
                
                # Pick a random edge index
                edge_idx = random.randint(0, len(short_cycle) - 2)
                u1 = short_cycle[edge_idx]
                v1 = short_cycle[edge_idx + 1]
                
                # Pick a random other edge (u2, v2) from the graph
                # To avoid bias, pick a random node u2, then a random neighbor v2
                # Ensure (u2, v2) is not in the cycle (or at least not the same edge)
                
                swapped = False
                for _ in range(20): # Try 20 random swaps
                    u2 = random.randint(0, n - 1)
                    if not adj[u2]:
                        continue
                    v2 = random.choice(adj[u2])
                    
                    # Avoid swapping with the same edge or adjacent edges in the cycle?
                    # Just check validity in perform_swap
                    if perform_swap(adj, n, (u1, v1), (u2, v2)):
                        swapped = True
                        break
                
                if not swapped:
                    # Try a different edge in the cycle?
                    # Or just break?
                    # Let's try another edge in the cycle
                    pass
            else:
                # Timeout or no improvement
                pass
                
        # Final check
        if verify_girth(n, adj, g):
            if n < best_n:
                best_n = n
                best_adj = adj

    if best_adj is None:
        # Fallback: Return a LUW graph if we couldn't find a valid lift
        # Re-use LUW from previous code? 
        # For now, return a dummy or raise.
        # But we must return something.
        # Let's try LUW(3, 7) as fallback.
        
        def luw_construct(k, g, d_override=None):
            if k != 3:
                return None, None
            q = 3
            d = d_override
            if d is None:
                d = 7
            
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)

            start = (0, (None,) + (0,) * d)
            index, order, edges = {start: 0}, [start], []
            for vertex in order:
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            return len(order), edges

        n_luw, edges_luw = luw_construct(3, g, 7)
        if n_luw and verify_girth(n_luw, edges_luw, 3, g):
            # Convert edges to adj
            adj_luw = [[] for _ in range(n_luw)]
            for u, v in edges_luw:
                adj_luw[u].append(v)
                adj_luw[v].append(u)
            best_n = n_luw
            best_adj = adj_luw

    if best_adj is None:
        # Last resort
        return 0, []

    # Convert adj back to edges list
    edges = []
    for u in range(best_n):
        for v in best_adj[u]:
            if u < v:
                edges.append((u, v))
                
    return best_n, edges

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
