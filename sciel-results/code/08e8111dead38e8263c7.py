import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Moore bound for k=3, g=16 is 510. n must be even.
    # We will try n = 512, 514, ...
    # If we find a valid graph, return it.
    
    best_n = None
    best_edges = None
    
    # Try a range of n values. 
    # It is hard to jump between n values if we don't have a valid graph.
    # We will pick a specific n and try hard. 
    # Given the difficulty, maybe start with a slightly larger n to ensure we find *some* valid graph, 
    # then if time permits, try smaller ones? 
    # No, the score is 1/n. Smaller n is better. 
    # Let's try n=512 first. If it fails quickly, move up.
    
    # However, "quickly" is subjective. 
    # Let's allocate time slices.
    # Slice 1: n=512, 10s
    # Slice 2: n=514, 10s
    # ...
    # If we find a valid graph in any slice, we return it.
    # If we find multiple, we keep the smallest n.
    
    # Actually, if we find a valid graph for n=600, we should return it immediately if we can't guarantee finding n=512.
    # But we don't know if n=512 is possible. It is the Moore bound. It might not exist.
    # The next possible values are 514, 516...
    
    # Strategy:
    # Iterate n from 512 upwards.
    # For each n, run local search for a fixed time (e.g., 10s).
    # If success, return immediately.
    # If time runs out, return the best valid graph found in previous iterations (if any).
    # If no valid graph found at all, return the seed's result? 
    # The seed's result is valid. We can hardcode the seed's logic as a fallback?
    # No, the prompt says "rewrite the program". The seed is the current program.
    # If we fail to find anything, we should probably return the seed's output to guarantee score >= 1.0?
    # But we are replacing the seed.
    # If we return an invalid graph, score is 0.
    # So we MUST return a valid graph.
    # If our search fails, we can fallback to a known valid construction (like the LUW one from the seed) 
    # but that requires implementing it. 
    # Alternatively, we can just make our search very robust.
    
    # Let's assume we can find a valid graph for n ~ 500-1000.
    
    def generate_random_cubic_graph(n):
        """Generates a random 3-regular graph on n vertices."""
        if n % 2 != 0:
            return None
        if n < 4:
            return None
        
        adj = [[] for _ in range(n)]
        # Use a pairing model or sequential edge addition.
        # Simple method: random permutation of half-edges.
        # Half-edges: n * k / 2 = 3n/2 pairs.
        # This is tricky to implement correctly without duplicates/loops.
        
        # Better: Start with empty graph, add edges one by one.
        # Ensure degree 3 for all.
        # This is hard to do randomly without bias.
        
        # Standard method:
        # 1. Create list of all possible edges? No, too many.
        # 2. Use the "configuration model" and then resolve multi-edges/loops?
        #    For n=512, multi-edges are likely.
        
        # Let's use a simple iterative construction:
        # Start with a cycle 0-1-2-...-n-1.
        # Then add random chords to make it 3-regular.
        
        adj = [[(i-1)%n, (i+1)%n] for i in range(n)]
        # Now each vertex has degree 2. We need to add n/2 more edges to make degree 3.
        # We need to pair up the "slots" at each vertex.
        # Each vertex has one slot left.
        # We have n slots. We need to form n/2 disjoint pairs of slots, such that no pair connects a vertex to itself or an already connected vertex.
        
        slots = list(range(n))
        random.shuffle(slots)
        
        # We will greedily pair them up.
        # This might fail if we create a loop or multi-edge.
        # Retry if fail.
        
        for _ in range(1000): # max retries
            slots = list(range(n))
            random.shuffle(slots)
            adj_temp = [list(neighs) for neighs in adj]
            success = True
            for i in range(0, n, 2):
                u = slots[i]
                v = slots[i+1]
                if u == v:
                    success = False
                    break
                if v in adj_temp[u]:
                    success = False
                    break
                adj_temp[u].append(v)
                adj_temp[v].append(u)
            if success:
                return adj_temp
        return None

    def find_short_cycles(adj, n, g):
        """Find all cycles of length < g. Returns list of cycles (lists of vertices)."""
        cycles = []
        for start in range(n):
            # BFS
            # dist[v] = distance from start
            # prev[v] = previous vertex
            dist = [-1] * n
            prev = [-1] * n
            queue = [start]
            dist[start] = 0
            
            while queue:
                u = queue.pop(0)
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    elif v != prev[u] and v != start:
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        length = dist[u] + dist[v] + 1
                        if length < g:
                            # Reconstruct cycle
                            path1 = []
                            curr = u
                            while curr != -1:
                                path1.append(curr)
                                if curr == start: break
                                curr = prev[curr]
                            path1.reverse()
                            
                            path2 = []
                            curr = v
                            while curr != -1:
                                path2.append(curr)
                                if curr == start: break
                                curr = prev[curr]
                            # path2 is from v to start
                            
                            # Combine
                            # The cycle is path1 + reversed(path2[1:])? 
                            # Actually, the cycle is the path from start to u, then edge u-v, then path from v back to start.
                            # path1: start -> ... -> u
                            # path2: start -> ... -> v
                            # Cycle: u -> ... -> start -> ... -> v -> u
                            # We just need the vertices.
                            cycle_vertices = path1 + path2[1:]
                            # Remove duplicates if any? 
                            # Ensure it's a simple cycle.
                            # For small graphs, this is fine.
                            cycles.append(cycle_vertices)
                    # Also check if v is start?
                    # If v == start, then we found a cycle of length dist[u] + 1.
                    elif v == start and dist[u] + 1 < g:
                        length = dist[u] + 1
                        if length < g:
                            path = []
                            curr = u
                            while curr != -1:
                                path.append(curr)
                                if curr == start: break
                                curr = prev[curr]
                            path.reverse()
                            cycles.append(path)
        return cycles

    def apply_switch(adj, u, v, x, y):
        """
        Swaps edges (u,v) and (x,y) with (u,x) and (v,y) or (u,y) and (v,x).
        Returns True if successful, False if invalid (loop or multi-edge).
        Assumes (u,v) and (x,y) are existing edges.
        """
        # Remove existing
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        # Try option 1: (u,x) and (v,y)
        if x not in adj[u] and y not in adj[v] and u != x and v != y:
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            return True
        
        # Revert option 1 if it failed
        adj[u].remove(x)
        adj[x].remove(u)
        adj[v].remove(y)
        adj[y].remove(v)
        
        # Try option 2: (u,y) and (v,x)
        if y not in adj[u] and x not in adj[v] and u != y and v != x:
            adj[u].append(y)
            adj[y].append(u)
            adj[v].append(x)
            adj[x].append(v)
            return True
            
        # Revert option 2
        adj[u].remove(y)
        adj[y].remove(u)
        adj[v].remove(x)
        adj[x].remove(v)
        
        # Re-add original
        adj[u].append(v)
        adj[v].append(u)
        adj[x].append(y)
        adj[y].append(x)
        return False

    # Main Loop
    for n in range(512, 1000, 2):
        if time.time() > deadline - 10:
            break
            
        # Try to find a solution for this n
        # Time limit per n: 10s
        n_start = time.time()
        
        adj = generate_random_cubic_graph(n)
        if not adj:
            continue
            
        # Local Search
        while time.time() - n_start < 10.0:
            if time.time() > deadline:
                return best_n, best_edges if best_n else (n, adj) # Fallback to current? No, must be valid.
                
            cycles = find_short_cycles(adj, n, g)
            if not cycles:
                # Valid graph found!
                # Extract edges
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            else:
                # Pick a random cycle
                cycle = random.choice(cycles)
                # Pick a random edge in the cycle
                # Cycle is a list of vertices [v0, v1, ..., vk-1]
                # Edges are (v0,v1), (v1,v2), ..., (vk-1, v0)
                # Note: my cycle reconstruction might not be perfectly ordered.
                # Let's just pick two non-adjacent vertices in the cycle? 
                # No, we need to break the cycle.
                # Pick a random vertex u in cycle, and its neighbor v in cycle.
                u_idx = random.randint(0, len(cycle)-1)
                u = cycle[u_idx]
                v = cycle[(u_idx+1) % len(cycle)]
                
                # Check if (u,v) is an edge
                if v not in adj[u]:
                    # Cycle reconstruction error? 
                    # Fallback: pick random edge from whole graph that is in a short cycle?
                    # Simpler: just pick random edge from graph and try to switch?
                    # This is inefficient.
                    # Let's assume cycle is valid.
                    continue
                    
                # Find another edge (x,y) to switch with.
                # To break the cycle, we should switch with an edge NOT in the cycle?
                # Actually, any switch on an edge in the cycle will break it.
                # But we need a partner edge.
                # Pick a random edge (x,y) from the graph.
                # Ensure x,y are not u,v.
                all_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            all_edges.append((i,j))
                
                x, y = random.choice(all_edges)
                if x == u or x == v or y == u or y == v:
                    # Try another
                    continue
                    
                # Apply switch
                if apply_switch(adj, u, v, x, y):
                    # Accepted
                    pass
                else:
                    # Reverted
                    pass

    # If we reach here, we didn't find a solution in the loop.
    # We must return *something*.
    # The seed program returns a valid graph.
    # We can't easily replicate the seed's LUW graph here without copying the code.
    # But we can return the last generated graph? No, it's invalid.
    # We should have kept the best valid graph found.
    # If we found a valid graph for n=600, we returned it.
    # If we found NO valid graph, we are in trouble.
    
    # Fallback: Return a known small valid graph? 
    # The Petersen graph has girth 5. Not enough.
    # We need girth 16.
    
    # If we fail, we should probably return the seed's result.
    # But we don't have the seed's code here.
    # However, the task says "rewrite the program".
    # If I can't find a valid graph, I score 0.
    # To mitigate risk, I should make the search very aggressive.
    
    # One more try: just return a random graph? No.
    
    # Let's assume the search works.
    # If not, this is a failure case.
    
    # To be safe, let's add a hardcoded fallback for a small valid graph if we find one?
    # No, we don't know one.
    
    # Let's just return the last n and its edges, hoping it's valid?
    # No, that's bad.
    
    # I will structure the code to return the best valid graph found.
    # If none found, I will return the seed's logic result?
    # I can copy the seed's construct function as a fallback.
    
    # Let's copy the seed's logic into a helper function `seed_construct` and call it if we fail.
    
    return 0, [] # Placeholder, will be replaced by fallback logic

# Fallback logic using the seed's approach
def seed_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = None
    best_edges = None
    
    # Try random search for small n
    for n in range(512, 800, 2):
        if time.time() > deadline - 5:
            break
        # Try to solve for n
        # ... (implementation as above) ...
        # If found, update best and return immediately? 
        # Yes, because smaller n is better.
        
    # If we found a valid graph in the loop, we returned it.
    # If we didn't, we return the seed's result.
    return seed_construct(k, g)

# I will now write the full complete code combining these parts.
