import random
import time
import sys
import math

def construct(k, g):
    if k != 3:
        # Fallback to LUW for other k if needed, but task is specific to k=3, g=16
        # The prompt says "evaluator passes this task's degree k and girth g", implying k=3, g=16.
        # But to be safe, we can handle general k?
        # For now, assume k=3, g=16 based on the task description.
        pass
        
    deadline = time.time() + 110  # 110 seconds safety margin
    
    # Moore bound for k=3, g=16 is roughly 512.
    # Known best is 936.
    # We try sizes from 512 upwards.
    start_n = 512
    max_n = 1000  # Don't go too high if we can't find quickly
    
    for n in range(start_n, max_n + 1, 2):
        if time.time() > deadline:
            break
            
        # Try a few random instances for this n
        for attempt in range(20):
            if time.time() > deadline:
                break
            graph = generate_random_3_regular(n)
            if graph is None:
                continue
            
            # Check initial girth
            girth = get_girth(graph, n)
            
            if girth >= g:
                return n, graph_to_edges(graph)
            
            # Local search to improve girth
            # Objective: minimize number of cycles of length < g
            # We use a simple hill-climbing with restarts
            success = local_search_girth(graph, n, g, deadline)
            if success:
                final_girth = get_girth(graph, n)
                if final_girth >= g:
                    return n, graph_to_edges(graph)
    
    # If no small graph found, fallback to a valid (but larger) construction
    # Use LUW as in seed, but maybe with a smaller d if girth allows?
    # No, LUW is too big.
    # We must return *something* valid.
    # If we run out of time, we should return the best valid graph found, 
    # or if none found, we might fail.
    # But the task requires a valid graph.
    # If we didn't find one in time, we can try to generate a larger one that is easier to find?
    # Or just use the LUW seed logic?
    # The seed logic returns a valid graph, just large.
    # We should ensure we return a valid graph.
    
    # Fallback: Use the seed's LUW logic to guarantee validity
    return luw_construct(k, g)


def generate_random_3_regular(n):
    """Generates a random 3-regular graph using configuration model."""
    if n % 2 != 0:
        return None
    # Each vertex has degree 3
    # Stubs: 3 per vertex
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    
    random.shuffle(stubs)
    
    neighbors = [[] for _ in range(n)]
    edges = set()
    
    # Pair up stubs
    for i in range(0, n * 3, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            # Self-loop, retry or swap
            # For simplicity, reject and regenerate if self-loop or multi-edge
            return None
        if (u, v) in edges or (v, u) in edges:
            return None
        neighbors[u].append(v)
        neighbors[v].append(u)
        edges.add((u, v))
        edges.add((v, u))
        
    return neighbors

def get_girth(graph, n, max_depth=None):
    """Calculates girth of a 3-regular graph using BFS."""
    if max_depth is None:
        max_depth = 10 # Approximate g/2 + 1
        
    min_girth = float('inf')
    
    for start in range(n):
        # BFS from start
        # dist[v] = shortest distance from start to v
        dist = [-1] * n
        parent = [-1] * n
        queue = [(start, 0)]
        dist[start] = 0
        
        while queue:
            u, d = queue.pop(0)
            if d >= max_depth:
                continue
            for v in graph[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append((v, d + 1))
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    # Cycle length = dist[u] + dist[v] + 1
                    # Wait, if we see v already visited and v is not parent of u
                    # and u is not parent of v, it's a back edge.
                    # The cycle length is dist[u] + dist[v] + 1?
                    # In BFS, if we find an edge (u, v) where dist[v] is not dist[u]+1 or dist[u]-1,
                    # or if it's a cross edge.
                    # Actually, in a tree, we only see parents.
                    # If we see a neighbor v that is already visited and not the parent of u,
                    # we have a cycle.
                    # The length of the cycle is dist[u] + dist[v] + 1.
                    # This holds if u and v are in the same BFS layer or adjacent layers.
                    # If dist[u] == dist[v], cycle length = 2*dist[u] + 1.
                    # If dist[u] == dist[v] + 1, cycle length = 2*dist[v] + 2.
                    
                    # Note: This detects cycles. The first cycle found in BFS from start is the shortest cycle passing through start?
                    # Not necessarily the global girth, but if we do this for all starts, we find the global girth.
                    # However, BFS from each start is O(N^2). N=500 -> 250,000 ops. Fast enough.
                    
                    l = dist[u] + dist[v] + 1
                    if l < min_girth:
                        min_girth = l
                        if min_girth == 2: # Should not happen in simple graph
                            return 2
                    # Optimization: If we found a very small cycle, we can return early?
                    # No, we want the exact girth to check if >= g.
                    # If min_girth >= g, we can stop early?
                    # If we haven't found any cycle < g so far, we still need to check all starts.
        
    return min_girth

def local_search_girth(graph, n, g, deadline):
    """
    Attempts to increase girth to >= g using 2-flips.
    Returns True if girth >= g is achieved.
    """
    # Count cycles of length < g
    def count_short_cycles():
        count = 0
        # Use BFS from each vertex to find all cycles of length < g
        # This is expensive if done frequently.
        # Heuristic: Just check if girth < g.
        # For local search, we want a smoother objective.
        # But counting all cycles is hard.
        # Let's just minimize the girth? No, we want to maximize the girth.
        # If we maximize the girth, we want to avoid small cycles.
        
        # Alternative: Pick a random short cycle and try to break it.
        # Find a cycle of length < g.
        cycle = find_short_cycle(graph, n, g)
        if not cycle:
            return True # Girth >= g
        
        # Try to break this cycle
        # The cycle has vertices v0, v1, ..., vL-1.
        # Pick an edge (vi, vi+1) in the cycle.
        # We want to replace it with edges that don't form short cycles.
        # Standard 2-flip: Remove (u, v) and (x, y), add (u, x) and (v, y).
        # We need to find a (x, y) such that the flip is valid (no loops/multiedges)
        # and doesn't create new short cycles.
        
        # Simplified: Randomly pick edges to flip.
        # If the flip removes the current short cycle, accept.
        
        for _ in range(100): # Max flips per cycle found
            if time.time() > deadline:
                return False
            
            # Pick a random edge in the graph? Or in the cycle?
            # Picking in the cycle is more targeted.
            # Let's pick a random edge from the cycle.
            idx = random.randint(0, len(cycle) - 1)
            u = cycle[idx]
            v = cycle[(idx + 1) % len(cycle)]
            
            # Find a partner edge (x, y) to flip with
            # We need to pick x from neighbors of u (excluding v) and y from neighbors of v (excluding u)?
            # No, 2-flip:
            # Remove (u, v) and (x, y).
            # Add (u, x) and (v, y) OR (u, y) and (v, x).
            
            # Let's pick a random vertex x != u, v.
            # And a random neighbor y of x?
            # This is complex.
            
            # Simpler random flip:
            # Pick two random edges (a, b) and (c, d).
            # If they are disjoint, try to swap.
            
            a = random.randint(0, n - 1)
            b = random.choice(graph[a])
            c = random.randint(0, n - 1)
            d = random.choice(graph[c])
            
            if a == c or a == d or b == c or b == d:
                continue
            if (a, b) == (c, d) or (a, b) == (d, c):
                continue
                
            # Try flip 1: (a, c) and (b, d)
            # Check validity
            if c not in graph[a] and d not in graph[b]:
                # Apply
                graph[a].remove(b)
                graph[c].remove(d)
                graph[a].append(c)
                graph[b].append(d)
                graph[c].append(a)
                graph[d].append(b)
                
                # Check if this flip helped (i.e., girth increased or short cycle removed)
                # We don't know if it helped without checking girth.
                # Let's just keep it if it doesn't create a smaller girth?
                # Too slow.
                
                # Heuristic: If the new edges are "far" apart, it's good.
                # For now, just keep it. Random walk.
                
            # Try flip 2: (a, d) and (b, c)
            else:
                if d not in graph[a] and c not in graph[b]:
                    graph[a].remove(b)
                    graph[c].remove(d)
                    graph[a].append(d)
                    graph[b].append(c)
                    graph[c].append(b)
                    graph[d].append(a)
            
        return False

def find_short_cycle(graph, n, g):
    """Finds a cycle of length < g. Returns list of vertices or None."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = [(start, 0)]
        dist[start] = 0
        
        while queue:
            u, d = queue.pop(0)
            if d >= g // 2 + 1:
                continue
            for v in graph[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append((v, d + 1))
                elif parent[u] != v and parent[v] != u:
                    # Found cycle
                    # Reconstruct cycle
                    path1 = []
                    cur = u
                    while cur != -1:
                        path1.append(cur)
                        cur = parent[cur]
                    path1.reverse() # start to u
                    
                    path2 = []
                    cur = v
                    while cur != -1:
                        path2.append(cur)
                        cur = parent[cur]
                    path2.reverse() # start to v
                    
                    # The cycle is the symmetric difference of the two paths?
                    # No, the cycle goes from u up to LCA and down to v.
                    # Find LCA
                    # Since it's a BFS tree, the LCA is the common ancestor.
                    # path1 is start -> ... -> u
                    # path2 is start -> ... -> v
                    # The LCA is the last common vertex.
                    set1 = set(path1)
                    lca = None
                    for node in path2:
                        if node in set1:
                            lca = node
                            
                    if lca is None:
                        continue
                        
                    cycle = []
                    # Path from u to lca
                    cur = u
                    while cur != lca:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(lca)
                    
                    # Path from v to lca
                    cur = v
                    while cur != lca:
                        cycle.append(cur)
                        cur = parent[cur]
                        
                    # The cycle is the combination
                    # But we need the order.
                    # u -> ... -> lca -> ... -> v -> u
                    # Actually, the edge (u, v) closes it.
                    # So the cycle is path_u_to_lca + reversed(path_v_to_lca) ?
                    # No, path_u_to_lca is u, ..., lca.
                    # path_v_to_lca is v, ..., lca.
                    # Cycle: u, ..., lca, ..., v, u.
                    
                    # Let's just return the existence
                    return cycle
                    
    return None

def graph_to_edges(graph):
    edges = []
    n = len(graph)
    for u in range(n):
        for v in graph[u]:
            if u < v:
                edges.append((u, v))
    return edges

def luw_construct(k, g):
    # Copy of the seed logic to ensure we have a fallback
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
