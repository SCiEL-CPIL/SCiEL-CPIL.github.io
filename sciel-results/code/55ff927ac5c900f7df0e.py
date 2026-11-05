import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    if k == 3:
        return _construct_spectral_switch(g, deadline, start_time)
    else:
        return _construct_luw(k, g)

def _construct_spectral_switch(g, deadline, start_time):
    """
    Construct a 3-regular graph with girth >= g using randomized edge-switching
    to minimize short cycles, with fast non-backtracking girth verification.
    """
    # Moore bound for (3, g): 2 * (2^(g//2 - 1) - 1) for even g, similar for odd
    # For g=16: Moore bound = 2*(2^7 - 1) = 254
    # Smallest known: 936. We'll try sizes from a reasonable lower bound upward.
    
    # Try a range of n values, starting from smaller ones
    # For 3-regular, n must be even
    candidate_ns = []
    # Start from a lower bound that's achievable
    # Moore bound for (3,16) is 254, but we need even n >= 254
    # Let's try from 256 up to 936 in steps
    n_start = 256
    n_end = 936
    step = 8  # step by 8 to keep n even and cover range
    
    for n in range(n_start, n_end + 1, step):
        if n % 2 != 0:
            n += 1
        if n % 2 != 0:
            continue
        candidate_ns.append(n)
    
    # Also include some specific values
    for n in [256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300,
              304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352,
              356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400,
              404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452,
              456, 460, 464, 468, 472, 476, 480, 484, 488, 492, 496, 500,
              504, 508, 512, 516, 520, 524, 528, 532, 536, 540, 544, 548, 552,
              556, 560, 564, 568, 572, 576, 580, 584, 588, 592, 596, 600,
              604, 608, 612, 616, 620, 624, 628, 632, 636, 640, 644, 648, 652,
              656, 660, 664, 668, 672, 676, 680, 684, 688, 692, 696, 700,
              704, 708, 712, 716, 720, 724, 728, 732, 736, 740, 744, 748, 752,
              756, 760, 764, 768, 772, 776, 780, 784, 788, 792, 796, 800,
              804, 808, 812, 816, 820, 824, 828, 832, 836, 840, 844, 848, 852,
              856, 860, 864, 868, 872, 876, 880, 884, 888, 892, 896, 900,
              904, 908, 912, 916, 920, 924, 928, 932, 936]:
        if n not in candidate_ns:
            candidate_ns.append(n)
    
    candidate_ns = sorted(set(candidate_ns))
    
    best_n = float('inf')
    best_edges = None
    
    for n in candidate_ns:
        if time.time() > deadline:
            break
        
        # Try to find a valid graph of size n
        result = _try_find_graph(n, g, deadline, start_time)
        if result is not None:
            edges = result
            best_n = n
            best_edges = edges
            break  # Found a valid graph, return it
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: LUW construction
    return _construct_luw(3, g)

def _try_find_graph(n, g, deadline, start_time):
    """
    Try to find a 3-regular graph of size n with girth >= g using edge-switching.
    Returns edges if found, None otherwise.
    """
    # Generate initial random 3-regular graph
    adj = _generate_random_3_regular(n)
    if adj is None:
        return None
    
    # Initial girth check
    girth = _compute_girth(n, adj, g)
    if girth >= g:
        return _adj_to_edges(n, adj)
    
    # Edge-switching heuristic to increase girth
    # A 2-switch: pick two edges (a,b) and (c,d), remove them, add (a,c) and (b,d)
    # This preserves 3-regularity
    
    max_iterations = 5000
    current_girth = girth
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick two random edges
        # Build edge list
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))
        
        if len(edge_set) < 2:
            break
        
        edges_list = list(edge_set)
        e1, e2 = random.sample(edges_list, 2)
        
        # Ensure edges are disjoint
        if len(set(e1) & set(e2)) > 0:
            continue
        
        # Try the switch
        a, b = e1
        c, d = e2
        
        # Remove old edges
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        
        # Add new edges
        new_edges = [(a, c), (b, d)]
        
        # Check for loops and duplicates
        valid = True
        for u, v in new_edges:
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
        
        if not valid:
            # Revert
            adj[a].append(b)
            adj[b].append(a)
            adj[c].append(d)
            adj[d].append(c)
            continue
        
        # Apply switch
        adj[a].append(c)
        adj[c].append(a)
        adj[b].append(d)
        adj[d].append(b)
        
        # Check girth (fast check)
        new_girth = _compute_girth(n, adj, g)
        
        if new_girth > current_girth:
            current_girth = new_girth
            if current_girth >= g:
                return _adj_to_edges(n, adj)
        else:
            # Revert
            adj[a].remove(c)
            adj[c].remove(a)
            adj[b].remove(d)
            adj[d].remove(b)
            adj[a].append(b)
            adj[b].append(a)
            adj[c].append(d)
            adj[d].append(c)
    
    return None

def _generate_random_3_regular(n):
    """Generate a random 3-regular graph using configuration model."""
    if n % 2 != 0:
        return None
    
    # Configuration model
    stubs = [i for i in range(n) for _ in range(3)]
    random.shuffle(stubs)
    
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        if (u, v) in edge_set or (v, u) in edge_set:
            return None
        edge_set.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify 3-regular
    for u in range(n):
        if len(adj[u]) != 3:
            return None
    
    return adj

def _compute_girth(n, adj, max_check):
    """
    Compute the girth of the graph, stopping if we find a cycle of length < max_check.
    Returns the minimum cycle length found, or max_check if no shorter cycle exists.
    Uses BFS from each vertex with parent tracking.
    """
    min_cycle = max_check
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_check // 2:
                continue
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle < max_check:
                            return min_cycle
    
    return min_cycle

def _adj_to_edges(n, adj):
    """Convert adjacency list to edge list."""
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _construct_luw(k, g, d=None):
    """LUW construction as fallback."""
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
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
