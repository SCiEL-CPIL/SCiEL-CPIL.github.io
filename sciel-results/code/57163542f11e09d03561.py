import time
import random
import numpy as np
from collections import deque, defaultdict

def _verify_girth(adj, n, g):
    """
    Verify girth >= g using non-backtracking BFS from each vertex.
    Optimized for speed: early termination if any cycle < g is found.
    """
    for s in range(n):
        # State: (current_node, prev_node)
        # dist[state] = length of non-backtracking walk from s to current_node via prev_node
        visited = set()
        visited.add((s, -1))
        q = deque()
        q.append((s, -1, 0))
        
        while q:
            u, p, d = q.popleft()
            if d >= g:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                state = (v, u)
                if state in visited:
                    continue
                visited.add(state)
                if v == s:
                    # Found a cycle of length d + 1
                    if d + 1 < g:
                        return False
                else:
                    if d + 1 < g:
                        q.append((v, u, d + 1))
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Moore bound calculation
    if g % 2 == 0:
        # Even girth
        # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        half_g = g // 2
        moore_bound = 2 * sum((k-1)**i for i in range(half_g))
        if moore_bound % 2 != 0:
            moore_bound += 1
    else:
        # Odd girth
        # n >= 1 + k * sum_{i=0}^{(g-1)/2 - 1} (k-1)^i
        half_g = (g - 1) // 2
        moore_bound = 1 + k * sum((k-1)**i for i in range(half_g))
        if moore_bound % 2 != 0:
            moore_bound += 1
        
    # For k=3, g=16:
    # moore_bound = 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254? 
    # Wait. Formula for even girth g:
    # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
    # For g=16, g/2 - 1 = 7.
    # Sum_{i=0}^7 2^i = 2^8 - 1 = 255.
    # n >= 2 * 255 = 510.
    # Next even is 512.
    
    # The current program uses LUW which gives 936.
    # We want to find a graph with n close to 512.
    
    # Strategy:
    # 1. Build a non-backtracking tree of depth g/2 - 1 from a root.
    #    This tree has exactly 1 + k * (1 + (k-1) + ... + (k-1)^(g/2-2)) vertices?
    #    No, let's look at the structure.
    #    Root (depth 0).
    #    Depth 1: k neighbors.
    #    Depth 2: k*(k-1) neighbors.
    #    ...
    #    Depth d: k*(k-1)^(d-1) neighbors.
    #    If we stop at depth D, the boundary vertices are at depth D.
    #    To form a k-regular graph, we need to connect the boundary vertices.
    #    The vertices at depth D have 1 parent (at depth D-1) and need k-1 more connections.
    #    If we connect them in a way that doesn't create short cycles, we are good.
    #    
    #    Let D = g/2 - 1.
    #    Number of boundary vertices B = k * (k-1)^(D-1).
    #    For k=3, g=16: D = 7.
    #    B = 3 * 2^6 = 192.
    #    Each boundary vertex needs 2 more edges.
    #    Total stubs to match = 192 * 2 = 384.
    #    We need to pair these stubs.
    #    Total vertices N = 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 382?
    #    Wait, sum_{i=0}^7 3*2^{i-1} for i>=1?
    #    Depth 0: 1
    #    Depth 1: 3
    #    Depth 2: 3*2 = 6
    #    Depth 3: 3*4 = 12
    #    Depth 4: 3*8 = 24
    #    Depth 5: 3*16 = 48
    #    Depth 6: 3*32 = 96
    #    Depth 7: 3*64 = 192
    #    Total = 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 382.
    #    This is less than 512. Why?
    #    Because the Moore bound counts the "ball" of radius g/2 around a vertex in a regular graph.
    #    In a k-regular graph of girth g, the number of vertices within distance < g/2 from any vertex is exactly the size of the non-backtracking tree of depth g/2 - 1.
    #    For g=16, distance < 8 means depths 0..7.
    #    The number of vertices is 1 + k * sum_{i=0}^{g/2-2} (k-1)^i.
    #    For k=3, g=16: 1 + 3 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 1 + 3 * 127 = 1 + 381 = 382.
    #    Wait, the Moore bound for even girth is 2 * sum_{i=0}^{g/2-1} (k-1)^i.
    #    For k=3, g=16: 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 2 * 255 = 510.
    #    Why the discrepancy?
    #    The Moore bound assumes that the graph is "perfect" in the sense that the balls of radius g/2 around any two vertices are disjoint or overlap only as required.
    #    Actually, the Moore bound is a lower bound on n.
    #    n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i for even g.
    #    My tree construction produces a graph with n=382? No, that's the size of the tree.
    #    If I connect the boundary, I get a graph of size 382.
    #    But 382 < 510. This implies my construction creates cycles shorter than g, OR I am miscounting.
    #    
    #    Let's re-evaluate.
    #    In a 3-regular graph of girth 16, pick a vertex v.
    #    Dist 0: 1
    #    Dist 1: 3
    #    Dist 2: 3*2 = 6
    #    ...
    #    Dist 7: 3*2^6 = 192.
    #    Total vertices in ball of radius 7 is 1 + 3 + 6 + ... + 192 = 382.
    #    If the graph has girth 16, then the balls of radius 7 around any vertex are disjoint?
    #    No, they can overlap if the distance between centers is <= 14.
    #    However, the Moore bound derivation states that n >= 2 * (1 + 2 + ... + 2^7) = 510.
    #    This bound is derived by considering that each edge can be "covered" by at most 2 balls? Or something similar.
    #    Actually, the Moore bound for even girth g is n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i.
    #    For k=3, g=16, this is 510.
    #    So a graph with n=382 CANNOT have girth 16.
    #    Why? Because if I build a tree of depth 7 and connect the leaves, I will inevitably create cycles.
    #    The shortest cycle will be determined by the connections between leaves.
    #    If I connect leaf u to leaf v, the cycle length is dist(u, v) + 1.
    #    In the tree, dist(u, v) = depth(u) + depth(v) - 2*depth(lca(u,v)).
    #    Since all leaves are at depth 7, dist(u, v) = 14 - 2*depth(lca).
    #    To avoid cycles < 16, we need 14 - 2*depth(lca) + 1 >= 16 => 15 - 2*depth(lca) >= 16 => -2*depth(lca) >= 1 => depth(lca) <= -0.5.
    #    This is impossible since depth(lca) >= 0.
    #    Wait, if depth(lca) = 0 (root), dist = 14. Cycle = 15. < 16.
    #    If depth(lca) = 1, dist = 12. Cycle = 13.
    #    So connecting leaves of a depth 7 tree in a 3-regular graph WILL create cycles of length <= 15.
    #    
    #    Therefore, the "tree + matching" approach with a single root tree does NOT work for even girth if we just connect leaves.
    #    We need a different structure.
    #    
    #    The LUW construction works because it's not a single tree rooted at one vertex. It's a "ladder" or "strip" structure that wraps around.
    #    
    #    Alternative Strategy:
    #    Use the fact that for k=3, g=16, the smallest known graph is 936.
    #    The Moore bound is 512.
    #    There is a large gap.
    #    Can we do better than 936?
    #    The current program returns 936.
    #    If we cannot find a graph < 936, we should return 936.
    #    But maybe we can find a graph between 512 and 936?
    #    
    #    Let's try a different construction:
    #    "Cage" graphs are often constructed using algebraic methods (LUW, etc.) or computational search.
    #    Since we are limited to 150s, we cannot do a massive search.
    #    
    #    Let's try to improve the LUW construction or use a different algebraic family.
    #    The LUW construction for k=3, g=16 gives 936.
    #    Is there a known construction with fewer vertices?
    #    According to the task description: "The smallest known such graph has 936 vertices".
    #    So 936 is the current best known upper bound.
    #    If we cannot find a new one, we should stick to 936.
    #    
    #    However, the directive says "Make a STRUCTURAL change".
    #    If we just return the same LUW graph, we are not changing the strategy.
    #    But if the LUW graph IS the best known, then any other strategy that doesn't find a better graph will score the same (1.0).
    #    If we try a heuristic search and fail, we might run out of time and return nothing or a bad graph.
    #    
    #    Let's look at the "inspiration" notes.
    #    "Vectorized numpy constraint checking... enables ~10x more repair attempts... find a smaller prime m... n=10m".
    #    This suggests a different construction family: n = 10 * m.
    #    For k=3, g=16, n=936 is not a multiple of 10.
    #    Maybe there is a construction with n=10m that is smaller than 936?
    #    936 is not divisible by 10.
    #    If there exists a 3-regular graph of girth 16 with n=930, 920, etc., that would be an improvement.
    #    
    #    Let's try to implement a heuristic search for a graph with n < 936.
    #    We can start with the LUW graph (936) and try to "shrink" it? No, that's hard.
    #    We can try to generate random 3-regular graphs of size n < 936 and check girth.
    #    The probability of finding one is low, but if we use a smart initialization (e.g., start with a high-girth graph and perturb), we might succeed.
    #    
    #    However, given the 150s limit and the difficulty, the most robust strategy is:
    #    1. Return the known best (936) as a baseline.
    #    2. Use remaining time to attempt a local search or random restart to find a smaller one.
    #    
    #    Let's implement a "Randomized Local Search" on the edge set.
    #    Start with the LUW graph (936).
    #    Try to remove 2 vertices and re-regularize? No, that changes n.
    #    
    #    Actually, the most likely "structural change" that could yield a score > 1.0 is finding a graph with n < 936.
    #    Since 936 is the *known* best, it is very unlikely we will find one in 150s with a simple heuristic.
    #    
    #    Wait, the task says "The smallest known such graph has 936 vertices".
    #    This implies that NO graph with n < 936 is known.
    #    If we find one, we make a major breakthrough.
    #    If we don't, we return 936.
    #    
    #    Is there any way to construct a graph with n < 936?
    #    The Moore bound is 512.
    #    There are no known graphs between 512 and 936?
    #    Actually, the gap is large. It is possible that a graph exists with, say, n=800.
    #    
    #    Let's try a "Genetic Algorithm" or "Simulated Annealing" approach.
    #    Representation: Edge list of a 3-regular graph.
    #    Fitness: Girth (maximize) and Size (minimize).
    #    But we must ensure girth >= 16.
    #    
    #    Given the constraints, I will implement a hybrid:
    #    1. Generate the LUW graph (936).
    #    2. Try to find a smaller graph via random sampling of 3-regular graphs with n in [512, 934] step 2.
    #       Use a fast girth checker.
    #       If found, return it.
    #    3. If not found, return 936.
    #    
    #    To make the search smarter, we can use the fact that high-girth graphs are rare.
    #    We can bias the random generation towards "tree-like" structures.
    #    E.g., generate a random tree, then add edges to make it 3-regular.
    #    
    #    Let's implement this.

    def generate_luw_graph(k, g):
        # Replicate the seed's LUW construction
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
        
        n = len(order)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return n, edges, adj

    # Get baseline
    baseline_n, baseline_edges, baseline_adj = generate_luw_graph(k, g)
    
    best_n = baseline_n
    best_edges = baseline_edges
    
    # Try to find a smaller graph
    if k == 3 and g == 16:
        # Range to search
        # Moore bound is 512.
        # We search from 512 up to 934.
        # We prioritize smaller n.
        # We use a time limit for the search.
        search_deadline = start_time + 130.0
        
        def try_random_graph(n):
            # Generate a random 3-regular graph
            # Method: Random matching of stubs
            # To increase chance of high girth, we can use a "tree-like" initialization
            # but random stub matching is standard.
            
            # Optimization: Use numpy for shuffling if possible? 
            # Python random.shuffle is fine for n=512.
            
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            
            edges = set()
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    return None
                if u > v: u, v = v, u
                if (u, v) in edges:
                    return None
                edges.add((u, v))
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Check degree
            for i in range(n):
                if len(adj[i]) != 3:
                    return None # Should not happen with perfect matching
            
            # Check girth
            # Fast check: BFS from each vertex up to depth g/2 - 1
            # If any cycle found < g, return False
            # We can prune if time is running out
            if time.time() > search_deadline:
                return None
            
            if _verify_girth(adj, n, g):
                return list(edges)
            return None

        # Search loop
        # We try a few random instances for each n
        max_attempts = 1000 # Will be limited by time
        
        for n_try in range(moore_bound, best_n, 2):
            if time.time() > search_deadline:
                break
            
            # Try a few times for this n
            attempts_for_n = 0
            while attempts_for_n < 20 and time.time() < search_deadline:
                edges = try_random_graph(n_try)
                if edges:
                    best_n = n_try
                    best_edges = edges
                    break
                attempts_for_n += 1
            
            if best_n < baseline_n:
                # Found an improvement! We can stop or continue to find even smaller.
                # Let's continue to see if we can go lower, but keep the best.
                pass

    return best_n, best_edges
