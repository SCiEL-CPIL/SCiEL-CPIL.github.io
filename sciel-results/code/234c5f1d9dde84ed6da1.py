"""Construct a 4-regular graph of girth 10 (cage problem n(4,10)).

Strategy:
Search for a circulant graph C_n({+/-1, +/-a}) with girth >= 10.
This is a 4-regular graph. We iterate n from the Moore bound upwards
and check all valid 'a' values.
"""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    if k != 4:
        return _construct_generic(k, g)
    
    # Target: k=4, g=10
    # Search for circulant graph C_n({1, -1, a, -a})
    # Girth is the length of the shortest non-trivial word in {1, -1, a, -a} summing to 0 mod n.
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Lower bound for n is 243 (from prompt). Start searching from there.
    # n must be even for a bipartite-like structure? Not necessarily, but 4-regular implies n*k is even.
    # If n is odd, sum of degrees is 4n (even), so possible.
    # However, for girth 10 (even), bipartite is common.
    
    for n in range(244, 500):
        if time.time() - start_time > time_limit:
            break
            
        # a must be such that {1, -1, a, -a} are distinct mod n.
        # a != 0, 1, -1, n/2 (if n even)
        # Also a != -a mod n => 2a != 0 mod n.
        
        for a in range(2, n // 2):
            # Check if generators are distinct
            if a == 1 or a == n - 1: continue
            if n % 2 == 0 and a == n // 2: continue
            if 2 * a % n == 0: continue
            
            # Check girth
            # We need the shortest word in {1, -1, a, -a} that sums to 0 mod n to have length >= g.
            # We can do a BFS on the "word" space or simply check all combinations of length < g.
            # Since g=10, checking length up to 9 is feasible.
            # State: (current_sum, length, last_gen) to avoid backtracking?
            # Actually, girth is the length of the shortest cycle.
            # A cycle corresponds to a sequence of generators $g_1, ..., g_L$ such that sum $g_i \equiv 0 \pmod n$.
            # We must ensure the walk is non-backtracking?
            # For girth, we count the length of the cycle.
            # A cycle of length L in the graph corresponds to a word of length L.
            # We need to ensure that the walk doesn't close early.
            
            # Efficient check: BFS on (sum, last_step)
            # State: (s, last) where s is current sum mod n, last is the generator used to get here.
            # Start: (0, None).
            # Neighbors: (s+g, g) for g in {1, -1, a, -a} if g != -last.
            # If we reach (0, g) with length L, we have a cycle of length L.
            # We want the minimum L to be >= 10.
            
            # Optimization: We can just check if there is any cycle of length < 10.
            # We can use a set of visited (sum, last) with the minimum length to reach that state.
            # If we reach (0, g) with length L < 10, girth is < 10.
            
            visited = {}
            queue = [(0, None, 0)] # (sum, last_gen, length)
            found_short_cycle = False
            
            while queue:
                s, last, l = queue.pop(0)
                if l >= g:
                    continue
                
                for g_next in [1, -1, a, -a]:
                    if last is not None and g_next == -last:
                        continue
                    new_s = (s + g_next) % n
                    new_state = (new_s, g_next)
                    new_l = l + 1
                    
                    if new_state in visited:
                        if visited[new_state] <= new_l:
                            continue
                    else:
                        visited[new_state] = new_l
                    
                    if new_s == 0 and new_l > 0:
                        # Cycle found
                        if new_l < g:
                            found_short_cycle = True
                            break
                        else:
                            # We found a cycle of length >= g, but we need to ensure NO shorter ones exist.
                            # Since we are doing BFS by length, if we are at length L, all shorter lengths have been processed.
                            # If we haven't found a short cycle yet, and we are at length >= g, we can stop?
                            # No, we need to ensure that *no* cycle of length < g exists.
                            # If we reach (0, g_next) at length L >= g, it doesn't prove girth >= g.
                            # We must exhaust all states reachable in < g steps.
                            pass
                    
                    if new_l < g:
                        queue.append((new_s, g_next, new_l))
            
            if found_short_cycle:
                continue
            
            # If we finished BFS for lengths < g without finding a cycle closing at 0,
            # then girth >= g.
            # Wait, the BFS above breaks if found_short_cycle.
            # If not found, we need to be sure.
            # The BFS explores all non-backtracking walks of length < g.
            # If none of them return to 0, then there are no cycles of length < g.
            
            # Construct edges
            edges = []
            for i in range(n):
                edges.append((i, (i + 1) % n))
                edges.append((i, (i - 1) % n))
                edges.append((i, (i + a) % n))
                edges.append((i, (i - a) % n))
            
            # Deduplicate and sort
            unique_edges = set()
            for u, v in edges:
                if u < v:
                    unique_edges.add((u, v))
                else:
                    unique_edges.add((v, u))
            
            return n, list(unique_edges)
    
    # Fallback if no circulant graph found
    return _construct_generic(4, 10)

def _construct_generic(k, g):
    """Fallback: Return a large random regular graph or a known valid one."""
    # For safety, return a valid 4-regular graph with large n (e.g. 2048 from LUW or similar)
    # But we need it to be valid.
    # Let's just return a cycle of length n with extra edges? No, must be regular.
    # A simple 4-regular graph: C_n with chords?
    # For k=4, g=10, if we fail, we return a known valid large graph.
    # The LUW graph from the previous code might be valid but large.
    # Let's reuse the LUW logic here as a fallback.
    
    q = 4
    d = 5
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    def gf4_add(x, y): return x ^ y
    def gf4_sub(x, y): return x ^ y
    def gf4_mul(x, y): return mul_table[x][y]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf4_add(x[c], gf4_mul(y[a], x[b]))
                else:
                    y[c] = gf4_sub(x[c], gf4_mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
