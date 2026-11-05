"""Combined program: LUW baseline + Cayley graph search + randomized construction."""
import time
import random
import math
from collections import deque
from itertools import product

def _verify_girth(n, edges, k, g):
    """Verify that the graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    for start in range(n):
        visited = set()
        queue = deque()
        for neighbor in adj[start]:
            queue.append((start, neighbor, 1))
            visited.add((start, neighbor))
        while queue:
            prev, curr, depth = queue.popleft()
            if depth >= g:
                continue
            for next_node in adj[curr]:
                if next_node == prev:
                    continue
                if next_node == start and depth + 1 < g:
                    return False
                if (curr, next_node) not in visited:
                    visited.add((curr, next_node))
                    queue.append((curr, next_node, depth + 1))
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _cayley_cyclic(m, a, b, g):
    """Check Cayley graph of Z_m with S={a,-a,b,-b} has girth >= g."""
    S = [a, -a, b, -b]
    for L in range(2, g):
        # Enumerate non-backtracking sequences of length L
        # Use iterative DFS to avoid recursion depth issues
        stack = [(S[0], 1, S[0], S[0])]  # (prev_step, length_done, current_sum, last_step)
        # Actually let's do it properly:
        def check_length(L):
            # Non-backtracking: consecutive steps can't be inverses
            # Build all sequences of length L from S where s_i != -s_{i+1}
            # Check if any sums to 0 mod m
            # Use iterative approach
            # State: (last_step, current_sum, length_done)
            if L == 1:
                return any(s % m == 0 for s in S)
            # BFS/DFS over non-backtracking sequences
            # For L up to 9, 4*3^8 = 26244, manageable
            results = []
            for s0 in S:
                results.append((s0, s0))  # (last, sum)
            for step in range(1, L):
                new_results = []
                for last, s in results:
                    for s2 in S:
                        if s2 != -last:
                            new_results.append((s2, s + s2))
                results = new_results
            for last, s in results:
                if s % m == 0:
                    return True
            return False
        if check_length(L):
            return False
    return True

def _cayley_search(k, g, deadline):
    """Search for Cayley graph on abelian groups with girth >= g."""
    if k != 4:
        return None
    
    best = None
    
    # Moore bound for (4,10) is 243
    moore = 243
    
    # Try cyclic groups Z_m
    for m in range(moore, moore + 200):
        if time.time() > deadline:
            break
        for a in range(1, m // 2 + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, m // 2 + 1):
                if a == m - a or b == m - b:
                    continue
                if a == b or a == m - b or b == m - a:
                    continue
                if math.gcd(math.gcd(a, b), m) != 1:
                    continue
                if _cayley_cyclic(m, a, b, g):
                    edge_set = set()
                    for i in range(m):
                        for s in [a, b]:
                            j = (i + s) % m
                            if i < j:
                                edge_set.add((i, j))
                            else:
                                edge_set.add((j, i))
                    edges = sorted(edge_set)
                    if best is None or m < best[0]:
                        best = (m, edges)
                    break
            if best is not None and best[0] == m:
                break
        if best is not None and best[0] == m:
            break
    
    if best is not None:
        return best
    
    # Try Z_a x Z_b
    for a in range(2, 20):
        if time.time() > deadline:
            break
        for b in range(a, 30):
            if time.time() > deadline:
                break
            m = a * b
            if m < moore or m > moore + 100:
                continue
            # Generating set: {(sa, sb), (-sa, -sb), (ta, tb), (-ta, -tb)}
            # where (sa,sb) and (ta,tb) are distinct elements
            # Try small generators
            for sa in range(1, a):
                for sb in range(0, b):
                    for ta in range(1, a):
                        for tb in range(0, b):
                            # Check distinctness and non-inverses
                            g1 = (sa, sb)
                            g2 = (ta, tb)
                            neg_g1 = (a - sa if sa != 0 else 0, b - sb if sb != 0 else 0)
                            neg_g2 = (a - ta if ta != 0 else 0, b - tb if tb != 0 else 0)
                            if g1 == g2 or g1 == neg_g2 or neg_g1 == g2:
                                continue
                            if g1 == neg_g1 or g2 == neg_g2:
                                continue
                            # Check connectedness: gcd of all coordinates with group orders
                            if math.gcd(math.gcd(math.gcd(sa, ta), a), b) != 1 and \
                               math.gcd(math.gcd(math.gcd(sb, tb), b), a) != 1:
                                continue
                            # Verify girth by checking non-backtracking closed walks
                            # This is expensive; skip for now and just verify
                            # Build the graph
                            edge_set = set()
                            for i in range(a):
                                for j in range(b):
                                    for (dx, dy) in [(sa, sb), (ta, tb)]:
                                        ni = (i + dx) % a
                                        nj = (j + dy) % b
                                        u = i * b + j
                                        v = ni * b + nj
                                        if u < v:
                                            edge_set.add((u, v))
                                        else:
                                            edge_set.add((v, u))
                            edges = sorted(edge_set)
                            if _verify_girth(m, edges, 4, g):
                                if best is None or m < best[0]:
                                    best = (m, edges)
                                return best
    return best

def _randomized_search(k, g, deadline, target_n):
    """Randomized search for k-regular graphs with girth >= g on target_n vertices."""
    if k != 4:
        return None
    
    best = None
    rng = random.Random(42)
    
    # Try to build random 4-regular graphs on various vertex counts
    # Start from the Moore bound and go up
    for n in range(243, target_n + 1, 7):  # n must be even for 4-regular
        if n % 2 != 0:
            n += 1
        if time.time() > deadline:
            break
        
        # Try a few random 4-regular graphs on n vertices
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Configuration model: 4 stubs per vertex
            stubs = []
            for i in range(n):
                for _ in range(4):
                    stubs.append(i)
            
            rng.shuffle(stubs)
            
            # Pair up stubs to form edges
            edges = []
            edge_set = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append(key)
            
            if not valid or len(edges) != 2 * n:
                continue
            
            # Verify it's 4-regular
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != 4 for d in deg):
                continue
            
            # Check girth
            if _verify_girth(n, edges, 4, g):
                if best is None or n < best[0]:
                    best = (n, edges)
                break  # Found one for this n, move to smaller n
    
    return best

def construct(k, g):
    deadline = time.time() + 130.0  # 20s safety margin
    
    # Get LUW baseline (always valid)
    luw_n, luw_edges = _luw_construct(k, g)
    
    if k != 4:
        # For other k, just return LUW
        return luw_n, luw_edges
    
    # Try Cayley graph search (structured, fast verification)
    cayley_result = _cayley_search(4, g, deadline - 60)  # Give 60s to Cayley
    if cayley_result is not None:
        best_n, best_edges = cayley_result
    else:
        best_n, best_edges = luw_n, luw_edges
    
    # Try randomized search to beat the current best
    if time.time() < deadline - 10:
        rand_result = _randomized_search(4, g, deadline - 10, min(best_n, 500))
        if rand_result is not None:
            if rand_result[0] < best_n:
                best_n, best_edges = rand_result
    
    return best_n, best_edges
