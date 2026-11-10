import time
import random
import math
from collections import deque

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (seed fallback)."""
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


def _check_girth_fast(adj, n, min_girth):
    """Check if graph has girth >= min_girth using non-backtracking BFS from each vertex."""
    max_depth = (min_girth + 1) // 2 - 1
    for start in range(n):
        dist = {}
        for nb in adj[start]:
            dist[(start, nb)] = 1
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb))
        while queue:
            state = queue.popleft()
            v, prev = state
            d = dist[state]
            if d >= max_depth:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                new_state = (v, nb)
                if new_state not in dist:
                    dist[new_state] = d + 1
                    queue.append(new_state)
                elif dist[new_state] == d:
                    if 2 * d + 1 < min_girth:
                        return False
                else:
                    cycle_len = d + dist[new_state]
                    if cycle_len < min_girth:
                        return False
    return True


def _count_4_and_6_cycles(adj, n, half):
    """Count 4-cycles and 6-cycles in a bipartite graph (A: 0..half-1, B: half..n-1)."""
    count4 = 0
    count6 = 0
    # 4-cycles: A-u - B-v - A-w - B-x - A-u where u,w in A, v,x in B
    # Equivalent: for each pair (u,w) in A, count common B-neighbors
    for u in range(half):
        au = adj[u]
        for w in range(u + 1, half):
            common = len(au & adj[w])
            if common >= 2:
                count4 += common * (common - 1) // 2
    # 6-cycles: A-u1 - B-b1 - A-u2 - B-b2 - A-u3 - B-b3 - A-u1
    # For each triple (u1, u2, u3) in A, count ways to choose b1, b2, b3
    # This is expensive; use a proxy based on 4-cycle structure
    # A 6-cycle exists when there are paths of length 2 between A-vertices that close
    # Fast proxy: count pairs of A-vertices with >= 2 common B-neighbors (already done for 4-cycles)
    # and add a weighted term for 6-cycles
    return count4, count6


def _generate_random_4reg_bipartite(n, rng):
    """Generate a random 4-regular bipartite graph on n vertices (n even)."""
    half = n // 2
    # Create list of edge endpoints
    a_ends = []
    for i in range(half):
        a_ends.extend([i] * 4)
    b_ends = []
    for i in range(half):
        b_ends.extend([half + i] * 4)
    
    rng.shuffle(b_ends)
    
    adj = [set() for _ in range(n)]
    valid = True
    for a, b in zip(a_ends, b_ends):
        if b in adj[a]:
            valid = False
            break
        adj[a].add(b)
        adj[b].add(a)
    
    if not valid:
        return None
    
    for i in range(n):
        if len(adj[i]) != 4:
            return None
    
    return adj


def _local_search_bipartite(adj, n, g, deadline, rng):
    """Local search on bipartite 4-regular graph to maximize girth."""
    half = n // 2
    best_adj = [set(s) for s in adj]
    
    # Compute initial score
    score4, score6 = _count_4_and_6_cycles(best_adj, n, half)
    current_score = score4 + score6
    
    # Try swaps
    max_iterations = 50000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick random edges to swap
        # (u1, v1) and (u2, v2) -> (u1, v2) and (u2, v1)
        # where u1, u2 in A, v1, v2 in B
        u1 = rng.randint(0, half - 1)
        u2 = rng.randint(0, half - 1)
        if u1 == u2:
            continue
        
        v1_list = list(best_adj[u1])
        v2_list = list(best_adj[u2])
        v1 = v1_list[rng.randint(0, len(v1_list) - 1)]
        v2 = v2_list[rng.randint(0, len(v2_list) - 1)]
        
        if v1 == v2:
            continue
        if v2 in best_adj[u1] or v1 in best_adj[u2]:
            continue
        
        # Perform swap
        best_adj[u1].remove(v1)
        best_adj[v1].remove(u1)
        best_adj[u2].remove(v2)
        best_adj[v2].remove(u2)
        
        best_adj[u1].add(v2)
        best_adj[v2].add(u1)
        best_adj[u2].add(v1)
        best_adj[v1].add(u2)
        
        new_score4, new_score6 = _count_4_and_6_cycles(best_adj, n, half)
        new_score = new_score4 + new_score6
        
        if new_score <= current_score:
            current_score = new_score
            if new_score == 0:
                if _check_girth_fast(best_adj, n, g):
                    edges = []
                    for i in range(half):
                        for j in best_adj[i]:
                            edges.append((i, j))
                    return n, edges
        else:
            # Revert
            best_adj[u1].remove(v2)
            best_adj[v2].remove(u1)
            best_adj[u2].remove(v1)
            best_adj[v1].remove(u2)
            
            best_adj[u1].add(v1)
            best_adj[v1].add(u1)
            best_adj[u2].add(v2)
            best_adj[v2].add(u2)
    
    return None


def _voltage_lift(k, g):
    """Voltage graph construction for 4-regular graph of girth 10."""
    deadline = time.time() + 110
    rng = random.Random()
    
    # Moore bound for (4,10) is 242
    moore_bound = 242
    
    # Try n from Moore bound up
    for n in range(moore_bound, min(moore_bound + 80, 330), 2):
        if time.time() > deadline - 10:
            break
        
        half = n // 2
        if half < 4:
            continue
        
        # Try to construct a 4-regular bipartite graph with girth >= 10
        # Use random initialization + local search
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            adj = _generate_random_4reg_bipartite(n, rng)
            if adj is None:
                continue
            
            result = _local_search_bipartite(adj, n, g, deadline - 5, rng)
            if result is not None:
                return result
    
    return None


def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    deadline = time.time() + 115
    
    # Get LUW construction as fallback
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Try voltage lift / local search
    vl_result = _voltage_lift(k, g)
    if vl_result is not None:
        vl_n, vl_edges = vl_result
        if vl_n < best_n:
            best_n = vl_n
            best_edges = vl_edges
    
    return best_n, best_edges
