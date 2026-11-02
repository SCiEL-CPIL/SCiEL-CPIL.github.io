"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    # Target: (4, 10) cage
    if k == 4 and g == 10:
        # Try to find a circulant graph on 320 vertices
        result = _find_circulant_cage(320, 10)
        if result is not None:
            return result
            
    # Fallback to LUW construction
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        return luw_result
        
    # Fallback to random
    return _random_regular(k, g, start_time, deadline)


def _find_circulant_cage(n, target_girth):
    """Find a circulant graph C_n(a, b) with girth >= target_girth."""
    # a, b are step sizes. We consider generators {a, -a, b, -b}
    # To avoid duplicates, assume 1 <= a < b <= n//2
    steps_base = []
    for a in range(1, n // 2 + 1):
        for b in range(a + 1, n // 2 + 1):
            # Check girth
            if _has_girth_circulant(n, a, b, target_girth):
                # Construct edges
                edges = []
                for v in range(n):
                    u1 = (v + a) % n
                    u2 = (v - a) % n
                    u3 = (v + b) % n
                    u4 = (v - b) % n
                    # Add unique edges
                    edges.append((v, min(v, u1)))
                    edges.append((v, min(v, u2)))
                    edges.append((v, min(v, u3)))
                    edges.append((v, min(v, u4)))
                # Remove duplicates and sort
                edge_set = set()
                for e in edges:
                    if e[0] == e[1]:
                        continue
                    if e[0] > e[1]:
                        e = (e[1], e[0])
                    edge_set.add(e)
                return n, list(edge_set)
    return None


def _has_girth_circulant(n, a, b, min_girth):
    """Check if C_n(a, b) has girth >= min_girth using non-backtracking BFS."""
    steps = [a, -a, b, -b]
    # State: (pos, last_step_index)
    # We start from 0, take a step, then continue.
    # If we return to 0 with length < min_girth, girth is too small.
    
    # visited[pos][last_step_idx]
    # last_step_idx: 0->a, 1->-a, 2->b, 3->-b
    # We need to track the first step to ensure the cycle is simple?
    # Actually, non-backtracking BFS from 0 is sufficient to detect short cycles through 0.
    # In a circulant graph, if a short cycle exists, it is likely to pass through 0 or be detected by the local structure.
    # However, strictly speaking, we should check all vertices. But due to symmetry, checking from 0 is often enough for small girths in circulants.
    
    # Let's do a full BFS from 0 on the state graph.
    # State: (current_pos, last_step_idx)
    # Start: (a, 0), (-a, 1), (b, 2), (-b, 3)
    
    queue = deque()
    visited = {}
    
    for i, s in enumerate(steps):
        pos = s % n
        # If pos == 0, we have a cycle of length 1 (loop), but a!=0 so pos!=0.
        # If pos == 0, return False.
        if pos == 0:
            return False
        # Store depth 1
        visited[(pos, i)] = 1
        queue.append((pos, i, 1))
        
    while queue:
        pos, last_idx, depth = queue.popleft()
        if depth >= min_girth:
            continue
            
        for i, s in enumerate(steps):
            # Non-backtracking: s != -steps[last_idx]
            if s == -steps[last_idx]:
                continue
                
            new_pos = (pos + s) % n
            new_depth = depth + 1
            
            # If we are back at 0, we found a cycle of length new_depth
            if new_pos == 0:
                if new_depth < min_girth:
                    return False
                # If new_depth >= min_girth, it's okay for this path, but we should still explore?
                # No, if we found a cycle of length >= min_girth, that's fine.
                # We only fail if we find a cycle of length < min_girth.
                
            if (new_pos, i) not in visited:
                visited[(new_pos, i)] = new_depth
                queue.append((new_pos, i, new_depth))
                
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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


def _random_regular(k, g, start_time, deadline):
    n = 500
    edges = _gen_random_4reg(n)
    if edges is not None:
        return n, edges
    return n, []


def _gen_random_4reg(n):
    if n < 4 or n * 4 % 2 != 0:
        return None
    for _ in range(100):
        stubs = [v for v in range(n) for _ in range(4)]
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        if valid:
            return list(edges)
    return None

# EVOLVE-BLOCK-END
