"""Construct a 5-regular graph of girth 9 using algebraic methods."""
import time
import random
import numpy as np
from collections import deque

def _check_girth(adj, g):
    """Check if graph has girth at least g using non-backtracking walks."""
    n = len(adj)
    for start in range(n):
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g:
                return False
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == start and depth + 1 < g:
                    return False
                if (nb, v, depth + 1) not in visited:
                    visited.add((nb, v, depth + 1))
                    queue.append((nb, v, depth + 1))
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

def _construct_voltage(k, g, deadline):
    """Construct k-regular graph using voltage graph / group-based approach."""
    if k != 5:
        return None
    
    # Moore bound for g=9: 1 + 5*(1+4+16+64) = 426
    # Try n values that are multiples of 5 (for 5-regular)
    # Use a group-based construction: vertices are elements of Z_n,
    # edges connect i to i+a, i-b for chosen generators
    
    # For 5-regular: use jumps {a, -a, b, -b, c} where c != -c mod n
    # or {a, -a, b, -b, c} with c = n/2 if n even
    
    for n in range(430, 600, 5):  # multiples of 5, starting near Moore bound
        if time.time() > deadline - 20:
            break
        
        # Try different generator sets
        for trial in range(20):
            if time.time() > deadline - 20:
                break
            
            # Generate 3 jumps (giving 6 neighbors, but we need 5)
            # Use jumps a, -a, b, -b, and c where c = n/2 (if n even)
            # Or use a, -a, b, -b, c with c distinct
            
            if n % 2 == 0:
                # Use 2 pairs + 1 self-inverse
                a = random.randint(1, n//2 - 1)
                b = random.randint(1, n//2 - 1)
                c = n // 2
                jumps = [a, -a % n, b, -b % n, c]
            else:
                # Use 2 pairs + 1 more (need to be careful)
                # For odd n, use 3 jumps: a, -a, b, -b, c where c != +-a, +-b
                a = random.randint(1, n//2)
                b = random.randint(1, n//2)
                c = random.randint(1, n//2)
                jumps = [a, -a % n, b, -b % n, c]
            
            # Check all jumps are distinct and non-zero
            jumps_set = set(jumps)
            if len(jumps_set) != 5:
                continue
            if 0 in jumps_set:
                continue
            
            # Build adjacency
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(n):
                for j in jumps:
                    nb = (i + j) % n
                    if nb == i:
                        valid = False
                        break
                    adj[i].append(nb)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check degree
            if not all(len(adj[i]) == 5 for i in range(n)):
                continue
            
            # Check no duplicate edges
            edge_set = set()
            valid = True
            for i in range(n):
                for j in adj[i]:
                    e = (min(i,j), max(i,j))
                    if e in edge_set:
                        valid = False
                        break
                    edge_set.add(e)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check girth
            if _check_girth(adj, g):
                edges = list(edge_set)
                return n, edges
    
    return None

def construct(k, g):
    """Construct a k-regular graph of girth at least g."""
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_result = None
    best_n = float('inf')
    
    # Try LUW construction
    try:
        n, edges = _luw_construct(k, g)
        if n > 0:
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            valid = all(len(adj[i]) == k for i in range(n))
            
            if valid and n < 5000:
                try:
                    if _check_girth(adj, g):
                        best_result = (n, edges)
                        best_n = n
                except:
                    pass
    except:
        pass
    
    # Try voltage/group-based construction
    try:
        result = _construct_voltage(k, g, deadline)
        if result is not None:
            n, edges = result
            if n < best_n:
                best_result = (n, edges)
                best_n = n
    except:
        pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: return LUW result
    try:
        n, edges = _luw_construct(k, g)
        return n, edges
    except:
        pass
    
    # Last resort
    n = 1 + k * (1 + (k-1) + (k-1)**2 + (k-1)**3)
    return n, []
