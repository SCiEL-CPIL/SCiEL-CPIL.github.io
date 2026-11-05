import math
import random
import time
import numpy as np
from collections import deque
import sys

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g, minimizing n.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_result = None
    
    # Compute Moore bound
    if g % 2 == 0:
        h = g // 2 - 1
        moore = 2 * k * (k - 1) ** h if h >= 0 else 2 * k
    else:
        h = (g - 1) // 2
        moore = 2 * (k - 1) ** h + k * (k - 1) ** (h - 1) if h >= 1 else 2 * k
    
    moore = max(moore, 2 * k)
    
    # Strategy 1: Try known explicit constructions (LUW)
    n_luw, edges_luw = _luw_graph(k, g)
    if n_luw > 0:
        best_result = (n_luw, edges_luw)
    
    # Strategy 1b: Try projective space incidence graph
    if time.time() < deadline - 15:
        n_proj, edges_proj = _projective_incidence_graph(k, g)
        if n_proj > 0 and (best_result is None or n_proj < best_result[0]):
            best_result = (n_proj, edges_proj)
    
    if best_result is None:
        # Fallback
        if k == 1:
            return (2, [(0, 1)])
        elif k == 2:
            n = max(g, 3)
            return (n, [(i, (i + 1) % n) for i in range(n)])
        else:
            n = moore * 4
            while (n * k) % 2 != 0:
                n += 1
            edges = _random_k_regular(n, k)
            if edges:
                return (n, edges)
            return (0, [])

    # Strategy 2: Random search with girth-aware generation
    if time.time() < deadline - 8:
        try:
            # Start closer to Moore bound for better chance of finding smaller graphs
            start_n = max(moore, 2 * k + 2)
            # Ensure n*k is even
            if (start_n * k) % 2 != 0:
                start_n += 1
            end_n = best_result[0] - 1 if best_result else start_n + 500
            
            for n in range(start_n, end_n + 1, 2):
                if time.time() > deadline - 5:
                    break
                if (n * k) % 2 != 0:
                    continue
                
                found_valid = False
                # More attempts for smaller n where random search is more likely to succeed
                attempts = 20 if n < 800 else 12 if n < 1200 else 6
                for attempt in range(attempts):
                    if time.time() > deadline - 5:
                        break
                    
                    try:
                        edges = _girth_aware_random(n, k, g)
                        if edges is None:
                            continue
                        
                        if _verify_girth_fast(k, g, n, edges):
                            if best_result is None or n < best_result[0]:
                                best_result = (n, edges)
                            found_valid = True
                            break
                    except Exception:
                        pass
                
                if found_valid:
                    break
        
        except Exception:
            pass
    
    return best_result


def _projective_incidence_graph(k, g):
    """
    Try to construct a k-regular bipartite graph from projective space PG(n, q).
    The incidence graph of PG(n,q) has 2*(q^(n+1)-1)/(q-1) vertices.
    Each point is incident to (q^n+q^(n-1)+...+1) hyperplanes, so it's not regular
    unless we select a subset. Instead, use a different approach:
    
    For small q and n, check if the incidence graph is k-regular.
    PG(2, q): points = (q^2+q+1), lines = (q^2+q+1), each point on (q+1) lines.
    So k = q+1, total vertices = 2*(q^2+q+1).
    
    For PG(3, q): points = (q^3+q^2+q+1), planes = same, each point on (q^2+q+1) planes.
    So k = q^2+q+1, total vertices = 2*(q^3+q^2+q+1).
    
    Girth of incidence graph of PG(n,q): The girth is 2*(n+1) for the full incidence graph
    (a point-hyperplane alternating cycle corresponds to a flag sequence).
    Actually for PG(2,q), girth is 6 (triangle in line graph). For PG(n,q), girth is 2*(n+1).
    
    So for g=9 (odd), we need girth >= 9, but bipartite graphs have even girth.
    Wait - the task says "no cycle shorter than g", so girth >= g.
    For bipartite graphs, girth is always even. So for g=9, we need girth >= 10.
    PG(4, q) has girth 10. k = q^3+q^2+q+1.
    For q=2: k = 15, n = 2*(31) = 62. Too few vertices but k=15 != 5.
    
    This approach only works when k matches the number of hyperplanes through a point.
    Let's check: for k=5, we need q^3+q^2+q+1 = 5? No solution.
    q+1 = 5 => q=4. PG(2,4): k=5, n=2*(16+4+1)=42. Girth=6 < 9. Not good.
    
    So projective space incidence graphs don't directly give us k=5, g=9.
    Return 0 to indicate no valid construction found.
    """
    if k == 1:
        return 2, [(0,1)]
    
    # Check PG(2, q) for k = q+1
    if k - 1 >= 0:
        q = k - 1
        # PG(2,q) has girth 6, need g <= 6
        if g <= 6:
            n_pts = q*q + q + 1
            n_total = 2 * n_pts
            edges = []
            # Enumerate points and lines in PG(2,q)
            # Points are 1D subspaces of F_q^3, lines are 2D subspaces
            # This is complex; skip for now
            pass
    
    # PG(3, q) for k = q^2+q+1
    # PG(4, q) for k = q^3+q^2+q+1
    # These don't match k=5 typically
    
    return 0, []


def _luw_graph(q, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    q-regular bipartite graph with girth at least d+5 (for odd d).
    """
    if q < 3:
        return 0, []
    
    # Find appropriate d
    d_min = g - 5
    if d_min % 2 == 0:
        d_min += 1
    d_min = max(3, d_min)
    
    # Try a few values of d
    for d in range(d_min, min(d_min + 5, 20)):
        if d < 3:
            continue
        try:
            n, edges = _luw_graph_params(q, d)
            if n > 0:
                return n, edges
        except Exception:
            continue
    
    return 0, []


def _luw_graph_params(q, d):
    """
    LUW graph with specific parameters.
    CD(d, q): q-regular bipartite graph with girth at least d+5.
    """
    if q < 3 or d < 3:
        return 0, []
    
    # Field operations - assume q is prime
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Rule for the graph
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Vertices are (side, x) where side in {0,1}, x is tuple of length d+1
    # x[0] is always None (unused), x[1..d] are field elements
    # Total vertices: 2 * q^d
    
    # Build all vertices and their neighbors
    # Vertex: (side, x1, x2, ..., xd)
    # For side=0: neighbor is (1, y) where y[c] = x[c] + y[a]*x[b] for c=2..d, y[1]=first
    # For side=1: neighbor is (0, y) where y[c] = x[c] - x[a]*y[b] for c=2..d, y[1]=first
    
    def compute_neighbor(side, x_tuple, first):
        """Compute the neighbor vertex given side, current x values, and first component."""
        x = [None] + list(x_tuple)  # x[0]=None, x[1..d]
        y = [None, first] + [0] * (d - 1)
        for c in range(2, d + 1):
            a, b = rule[c]
            if side == 0:
                y[c] = add(x[c], mul(y[a], x[b]))
            else:
                y[c] = sub(x[c], mul(x[a], y[b]))
        return 1 - side, tuple(y[1:])
    
    # Enumerate all vertices
    # Side 0: all combinations of x1..xd in F_q
    # Side 1: all combinations of x1..xd in F_q
    index = {}
    order = []
    
    # Add all side-0 vertices
    for x1 in range(q):
        for x2 in range(q):
            # ... this gets complex for large d; use iterative approach
            pass
    
    # Use BFS from a starting vertex to enumerate the connected component
    start = (0, tuple([0] * d))
    index[start] = 0
    order = [start]
    edges = []
    
    for vertex in order:
        side, x_tuple = vertex
        for first in range(q):
            other = compute_neighbor(side, x_tuple, first)
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge from side-0 to side-1 to avoid duplicates
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _random_k_regular(n, k):
    """
    Generate a random k-regular graph on n vertices.
    """
    if n * k < 2 or (n * k) % 2 != 0:
        return None
    
    for _ in range(10):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        valid = True
        edge_set = set()
        
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
        
        if valid:
            return edges
    
    return None


def _girth_aware_random(n, k, g):
    """
    Generate a random k-regular graph while avoiding creating cycles shorter than g.
    Uses a modified configuration model that checks for short cycles before accepting edges.
    Improved: use retry with different random seeds and more aggressive pruning.
    """
    if n * k < 2 or (n * k) % 2 != 0:
        return None
    
    max_attempts = 12
    
    for attempt in range(max_attempts):
        # Build adjacency as a list of sets for fast lookup
        adj = [set() for _ in range(n)]
        edge_set = set()
        
        # Create stubs with randomization
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        valid = True
        rejected = 0
        max_rejected = n * k // 4  # Give up if too many rejections
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            
            # Check if adding this edge creates a short cycle
            if _creates_short_cycle(adj, u, v, g):
                rejected += 1
                if rejected > max_rejected:
                    valid = False
                    break
                # Try swapping with a later stub
                swapped = False
                for j in range(i + 2, len(stubs), 2):
                    u2, v2 = stubs[j], stubs[j + 1] if j + 1 < len(stubs) else None
                    if v2 is None:
                        continue
                    # Try swapping v with v2
                    if u2 != u and v2 != u and (min(u, v2), max(u, v2)) not in edge_set:
                        if not _creates_short_cycle(adj, u, v2, g):
                            stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                            v = stubs[i + 1]
                            key = (min(u, v), max(u, v))
                            swapped = True
                            break
                    # Try swapping u with u2
                    if u2 != v and v2 != v and (min(u2, v), max(u2, v)) not in edge_set:
                        if not _creates_short_cycle(adj, u2, v, g):
                            stubs[i], stubs[j] = stubs[j], stubs[i]
                            u = stubs[i]
                            key = (min(u, v), max(u, v))
                            swapped = True
                            break
                
                if not swapped:
                    valid = False
                    break
            
            edge_set.add(key)
            adj[u].add(v)
            adj[v].add(u)
            edges.append(key)
        
        if valid:
            return edges
    
    return None


def _creates_short_cycle(adj, u, v, g):
    """
    Check if adding edge (u,v) to the current graph creates a cycle of length < g.
    A cycle is created if there's already a path from u to v of length < g-1.
    We check for paths of length up to g-2 using BFS with non-backtracking.
    """
    # BFS from u to find if v is reachable in < g-1 steps (non-backtracking)
    # Path length L means cycle length L+1, need L+1 < g => L < g-1
    max_path_len = g - 2  # if path has this many edges, cycle has g-1 edges < g
    
    if max_path_len < 1:
        return False
    
    # BFS with state (current, prev) to enforce non-backtracking
    # Use a set of tuples for visited states
    visited = set()
    queue = deque()
    
    for nb in adj[u]:
        if nb == v:
            continue
        visited.add((u, nb))
        queue.append((u, nb, 1))
    
    while queue:
        prev, curr, dist = queue.popleft()
        
        if dist >= max_path_len:
            continue
        
        for next_v in adj[curr]:
            if next_v == prev:
                continue
            
            if next_v == v:
                # Found path of length dist+1 from u to v
                # Cycle length = dist + 1 + 1 = dist + 2
                if dist + 2 < g:
                    return True
                continue
            
            key = (curr, next_v)
            if key not in visited:
                visited.add(key)
                queue.append((curr, next_v, dist + 1))
    
    return False


def _verify_girth_fast(k, g, n, edges):
    """
    Verify girth >= g using non-backtracking BFS.
    Optimized with early termination and efficient data structures.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth using non-backtracking walks from each vertex
    # State: (prev, curr) -> distance
    # For efficiency, use a list of dicts keyed by prev
    for start in range(n):
        # visited[prev][curr] = dist
        visited = {}
        queue = deque()
        
        for neighbor in adj[start]:
            if start not in visited:
                visited[start] = {}
            visited[start][neighbor] = 1
            queue.append((start, neighbor, 1))
        
        while queue:
            prev, curr, dist = queue.popleft()
            
            if dist >= g - 1:
                continue
            
            for next_v in adj[curr]:
                if next_v == prev:
                    continue
                
                if curr in visited and next_v in visited[curr]:
                    cycle_len = visited[curr][next_v] + dist + 1
                    if cycle_len < g:
                        return False
                else:
                    if curr not in visited:
                        visited[curr] = {}
                    visited[curr][next_v] = dist + 1
                    if dist + 1 < g - 1:
                        queue.append((curr, next_v, dist + 1))
    
    return True


def _verify_girth_numpy(k, g, n, edges):
    """
    Verify girth >= g using non-backtracking BFS.
    """
    return _verify_girth_fast(k, g, n, edges)


def _generate_valid_graph(k, g, moore):
    """
    Fallback method.
    """
    n, edges = _luw_graph(k, g)
    if n > 0:
        return n, edges
    
    n, edges = _projective_incidence_graph(k, g)
    if n > 0:
        return n, edges
    
    for n in range(moore, moore + 500):
        if (n * k) % 2 != 0:
            continue
        try:
            edges = _girth_aware_random(n, k, g)
            if edges is None:
                edges = _random_k_regular(n, k)
            if edges is not None and _verify_girth_fast(k, g, n, edges):
                return n, edges
        except Exception:
            continue
    
    return 0, []
