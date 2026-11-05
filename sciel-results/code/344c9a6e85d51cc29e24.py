"""Construct k-regular graph of girth g using random Cayley graphs on finite groups."""
# EVOLVE-BLOCK-START
import random
import itertools
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try to build a Cayley graph on a finite group using involutory generators.
    Try several group structures and random generator sets, verifying girth via
    non-backtracking walks. Fall back to L-U-W construction if needed.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
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
    
    def verify_girth(n, edges, g):
        """Verify no cycle shorter than g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Non-backtracking BFS from each vertex
        for start in range(n):
            # visited[(node, prev)] = distance
            visited = {}
            queue = deque()
            for nb in adj[start]:
                visited[(nb, start)] = 1
                queue.append((nb, start, 1))
            
            while queue:
                node, prev, dist = queue.popleft()
                if dist >= g:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if (nb, node) in visited:
                        # Found a cycle of length dist + 1
                        if dist + 1 < g:
                            return False
                    else:
                        visited[(nb, node)] = dist + 1
                        if dist + 1 < g:
                            queue.append((nb, node, dist + 1))
        return True
    
    def try_cayley_z2m(m, k, g, seed=42):
        """Try Cayley graph on (Z/2Z)^m with k involutory generators."""
        random.seed(seed)
        # Generate k random non-zero vectors in (Z/2Z)^m
        # They must be linearly independent to get a connected graph
        gens = []
        attempts = 0
        while len(gens) < k and attempts < 1000:
            v = tuple(random.randint(0, 1) for _ in range(m))
            if all(c == 0 for c in v):
                continue
            # Check linear independence
            independent = True
            for existing in gens:
                # Check if v is in span of existing (simple check: not equal to any existing)
                if v == existing:
                    independent = False
                    break
            # More careful: check if v is linear combination of existing
            # For small m, we can do Gaussian elimination
            if independent:
                # Verify linear independence properly
                mat = [list(g) for g in gens] + [list(v)]
                rank = 0
                for col in range(m):
                    pivot = None
                    for row in range(rank, len(mat)):
                        if mat[row][col]:
                            pivot = row
                            break
                    if pivot is not None:
                        mat[rank], mat[pivot] = mat[pivot], mat[rank]
                        for row in range(len(mat)):
                            if row != rank and mat[row][col]:
                                mat[row] = [(a ^ b) for a, b in zip(mat[row], mat[rank])]
                        rank += 1
                if rank == len(mat):
                    gens.append(v)
            attempts += 1
        
        if len(gens) < k:
            return None
        
        n = 2 ** m
        # Build edges
        edges = []
        for u in range(n):
            for gvec in gens:
                v = u
                for i in range(m):
                    if gvec[i]:
                        v ^= (1 << i)
                if v > u:  # Avoid duplicate edges
                    edges.append((u, v))
        
        if len(edges) != n * k // 2:
            return None
        
        # Verify girth
        if verify_girth(n, edges, g):
            return n, edges
        return None
    
    def try_cayley_product(m1, m2, k, g, seed=42):
        """Try Cayley graph on (Z/2Z)^m1 x (Z/2Z)^m2 with mixed generators."""
        random.seed(seed)
        m = m1 + m2
        n = 2 ** m
        
        # Generate k random non-zero vectors
        gens = []
        attempts = 0
        while len(gens) < k and attempts < 500:
            v = tuple(random.randint(0, 1) for _ in range(m))
            if all(c == 0 for c in v):
                continue
            # Check not equal to existing
            if v not in gens:
                # Check linear independence
                mat = [list(g) for g in gens] + [list(v)]
                rank = 0
                for col in range(m):
                    pivot = None
                    for row in range(rank, len(mat)):
                        if mat[row][col]:
                            pivot = row
                            break
                    if pivot is not None:
                        mat[rank], mat[pivot] = mat[pivot], mat[rank]
                        for row in range(len(mat)):
                            if row != rank and mat[row][col]:
                                mat[row] = [(a ^ b) for a, b in zip(mat[row], mat[rank])]
                        rank += 1
                if rank == len(mat):
                    gens.append(v)
            attempts += 1
        
        if len(gens) < k:
            return None
        
        edges = []
        for u in range(n):
            for gvec in gens:
                v = u
                for i in range(m):
                    if gvec[i]:
                        v ^= (1 << i)
                if v > u:
                    edges.append((u, v))
        
        if len(edges) != n * k // 2:
            return None
        
        if verify_girth(n, edges, g):
            return n, edges
        return None
    
    # Try various Cayley graph constructions
    # For k=5, g=9, we need n >= Moore bound
    # Moore bound for k=5, g=9: 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # So n >= 426, meaning m >= 9 (since 2^9 = 512)
    
    best_result = None
    start_time = time.time()
    time_limit = 100  # Leave some time for fallback
    
    # Try (Z/2Z)^m for m from 9 upward
    for m in range(9, 12):
        if time.time() - start_time > time_limit:
            break
        for seed in range(20):
            if time.time() - start_time > time_limit:
                break
            result = try_cayley_z2m(m, k, g, seed=seed)
            if result is not None:
                n, edges = result
                if best_result is None or n < best_result[0]:
                    best_result = result
                if time.time() - start_time > time_limit:
                    break
    
    # Try product structures
    for m1 in range(4, 7):
        for m2 in range(4, 7):
            if time.time() - start_time > time_limit:
                break
            m = m1 + m2
            if m < 9 or m > 11:
                continue
            for seed in range(10):
                if time.time() - start_time > time_limit:
                    break
                result = try_cayley_product(m1, m2, k, g, seed=seed)
                if result is not None:
                    n, edges = result
                    if best_result is None or n < best_result[0]:
                        best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fall back to L-U-W construction
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
