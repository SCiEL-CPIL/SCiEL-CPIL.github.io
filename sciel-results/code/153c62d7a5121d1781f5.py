import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 115  # Leave 5s margin
    
    # 1. Get baseline from LUW
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        n_luw, edges_luw = luw_result
        if _verify_graph(edges_luw, n_luw, k, g):
            best_n = n_luw
            best_edges = edges_luw
        else:
            best_n = float('inf')
            best_edges = None
    else:
        best_n = float('inf')
        best_edges = None

    # 2. Try to find a smaller graph using local search on bipartite graphs
    # The (k,g)-cage for k=4, g=10 is bipartite.
    # We search for n starting from Moore bound up to best_n - 1.
    
    if k % 2 != 0:
        if best_edges is not None:
            return best_n, best_edges
        return _luw_construct(k, g)
        
    # Moore bound
    moore_bound = _moore_bound(k, g)
    
    # Search range
    # We know 320 is a candidate.
    # Let's try n from moore_bound to min(best_n - 1, 400)
    
    # Optimization: If best_n is large, we should definitely try small n first.
    # The local search is more likely to succeed on smaller n if a solution exists.
    
    # We will try n values in increasing order.
    # But checking every n is expensive.
    # Let's try a few specific candidates: 320, 322, ... up to best_n.
    
    # To be safe and fast, let's try n = 320 first if it's >= moore_bound.
    # If not found, try 322, 324...
    
    candidates = []
    start_n = max(moore_bound, 320) # We know 320 is the known best
    if start_n < best_n:
        # Try 320, 322, 324 ... up to best_n - 1
        # But don't try too many.
        for n in range(start_n, min(best_n, start_n + 100), 2):
            candidates.append(n)
            
    for n in candidates:
        if time.time() > deadline - 5:
            break
        result = _local_search_bipartite(k, g, n, deadline)
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found
                # If we find 320, we can stop? 
                if n_found == 320:
                    return best_n, best_edges

    if best_edges is not None:
        return best_n, best_edges
        
    return _luw_construct(k, g)

def _moore_bound(k, g):
    if g % 2 == 0:
        m = g // 2
        s = 0
        for i in range(m):
            s += (k - 1) ** i
        return 2 * s
    else:
        m = (g - 1) // 2
        s = 0
        for i in range(m):
            s += (k - 1) ** i
        return 1 + k * s

def _verify_graph(edges, n, k, g):
    if n <= 0 or len(edges) == 0:
        return False
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != k:
            return False
            
    # Girth check
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        max_depth = (g - 2) // 2
        
        for d in range(1, max_depth + 1):
            new_queue = []
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = d
                        parent[v] = u
                        new_queue.append(v)
                    else:
                        if dist[v] < d - 1:
                            cycle_len = d + dist[v] + 1
                            if cycle_len < g:
                                return False
                        elif dist[v] == d - 1 and parent[u] != v:
                            cycle_len = 2 * d
                            if cycle_len < g:
                                return False
            queue = new_queue
    return True

def _luw_construct(k, g):
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

def _local_search_bipartite(k, g, n, deadline):
    M = n // 2
    if M < 10:
        return None
        
    # Represent graph by 4 permutations P1, P2, P3, P4 of range(M)
    # A = [[0, P1+P2+P3+P4], [P1^T+P2^T+P3^T+P4^T, 0]]
    
    def random_init():
        perms = []
        for _ in range(k):
            p = np.random.permutation(M)
            perms.append(p)
        return perms

    def calc_cost(perms):
        # Build A
        # A is 2M x 2M
        # Top-right block B = sum of permutation matrices
        B = np.zeros((M, M), dtype=np.int64)
        for p in perms:
            B[np.arange(M), p] = 1
        A = np.zeros((2*M, 2*M), dtype=np.int64)
        A[:M, M:] = B
        A[M:, :M] = B.T
        
        # Compute traces
        # A^2
        A2 = A @ A
        # A^4
        A4 = A2 @ A2
        # A^6
        A6 = A4 @ A2
        # A^8
        A8 = A4 @ A4
        
        c4 = np.trace(A4)
        c6 = np.trace(A6)
        c8 = np.trace(A8)
        
        # We want to minimize c4 + c6 + c8
        # But if c4 > 0, c6 and c8 are contaminated.
        # So weight c4 heavily.
        return c4 * 1000 + c6 * 100 + c8

    # Simulated Annealing
    perms = random_init()
    current_cost = calc_cost(perms)
    
    if current_cost == 0:
        return _perms_to_edges(perms, M)
        
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a move
        # Swap two elements in two different permutations?
        # Or swap edges between two permutations?
        # Let's pick two permutations i, j and two indices a, b.
        # Swap perms[i][a] and perms[j][b]? No, that breaks permutation property of i and j?
        # If we swap values, we must swap the entire entries in the permutation arrays.
        # Let's pick two permutations i, j.
        # Pick two indices a, b.
        # Swap perms[i][a] and perms[i][b]? This is a 2-opt within one permutation.
        # Or swap perms[i][a] and perms[j][a]? This swaps the neighbor of node a in perm i and perm j.
        
        i = random.randint(0, k-1)
        j = random.randint(0, k-1)
        if i == j:
            continue
        a = random.randint(0, M-1)
        b = random.randint(0, M-1)
        
        # Save original
        orig_i_a = perms[i][a]
        orig_i_b = perms[i][b]
        orig_j_a = perms[j][a]
        orig_j_b = perms[j][b]
        
        # Move: Swap perms[i][a] and perms[j][b]
        # This changes the graph.
        perms[i][a], perms[j][b] = perms[j][b], perms[i][a]
        
        new_cost = calc_cost(perms)
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost == 0:
                return _perms_to_edges(perms, M)
        else:
            # Revert
            perms[i][a], perms[j][b] = orig_i_a, orig_i_b
            
        T *= cooling
        if T < 0.1:
            T = 0.1
            # Restart with new random init if stuck?
            # For now, just keep going.

    return None

def _perms_to_edges(perms, M):
    edges = []
    for i in range(M):
        for p in perms:
            edges.append((i, M + p[i]))
    return 2*M, edges
