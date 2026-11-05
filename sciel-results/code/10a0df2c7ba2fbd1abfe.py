"""Improved program: construct k-regular graph of girth g using random Cayley graphs on non-abelian groups."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import deque
import sys

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: 
    1. Try random Cayley graph search on non-abelian groups (products of symmetric groups)
    2. Fall back to the Lazebnik-Ustimenko-Woldar construction if no better found
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    time_start = time.time()
    TIME_LIMIT = 110  # Leave 10s buffer for verification
    
    # Moore bound calculation
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction as reliable fallback
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
    
    # Get the L-U-W baseline
    luw_n, luw_edges = luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Random Cayley graph search on non-abelian groups
    # For k=5, g=9: try groups like S_m x S_m, or A_m, etc.
    # We need k involutory generators (since we want a simple graph, each generator is its own inverse)
    
    def verify_girth(n, edges, g):
        """Verify that the graph has girth at least g using BFS from each vertex."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS tracking parent to detect cycles
        # A cycle of length < g would be found if we reach a visited node at distance < g-1
        # More precisely: from vertex v, do BFS. If we find a node already visited 
        # at distance d1, and current path has distance d2, and d1 + d2 + 1 < g, 
        # we have a cycle shorter than g.
        
        # Better approach: for each vertex, do BFS and check if any neighbor 
        # is reached via a path that creates a cycle of length < g.
        
        # Actually, the standard way: for each vertex s, do BFS. When we visit 
        # a node u at distance dist[u], if u is already visited from a different 
        # parent path creating a shorter cycle, we detect it.
        
        # Simplest correct approach: for each vertex s, do BFS. If we find a 
        # cycle of length L < g, return False.
        
        # For efficiency, we can limit BFS depth to g-2 (since a cycle of length 
        # g-1 or less would be detected).
        
        for s in range(n):
            # BFS from s, tracking the path
            visited = [-1] * n  # -1 means unvisited, otherwise distance
            parent = [-1] * n
            visited[s] = 0
            queue = deque([(s, 0)])
            
            while queue:
                u, dist = queue.popleft()
                
                # Limit depth: if dist >= g-1, no need to go further
                # (a cycle through s would need to return to s, so max useful depth is g-2)
                if dist >= g - 1:
                    continue
                    
                for v in adj[u]:
                    if v == s and dist + 1 >= g:
                        # This would be a cycle of length dist+1 >= g, OK
                        continue
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        parent[v] = u
                        queue.append((v, dist + 1))
                    elif v != parent[u] and v != s:
                        # Found a back edge: cycle length = dist + 1 + visited[v]
                        cycle_len = dist + 1 + visited[v]
                        if cycle_len < g:
                            return False
                    elif v == s:
                        # Cycle back to start
                        cycle_len = dist + 1
                        if cycle_len < g:
                            return False
            # Check for self-loops or multi-edges (shouldn't happen in simple graph)
        
        return True
    
    # Try random Cayley graphs on various non-abelian groups
    # For k=5, g=9, let's try:
    # - S_5 (order 120), S_6 (order 720), S_7 (order 5040)
    # - Products like S_4 x S_4, etc.
    
    # For simplicity and speed, let's try small symmetric groups
    # and check if we can find 5 involutions that generate a graph with girth >= 9
    
    # In S_n, involutions are elements of order 2: products of disjoint transpositions
    # For S_5: involutions are single transpositions (10 of them) or products of 2 disjoint transpositions (15 of them)
    
    def get_involutions_Sn(n):
        """Get all involutions in S_n as tuples representing permutations."""
        invs = set()
        # Single transpositions
        for i in range(n):
            for j in range(i+1, n):
                perm = list(range(n))
                perm[i], perm[j] = perm[j], perm[i]
                invs.add(tuple(perm))
        # Products of 2 disjoint transpositions (if n >= 4)
        if n >= 4:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            perm = list(range(n))
                            perm[i], perm[j] = perm[j], perm[i]
                            perm[k], perm[l] = perm[l], perm[k]
                            invs.add(tuple(perm))
        # Products of 3 disjoint transpositions (if n >= 6)
        if n >= 6:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            for m in range(l+1, n):
                                for p in range(m+1, n):
                                    perm = list(range(n))
                                    perm[i], perm[j] = perm[j], perm[i]
                                    perm[k], perm[l] = perm[l], perm[k]
                                    perm[m], perm[p] = perm[p], perm[m]
                                    invs.add(tuple(perm))
        return list(invs)
    
    def perm_compose(p, q):
        """Compose permutations p and q: (p ∘ q)(x) = p(q(x))"""
        n = len(p)
        return tuple(p[q[i]] for i in range(n))
    
    def perm_invert(p):
        """Invert permutation p"""
        n = len(p)
        inv = [0] * n
        for i in range(n):
            inv[p[i]] = i
        return tuple(inv)
    
    def try_cayley_on_Sn(n_sym, k, g, time_limit):
        """Try to find a Cayley graph on S_{n_sym} with k involutory generators and girth >= g."""
        if len(get_involutions_Sn(n_sym)) < k:
            return None
        
        invs = get_involutions_Sn(n_sym)
        group_size = len(invs)  # This is wrong, group size is n_sym!
        
        # Actually, the group S_{n_sym} has n_sym! elements
        # For n_sym=5, that's 120; for n_sym=6, that's 720; for n_sym=7, that's 5040
        
        # We need to pick k involutions from the set of all involutions in S_{n_sym}
        # and check if the Cayley graph has girth >= g
        
        # For efficiency, let's limit the number of random trials
        max_trials = 50 if n_sym <= 6 else 20
        
        for trial in range(max_trials):
            if time.time() - time_start > time_limit:
                return None
            
            # Randomly select k involutions
            selected = random.sample(invs, k)
            
            # Build the Cayley graph
            # Vertices are all permutations in S_{n_sym}
            # For small n_sym, we can enumerate them
            
            if n_sym > 7:
                return None  # Too large to enumerate
            
            # Generate all permutations
            all_perms = list(itertools.permutations(range(n_sym)))
            perm_to_idx = {p: i for i, p in enumerate(all_perms)}
            group_order = len(all_perms)
            
            # Build adjacency: for each vertex (permutation) and each generator, 
            # the neighbor is the product
            adj = [[] for _ in range(group_order)]
            for i, p in enumerate(all_perms):
                for gen in selected:
                    # Neighbor is p * gen (right multiplication)
                    q = perm_compose(p, gen)
                    j = perm_to_idx[q]
                    adj[i].append(j)
            
            # Check that it's a simple graph (no self-loops, no multi-edges)
            # and k-regular
            valid = True
            for i in range(group_order):
                if len(adj[i]) != k:
                    valid = False
                    break
                if len(set(adj[i])) != k:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check for self-loops
            for i in range(group_order):
                if i in adj[i]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Build edge list
            edges = []
            for i in range(group_order):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            # Verify girth
            if verify_girth(group_order, edges, g):
                return group_order, edges
        
        return None
    
    # Helper to get involutions for S4
    def get_involutions_S4():
        invs = []
        # Transpositions
        for i in range(4):
            for j in range(i+1, 4):
                p = list(range(4))
                p[i], p[j] = p[j], p[i]
                invs.append(tuple(p))
        # Double transpositions
        p1 = [1, 0, 3, 2]
        p2 = [2, 3, 0, 1]
        p3 = [3, 2, 1, 0]
        invs.append(tuple(p1))
        invs.append(tuple(p2))
        invs.append(tuple(p3))
        return invs

    # Try Cayley graphs on S_4 x S_4 (Order 576)
    if time.time() - time_start < TIME_LIMIT:
        invs_S4 = get_involutions_S4()
        all_perms_S4 = list(itertools.permutations(range(4)))
        perm_to_idx_S4 = {p: i for i, p in enumerate(all_perms_S4)}
        
        # Involutions in S4 x S4 are pairs (a, b) where a, b are involutions in S4
        invs_prod = [(a, b) for a in invs_S4 for b in invs_S4]
        
        # All elements in S4 x S4
        all_elems = [(a, b) for a in all_perms_S4 for b in all_perms_S4]
        elem_to_idx = {e: i for i, e in enumerate(all_elems)}
        group_order = len(all_elems)
        
        # Try random sets of 5 involutions
        max_trials = 1000
        for trial in range(max_trials):
            if time.time() - time_start > TIME_LIMIT:
                break
            selected = random.sample(invs_prod, k)
            
            # Build adjacency
            adj = [[] for _ in range(group_order)]
            valid = True
            for i, e in enumerate(all_elems):
                a, b = e
                for gen in selected:
                    ga, gb = gen
                    # Right multiplication: (a, b) * (ga, gb) = (a*ga, b*gb)
                    # Composition: a(ga(x))
                    na = tuple(a[ga[x]] for x in range(4))
                    nb = tuple(b[gb[x]] for x in range(4))
                    ne = (na, nb)
                    j = elem_to_idx[ne]
                    if j == i:
                        valid = False
                        break
                    adj[i].append(j)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check for multi-edges (duplicates in adj[i])
            for i in range(group_order):
                if len(adj[i]) != k or len(set(adj[i])) != k:
                    valid = False
                    break
            if not valid:
                continue
            
            # Build edge list
            edges = []
            for i in range(group_order):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            # Verify girth
            if verify_girth(group_order, edges, g):
                best_n = group_order
                best_edges = edges
                break

    # Try Cayley graphs on S_5, S_6 with more trials
    # S_5 (120) is too small for girth 9, but we try S_6 (720)
    for n_sym in [6]:
        if time.time() - time_start > TIME_LIMIT:
            break
        # Increase trials for S_6
        invs = get_involutions_Sn(n_sym)
        all_perms = list(itertools.permutations(range(n_sym)))
        perm_to_idx = {p: i for i, p in enumerate(all_perms)}
        group_order = len(all_perms)
        
        max_trials = 500
        for trial in range(max_trials):
            if time.time() - time_start > TIME_LIMIT:
                break
            selected = random.sample(invs, k)
            
            adj = [[] for _ in range(group_order)]
            valid = True
            for i, p in enumerate(all_perms):
                for gen in selected:
                    q = perm_compose(p, gen)
                    j = perm_to_idx[q]
                    if j == i:
                        valid = False
                        break
                    adj[i].append(j)
                if not valid:
                    break
            
            if not valid:
                continue
            
            for i in range(group_order):
                if len(adj[i]) != k or len(set(adj[i])) != k:
                    valid = False
                    break
            if not valid:
                continue
            
            edges = []
            for i in range(group_order):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            if verify_girth(group_order, edges, g):
                if group_order < best_n:
                    best_n = group_order
                    best_edges = edges
                break

    # Return the best result found
    return best_n, best_edges
# EVOLVE-BLOCK-END
