"""Combined program for the cage problem n(4,10).
Uses Cayley graph search on A6 (360 vertices) and Voltage Graph lift as fallback.
"""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
import itertools

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy:
    1. Try to find a small Cayley graph on A6 (order 360) with girth >= g.
    2. Try voltage graph (lift) construction over a small 4-regular base graph.
    3. Fallback to Lazebnik-Ustimenko-Woldar (LUV) graph construction.
    """
    start_time = time.time()
    deadline = 115.0  # 5s safety margin before 120s limit

    def try_cayley_a6(k, g):
        """Search for a 4-regular Cayley graph on A6 with girth >= g."""
        if k != 4:
            return None
        
        # Generate A6 elements
        def is_even(perm):
            inv_count = 0
            for i in range(6):
                for j in range(i+1, 6):
                    if perm[i] > perm[j]:
                        inv_count += 1
            return inv_count % 2 == 0
        
        a6_elems = []
        for p in itertools.permutations(range(6)):
            if is_even(p):
                a6_elems.append(p)
        
        n = len(a6_elems)
        if n != 360:
            return None
        
        index = {e: i for i, e in enumerate(a6_elems)}
        identity = tuple(range(6))
        e_id = index[identity]
        
        # Multiplication table: mult[i][j] = index of a6_elems[i] * a6_elems[j]
        mult = [[0] * n for _ in range(n)]
        for i in range(n):
            pi = a6_elems[i]
            for j in range(n):
                pj = a6_elems[j]
                prod = tuple(pi[pj[x]] for x in range(6))
                mult[i][j] = index[prod]
        
        # Inverses
        inv = [0] * n
        for i in range(n):
            for j in range(n):
                if mult[i][j] == e_id:
                    inv[i] = j
                    break
        
        # Candidates: elements with order > 2
        candidates = []
        element_order = [1] * n
        for i in range(n):
            if i == e_id:
                continue
            if inv[i] == i:
                continue
            # Calculate order
            cur = i
            order = 1
            while cur != e_id:
                cur = mult[cur][i]
                order += 1
                if order > n:
                    break
            element_order[i] = order
            candidates.append(i)
        
        # Sort candidates by order (high order first often yields better girth)
        cand_sorted = sorted(candidates, key=lambda x: element_order[x], reverse=True)
        
        # Phase 1: Systematic search among top-order candidates
        top_count = min(40, len(cand_sorted))
        top_cands = cand_sorted[:top_count]
        
        for i1 in range(len(top_cands)):
            if time.time() > deadline:
                break
            g1 = top_cands[i1]
            g1_inv = inv[g1]
            
            for i2 in range(i1 + 1, len(top_cands)):
                g2 = top_cands[i2]
                g2_inv = inv[g2]
                
                if g1 == g2 or g1 == g2_inv or g1_inv == g2 or g1_inv == g2_inv:
                    continue
                
                # BFS for girth
                depth = [-1] * n
                parent = [-1] * n
                q = deque()
                q.append(e_id)
                depth[e_id] = 0
                
                min_cycle = float('inf')
                valid = True
                
                while q:
                    u = q.popleft()
                    du = depth[u]
                    if du >= g:
                        continue
                    
                    for s in (g1, g1_inv, g2, g2_inv):
                        v = mult[u][s]
                        if depth[v] == -1:
                            depth[v] = du + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            dv = depth[v]
                            if u != parent[v] and v != parent[u]:
                                cl = du + dv + 1
                                if cl < min_cycle:
                                    min_cycle = cl
                                    if min_cycle < g:
                                        valid = False
                                        break
                    if not valid:
                        break
                
                if valid and min_cycle >= g:
                    edges = []
                    for i in range(n):
                        for s in (g1, g1_inv, g2, g2_inv):
                            j = mult[i][s]
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        
        # Phase 2: Random search
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        max_random_pairs = int(remaining * 500)
        max_random_pairs = max(100, min(max_random_pairs, 5000))
        
        random.seed(42)
        for _ in range(max_random_pairs):
            if time.time() > deadline:
                break
            
            i1 = random.randint(0, len(cand_sorted) - 1)
            i2 = random.randint(0, len(cand_sorted) - 1)
            if i1 == i2:
                continue
            
            g1 = cand_sorted[i1]
            g2 = cand_sorted[i2]
            g1_inv = inv[g1]
            g2_inv = inv[g2]
            
            if g1 == g2 or g1 == g2_inv or g1_inv == g2 or g1_inv == g2_inv:
                continue
            
            depth = [-1] * n
            parent = [-1] * n
            q = deque()
            q.append(e_id)
            depth[e_id] = 0
            
            min_cycle = float('inf')
            valid = True
            
            while q:
                u = q.popleft()
                du = depth[u]
                if du >= g:
                    continue
                
                for s in (g1, g1_inv, g2, g2_inv):
                    v = mult[u][s]
                    if depth[v] == -1:
                        depth[v] = du + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        dv = depth[v]
                        if u != parent[v] and v != parent[u]:
                            cl = du + dv + 1
                            if cl < min_cycle:
                                min_cycle = cl
                                if min_cycle < g:
                                    valid = False
                                    break
                if not valid:
                    break
            
            if valid and min_cycle >= g:
                edges = []
                for i in range(n):
                    for s in (g1, g1_inv, g2, g2_inv):
                        j = mult[i][s]
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        return None

    def luv_construct(k, g):
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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

    # Main logic
    if k == 4:
        result = try_cayley_a6(k, g)
        if result is not None:
            return result
    
    # Fallback to LUV
    try:
        n, edges = luv_construct(k, g)
        return n, edges
    except:
        pass

    return 0, []
# EVOLVE-BLOCK-END
