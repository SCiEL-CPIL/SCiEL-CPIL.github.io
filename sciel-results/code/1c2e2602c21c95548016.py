"""Portfolio of explicit cage constructions with fast girth validation."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque
import itertools

def construct(k, g):
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def check_deadline():
        return time.time() > deadline

    # --- Fast girth check using non-backtracking BFS from directed edges ---
    def validate_fast(n, edges, k, g):
        if n <= 0:
            return False
        # Build adjacency and check simplicity + regularity
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        
        # Non-backtracking walk BFS from each directed edge
        # State: (current_vertex, prev_vertex, depth)
        # We want to find if any directed edge (u,v) can reach (v,u) in < g-1 steps
        # without backtracking. If so, cycle length <= depth+1 < g.
        
        for s in range(n):
            # BFS from vertex s, tracking for each vertex the set of "forbidden" prev vertices
            # Use: dist[v] = shortest non-backtracking walk length from s to v
            # But we need to be careful: multiple paths may reach v with different prevs
            # Use a queue of (vertex, prev_vertex, depth)
            visited = {}  # (v, prev) -> depth
            q = deque()
            for nb in adj[s]:
                visited[(nb, s)] = 1
                q.append((nb, s, 1))
            while q:
                u, prev, d = q.popleft()
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue  # non-backtracking
                    if (v, u) in visited:
                        continue
                    # Check if this creates a short cycle
                    # If v == s, then we have a cycle of length d+1
                    if v == s:
                        if d + 1 < g:
                            return False
                    visited[(v, u)] = d + 1
                    q.append((v, u, d + 1))
        return True

    # --- Family 1: Lazebnik-Ustimenko-Woldar with parameter search ---
    def luw_family():
        """
        LUW construction: vertices are (side, x) where side in {0,1}, x in F_q^{d+1}
        Edges connect (0, x) to (1, y) where y_c = x_c + prod_{i in S_c} x_i
        and (1, x) to (0, y) where y_c = x_c - prod_{i in S_c} x_i
        The sets S_c determine the girth.
        """
        q = k  # field size = degree
        
        # For q=4, use GF(4) with special tables
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        elif q == 2:
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: x*y
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
        # Search over different d and rule patterns
        # d is the dimension of the vector part
        # For girth g, we need d >= (g-1)/2 roughly
        
        min_d = max(3, (g - 3) // 2)
        max_d = min(min_d + 3, 15)  # limit search space
        
        best_n = None
        best_edges = None
        best_d = None
        
        for d in range(min_d, max_d + 1):
            if check_deadline():
                break
                
            # Try different rule patterns
            # The rule for each coordinate c specifies which previous coordinates to multiply
            # For good girth, we want "sparse" rules
            
            # Pattern 1: original LUW
            rule1 = {}
            rule1[2] = (1, 1)
            rule1[3] = (2, 1)
            rule1[4] = (1, 2)
            for c in range(5, d+1):
                r = (c-5) % 4
                c0 = c - r
                pattern = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)]
                if r < len(pattern):
                    rule1[c] = pattern[r]
                else:
                    rule1[c] = (1, c-1)
            
            # Pattern 2: shifted
            rule2 = {}
            rule2[2] = (1, 1)
            rule2[3] = (1, 2)
            rule2[4] = (2, 1)
            for c in range(5, d+1):
                r = (c-4) % 4
                c0 = c - r
                pattern = [(1,c0-1),(c0,1),(1,c0),(c0-2,1)]
                if r < len(pattern):
                    rule2[c] = pattern[r]
                else:
                    rule2[c] = (1, c-1)
            
            # Pattern 3: alternating
            rule3 = {}
            rule3[2] = (1, 1)
            rule3[3] = (2, 1)
            rule3[4] = (1, 3)
            for c in range(5, d+1):
                if c % 2 == 0:
                    rule3[c] = (1, c-1)
                else:
                    rule3[c] = (c-2, 1)
            
            # Pattern 4: sparse - use only recent coordinates
            rule4 = {}
            rule4[2] = (1, 1)
            rule4[3] = (1, 2)
            rule4[4] = (2, 3)
            for c in range(5, d+1):
                rule4[c] = (c-2, c-1)
            
            rules_to_try = [rule1, rule2, rule3, rule4]
            
            for rule in rules_to_try:
                if check_deadline():
                    break
                    
                def neighbours(vertex, rule=rule):
                    side, x = vertex
                    x_list = list(x)
                    for first in range(q):
                        y = [None, first] + [0]*d
                        for c in range(2, d+1):
                            a, b = rule[c]
                            if side == 0:
                                y[c] = add(x_list[c], mul(y[a], x_list[b]))
                            else:
                                y[c] = sub(x_list[c], mul(x_list[a], y[b]))
                        yield 1-side, tuple(y)
                
                start = (0, (None,)+(0,)*d)
                index, order, edges = {start:0}, [start], []
                
                # BFS to build the graph
                for vertex in order:
                    for other in neighbours(vertex):
                        if other not in index:
                            index[other] = len(order)
                            order.append(other)
                        if vertex[0] == 0:
                            edges.append((index[vertex], index[other]))
                
                n = len(order)
                
                # Quick check: is it k-regular?
                # In LUW, each vertex should have exactly q neighbors
                # But we need to verify
                if n > 0 and n <= (best_n if best_n else float('inf')):
                    # Validate
                    if validate_fast(n, edges, k, g):
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                            best_d = d
                        break  # Found valid graph for this rule, try next d
        
        if best_n is not None:
            return best_n, best_edges
        
        # Fallback: use the first rule with d = min_d
        d = min_d
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d+1):
            r = (c-5) % 4
            c0 = c - r
            pattern = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)]
            if r < len(pattern):
                rule[c] = pattern[r]
            else:
                rule[c] = (1, c-1)
        
        def neighbours(vertex, rule=rule):
            side, x = vertex
            x_list = list(x)
            for first in range(q):
                y = [None, first] + [0]*d
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x_list[c], mul(y[a], x_list[b]))
                    else:
                        y[c] = sub(x_list[c], mul(x_list[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # --- Family 2: Circulant-based construction ---
    def circulant_family():
        # Try to find a circulant graph C(n, S) that is k-regular with girth >= g
        # For k=5, we need 5 distinct jumps (or symmetric pairs)
        
        best_n = None
        best_edges = None
        
        # Moore bound for girth g, degree k
        moore = 1 + k
        for i in range(1, (g-1)//2 + 1):
            moore += k * (k-1)**i
        
        # Try circulant constructions
        for n in range(moore, moore + 100):
            if check_deadline():
                break
            
            # For k=5, we need exactly 5 neighbors per vertex
            # In a circulant, this means 5 distinct absolute jump values
            # (or combinations with self-inverse jumps)
            
            # Try systematic search for small jump sets
            # This is expensive, so limit the search
            found = False
            for j1 in range(1, min(50, n//4 + 1)):
                if check_deadline():
                    break
                for j2 in range(j1+1, min(100, n//3 + 1)):
                    if j2 == n - j2 and n % 2 != 0:
                        continue
                    for j3 in range(j2+1, min(150, n//2 + 1)):
                        if j3 == n - j3 and n % 2 != 0:
                            continue
                        for j4 in range(j3+1, min(200, n//2 + 1)):
                            if j4 == n - j4 and n % 2 != 0:
                                continue
                            for j5 in range(j4+1, min(250, n//2 + 1)):
                                if j5 == n - j5 and n % 2 != 0:
                                    continue
                                
                                jumps = [j1, j2, j3, j4, j5]
                                # Check if this gives a valid graph
                                # Build edges
                                edges = []
                                seen_edges = set()
                                valid = True
                                for i in range(n):
                                    for j in jumps:
                                        u = i
                                        v = (i + j) % n
                                        key = (min(u,v), max(u,v))
                                        if key in seen_edges:
                                            valid = False
                                            break
                                        seen_edges.add(key)
                                        edges.append((u, v))
                                    if not valid:
                                        break
                                
                                if valid and len(edges) == n * k // 2:
                                    # Check girth
                                    if validate_fast(n, edges, k, g):
                                        if best_n is None or n < best_n:
                                            best_n = n
                                            best_edges = edges
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        return best_n, best_edges

    # --- Family 3: Algebraic construction using finite fields ---
    def algebraic_family():
        # Try a construction based on the incidence graph of a projective plane
        # Or use a known small cage if available
        # For now, return None
        return None, None

    # Try all families and keep the best valid one
    candidates = []
    
    # LUW is the primary family
    n_luw, e_luw = luw_family()
    if n_luw is not None and e_luw is not None:
        candidates.append((n_luw, e_luw))
    
    # Try other families
    n_circ, e_circ = circulant_family()
    if n_circ is not None and e_circ is not None:
        candidates.append((n_circ, e_circ))
    
    n_alg, e_alg = algebraic_family()
    if n_alg is not None and e_alg is not None:
        candidates.append((n_alg, e_alg))
    
    # Validate each candidate and keep the smallest valid one
    best_n = None
    best_edges = None
    
    for n, e in sorted(candidates, key=lambda x: x[0]):
        if check_deadline():
            break
        if validate_fast(n, e, k, g):
            best_n = n
            best_edges = e
            break
    
    # If no valid candidate found, return LUW as fallback
    if best_n is None and n_luw is not None:
        best_n = n_luw
        best_edges = e_luw
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
