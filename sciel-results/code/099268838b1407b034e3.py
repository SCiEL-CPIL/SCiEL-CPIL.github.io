"""Construct k-regular graph of girth g using Benson construction and random Cayley graphs."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque
import math

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: 
    1. Try Benson-style construction (unitriangular matrices over GF(q))
    2. Try random Cayley graphs on (Z/2Z)^d
    3. Fall back to L-U-W construction
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin for verification and overhead
    
    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction as fallback
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
    
    # Benson construction using unitriangular matrices over GF(q)
    def benson_construct(k, g):
        """
        Benson-style construction using unitriangular matrices over GF(q).
        For GF(2): 4x4 unitriangular matrices, group size 2^6=64
        The coset graph has degree related to the number of generators.
        
        For odd girth, we use a non-bipartite construction.
        """
        # For k=5, g=9, try GF(2) and GF(3)
        # GF(2): 4x4 unitriangular, 6 degrees of freedom, |G|=64
        # GF(3): 4x4 unitriangular, 6 degrees of freedom, |G|=729
        
        best_result = None
        best_n = float('inf')
        
        for q in [2, 3]:
            if time.time() - start_time > time_budget:
                break
            
            # 4x4 unitriangular matrices over GF(q)
            # Matrix form:
            # [1 a b c]
            # [0 1 d e]
            # [0 0 1 f]
            # [0 0 0 1]
            # 6 parameters: a,b,c,d,e,f in GF(q)
            
            # Generate all group elements
            elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        for d in range(q):
                            for e in range(q):
                                for f in range(q):
                                    elements.append((a, b, c, d, e, f))
            
            n_elements = len(elements)
            
            # Define group multiplication
            def mul_elem(x, y):
                """Multiply two unitriangular matrices represented as (a,b,c,d,e,f)"""
                ax, bx, cx, dx, ex, fx = x
                ay, by, cy, dy, ey, fy = y
                
                # Matrix multiplication mod q
                # x = [[1,a,b,c],[0,1,d,e],[0,0,1,f],[0,0,0,1]]
                # y = [[1,ay,by,cy],[0,1,dy,ey],[0,0,1,fy],[0,0,0,1]]
                
                # Result:
                # [1, a+ay, b+by+a*dy, c+cy+a*ey+b*fy+d*fy]
                # [0, 1,    d+dy,     e+ey+d*fy]
                # [0, 0,    1,        f+fy]
                # [0, 0,    0,        1]
                
                na = (ax + ay) % q
                nb = (bx + by + ax * dy) % q
                nc = (cx + cy + ax * ey + bx * fy + dx * fy) % q
                nd = (dx + dy) % q
                ne = (ex + ey + dx * fy) % q
                nf = (fx + fy) % q
                
                return (na, nb, nc, nd, ne, nf)
            
            # Build multiplication table
            mult_table = {}
            for i, x in enumerate(elements):
                for j, y in enumerate(elements):
                    mult_table[(i, j)] = elements.index(mul_elem(x, y))
            
            # Inverse of element
            def inv_elem(x):
                """Inverse of unitriangular matrix (a,b,c,d,e,f)"""
                ax, bx, cx, dx, ex, fx = x
                # Inverse of [[1,a,b,c],[0,1,d,e],[0,0,1,f],[0,0,0,1]]
                # = [[1,-a,ad-b,-(ae+af+cd-2bd+ce)],[0,1,-d,de-e],[0,0,1,-f],[0,0,0,1]]
                # Wait, let me compute properly
                # For unitriangular 4x4:
                # inv = [[1, -a, a*d-b, -(a*e+a*f+c*d-2*b*d+c*e)],
                #        [0, 1, -d, d*e-e],  # This is wrong
                #        [0, 0, 1, -f],
                #        [0, 0, 0, 1]]
                
                # Let me just compute by finding y such that x*y = identity
                # Identity is (0,0,0,0,0,0)
                for y in elements:
                    if mul_elem(x, y) == (0, 0, 0, 0, 0, 0):
                        return y
                raise ValueError("No inverse found")
            
            # Try different sets of generators
            # We need k generators that are involutions (order 2)
            # In this group, elements of order 2 satisfy x*x = identity
            
            involutions = []
            for i, x in enumerate(elements):
                if x == (0, 0, 0, 0, 0, 0):
                    continue
                if mul_elem(x, x) == (0, 0, 0, 0, 0, 0):
                    involutions.append(i)
            
            if len(involutions) < k:
                continue
            
            # Try random subsets of k involutions
            random.seed(42)
            trials = 500
            
            for trial in range(trials):
                if time.time() - start_time > time_budget:
                    break
                
                gens = random.sample(involutions, k)
                
                # Verify girth using non-backtracking BFS
                # State: (vertex, last_gen_idx)
                # Start: for each gen i, state (gens[i], i) at depth 1
                # Goal: reach state (0, i) for any i with depth < g
                
                min_depth = {}
                queue = deque()
                
                for gi, gen_idx in enumerate(gens):
                    v = mult_table[(0, gen_idx)]  # 0 * gen = gen
                    state = (v, gi)
                    min_depth[state] = 1
                    queue.append((v, gi, 1))
                
                valid = True
                while queue:
                    v, last_gen, depth = queue.popleft()
                    
                    if depth >= g - 1:
                        continue
                    
                    for gi, gen_idx in enumerate(gens):
                        if gi == last_gen:
                            continue
                        new_v = mult_table[(v, gen_idx)]
                        new_state = (new_v, gi)
                        new_depth = depth + 1
                        
                        if new_state not in min_depth:
                            min_depth[new_state] = new_depth
                            if new_v == 0 and new_depth < g:
                                valid = False
                                break
                            queue.append((new_v, gi, new_depth))
                    
                    if not valid:
                        break
                
                if valid and n_elements < best_n:
                    best_n = n_elements
                    # Build edge list
                    edges = []
                    for v in range(n_elements):
                        for gen_idx in gens:
                            u = mult_table[(v, gen_idx)]
                            if v < u:
                                edges.append((v, u))
                    best_result = (n_elements, edges)
        
        return best_result
    
    # Random Cayley graph search on (Z/2Z)^d
    def cayley_search(k, g):
        """Try random Cayley graphs on (Z/2Z)^d with k involutory generators."""
        def verify_girth_cayley(gens, g, d):
            """Check if Cayley graph on (Z/2Z)^d with generators gens has girth >= g."""
            min_depth = {}
            queue = deque()
            
            for i in range(len(gens)):
                v = gens[i]
                state = (v, i)
                min_depth[state] = 1
                queue.append((v, i, 1))
            
            while queue:
                v, last_gen, depth = queue.popleft()
                
                if depth >= g - 1:
                    continue
                
                for gi in range(len(gens)):
                    if gi == last_gen:
                        continue
                    new_v = v ^ gens[gi]
                    new_state = (new_v, gi)
                    new_depth = depth + 1
                    
                    if new_state not in min_depth:
                        min_depth[new_state] = new_depth
                        if new_v == 0 and new_depth < g:
                            return False
                        queue.append((new_v, gi, new_depth))
            
            return True
        
        def generate_random_generators(d, k):
            """Generate k random distinct non-zero elements of (Z/2Z)^d."""
            gens = set()
            while len(gens) < k:
                x = random.randint(1, (1 << d) - 1)
                gens.add(x)
            return list(gens)
        
        def build_edges_cayley(gens, d):
            """Build edge list for Cayley graph on (Z/2Z)^d with generators gens."""
            n = 1 << d
            edges = []
            for v in range(n):
                for gen in gens:
                    u = v ^ gen
                    if v < u:
                        edges.append((v, u))
            return n, edges
        
        best_result = None
        best_n = float('inf')
        
        min_d = max(1, mb.bit_length() - 1)
        max_d = min(20, 15)
        
        random.seed(42)
        
        for d in range(min_d, max_d + 1):
            if time.time() - start_time > time_budget:
                break
            
            n = 1 << d
            if n < mb:
                continue
            
            trials = 2000 if d <= 10 else 500
            
            for trial in range(trials):
                if time.time() - start_time > time_budget:
                    break
                
                gens = generate_random_generators(d, k)
                
                if verify_girth_cayley(gens, g, d):
                    if n < best_n:
                        best_n = n
                        best_result = build_edges_cayley(gens, d)
        
        return best_result
    
    # Try all strategies and pick the best
    results = []
    
    # 1. Try Benson construction
    benson_result = benson_construct(k, g)
    if benson_result is not None:
        results.append(benson_result)
    
    # 2. Try Cayley graph search
    cayley_result = cayley_search(k, g)
    if cayley_result is not None:
        results.append(cayley_result)
    
    # 3. L-U-W fallback
    luw_result = luw_construct(k, g)
    results.append(luw_result)
    
    # Pick the smallest valid result
    best = min(results, key=lambda x: x[0])
    
    return best
# EVOLVE-BLOCK-END
