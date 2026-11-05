"""Construct k-regular graph of girth g using random Cayley graphs on non-abelian groups."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: 
    1. Try random Cayley graph approach on non-abelian groups (product of S_3, etc.)
    2. Fall back to L-U-W construction if random search fails within time budget.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s margin for overhead
    
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W fallback
    def luw_construct(k, g):
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
    
    # Random Cayley graph approach on S_3^m
    # S_3 has 6 elements, we can form S_3^m with 6^m elements
    # Generators: pick k involutions from S_3^m
    # S_3 elements: identity, (12), (13), (23), (123), (132)
    # Involutions in S_3: (12), (13), (23) - order 2
    # In S_3^m, involutions are tuples where each component is either identity or a transposition
    # and the total number of non-identity components is even (for order 2)
    
    # Actually, let's think more carefully. In S_3, the elements of order 2 are the three transpositions.
    # In S_3^m, an element has order 2 if each component has order dividing 2, and at least one has order 2.
    # Components of order dividing 2: identity and the three transpositions.
    
    # For a Cayley graph to be k-regular, we need k generators, all of which must be involutions
    # (order 2) so that the graph is undirected.
    
    # Let's try S_3^m for small m and see if we can find k=5 involutions that give girth >= 9.
    
    # Represent S_3^m elements as tuples of length m, each entry in {0,1,2,3,4,5}
    # where 0=identity, 1=(12), 2=(13), 3=(23), 4=(123), 5=(132)
    
    # Multiplication table for S_3:
    # 0: identity
    # 1: (12)
    # 2: (13)  
    # 3: (23)
    # 4: (123)
    # 5: (132)
    
    # Let me build the multiplication table
    def build_s3_table():
        # Permutations as tuples (p0, p1, p2) meaning i -> p[i]
        # 0: (0,1,2) identity
        # 1: (1,0,2) (12)
        # 2: (2,1,0) (13)
        # 3: (0,2,1) (23)
        # 4: (1,2,0) (123)
        # 5: (2,0,1) (132)
        perms = [
            (0,1,2),  # 0: id
            (1,0,2),  # 1: (12)
            (2,1,0),  # 2: (13)
            (0,2,1),  # 3: (23)
            (1,2,0),  # 4: (123)
            (2,0,1),  # 5: (132)
        ]
        table = [[0]*6 for _ in range(6)]
        for a in range(6):
            for b in range(6):
                # compose a * b: apply b first, then a
                pa, pb = perms[a], perms[b]
                result = tuple(pa[pb[i]] for i in range(3))
                for c in range(6):
                    if perms[c] == result:
                        table[a][b] = c
                        break
        return table
    
    s3_table = build_s3_table()
    
    def mul_s3(a, b):
        return s3_table[a][b]
    
    def inv_s3(a):
        # Inverse in S_3
        # 0->0, 1->1, 2->2, 3->3, 4->5, 5->4
        inv = {0:0, 1:1, 2:2, 3:3, 4:5, 5:4}
        return inv[a]
    
    def is_involution_s3(a):
        return s3_table[a][a] == 0 and a != 0
    
    # For S_3^m, multiplication is component-wise
    def mul_s3m(a, b, m):
        return tuple(s3_table[a[i]][b[i]] for i in range(m))
    
    def inv_s3m(a, m):
        return tuple(inv_s3(a[i]) for i in range(m))
    
    def is_involution_s3m(a, m):
        if any(a[i] not in (1,2,3) for i in range(m)):
            return False
        # Check that applying twice gives identity
        return all(s3_table[a[i]][a[i]] == 0 for i in range(m))
    
    # For k=5, g=9, try S_3^m for m=3,4,5,...
    # n = 6^m
    # We need to pick 5 involutions from S_3^m that generate a Cayley graph with girth >= 9
    
    # To check girth, we do BFS from identity and look for the shortest cycle
    # A cycle of length L exists if there's a non-trivial word of length L in the generators that equals identity
    
    # For Cayley graph, girth is the length of the shortest non-trivial relation
    # i.e., shortest word g_{i1} g_{i2} ... g_{iL} = e where consecutive generators are distinct
    # (to avoid backtracking)
    
    def check_girth_cayley(generators, m, girth_target):
        """Check if Cayley graph on S_3^m with given generators has girth >= girth_target."""
        n = 6**m
        # BFS from identity, tracking (element, previous_generator)
        # State: (element_tuple, last_gen_index)
        # We want to find shortest path from (e, -1) to (e, j) for any j != -1
        # where the path doesn't use the same generator consecutively
        
        identity = tuple([0]*m)
        
        # BFS: distance[element][last_gen] = shortest distance to reach element with last_gen as last step
        # last_gen = -1 means start (no previous generator)
        
        # Use dictionary: key = (element, last_gen), value = distance
        visited = {}
        queue = deque()
        
        start_state = (identity, -1)
        visited[start_state] = 0
        queue.append(start_state)
        
        while queue:
            state, dist = queue.popleft()
            elem, last_gen = state
            
            if dist >= girth_target:
                # We've explored up to girth_target, no shorter cycle found
                return True, girth_target
            
            for gen_idx, gen in enumerate(generators):
                if gen_idx == last_gen:
                    continue  # No backtracking
                
                new_elem = mul_s3m(elem, gen, m)
                new_state = (new_elem, gen_idx)
                
                if new_state in visited:
                    # Found a cycle! Length is dist + 1
                    cycle_len = dist + 1
                    return cycle_len >= girth_target, cycle_len
                else:
                    visited[new_state] = dist + 1
                    queue.append(new_state)
        
        # If we exhausted BFS without finding a cycle (shouldn't happen for finite group)
        return True, girth_target
    
    # Generate random sets of k involutions in S_3^m
    def get_all_involutions_s3m(m):
        """Get all non-identity involutions in S_3^m."""
        involutions = []
        # Each component is 0,1,2,3
        # Must have at least one non-zero component
        # And each non-zero component must be an involution (1,2,3 are all involutions in S_3)
        for combo in itertools.product([0,1,2,3], repeat=m):
            if all(c == 0 for c in combo):
                continue
            # All components with value in {1,2,3} are involutions
            # So this is an involution iff all non-zero components are in {1,2,3}
            # which is always true by construction
            involutions.append(combo)
        return involutions
    
    # For k=5, g=9: try various m values
    # n = 6^m, so m=3 gives 216, m=4 gives 1296, m=5 gives 7776
    # Moore bound for k=5, g=9: 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # So we need n >= 426, meaning m >= 4 (6^3=216 < 426)
    
    best_n = None
    best_generators = None
    best_m = None
    
    # Try m=4 first (n=1296), then m=5 if needed
    for m in [4, 5]:
        if time.time() - start_time > time_limit - 10:
            break
        
        n = 6**m
        if n < mb:
            continue
        
        involutions = get_all_involutions_s3m(m)
        num_inv = len(involutions)
        
        if num_inv < k:
            continue
        
        # Try random subsets of k involutions
        max_attempts = 50 if m == 4 else 20
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit - 5:
                break
            
            random.seed(attempt * 1000 + m)
            generators = random.sample(involutions, k)
            
            # Check if these generators generate the full group
            # and check girth
            ok, cycle_len = check_girth_cayley(generators, m, g)
            
            if ok:
                # Verify they generate the full group
                generated = {tuple([0]*m)}
                frontier = [tuple([0]*m)]
                while frontier:
                    new_frontier = []
                    for elem in frontier:
                        for gen in generators:
                            new_elem = mul_s3m(elem, gen, m)
                            if new_elem not in generated:
                                generated.add(new_elem)
                                new_frontier.append(new_elem)
                    frontier = new_frontier
                
                if len(generated) == n:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_generators = generators
                        best_m = m
                    break  # Found valid for this m, try smaller m if possible
        
        if best_n is not None and best_n <= 6**4:
            break
    
    # Build the graph from best Cayley construction
    if best_n is not None and best_generators is not None:
        m = best_m
        n = 6**m
        generators = best_generators
        
        # Enumerate all elements of S_3^m
        elements = list(itertools.product(range(6), repeat=m))
        elem_to_idx = {elem: i for i, elem in enumerate(elements)}
        
        edges = []
        for i, elem in enumerate(elements):
            for gen in generators:
                neighbor = mul_s3m(elem, gen, m)
                j = elem_to_idx[neighbor]
                if i < j:
                    edges.append((i, j))
        
        return n, edges
    
    # Fallback to L-U-W
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
