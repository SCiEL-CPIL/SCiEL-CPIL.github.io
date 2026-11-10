import time
import random
from collections import deque

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

def _sl2_construct(k, g, deadline):
    if k != 3:
        return None
    
    # We focus on q=9 for n=720
    q = 9
    if q * (q*q - 1) >= 936: # If it's not better than known best, skip? 
        # Actually 720 < 936, so we try.
        pass

    # Build SL(2,q)
    elements = []
    elem_index = {}
    # q=9 is not prime, but SL(2,9) is well-defined.
    # We iterate over all 4x4 matrices with det 1.
    # a,b,c,d in 0..8
    # ad - bc = 1 mod 9
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # d = (1 + bc) * a^-1 mod 9
                # a must be invertible mod 9. gcd(a,9)=1 => a in {1,2,4,5,7,8}
                if a % 3 == 0:
                    continue
                # Find inverse of a mod 9
                a_inv = None
                for x in range(q):
                    if (a * x) % q == 1:
                        a_inv = x
                        break
                if a_inv is None:
                    continue
                d = (1 + b * c) % q * a_inv % q
                elem = (a, b, c, d)
                # Check det
                if (a*d - b*c) % q != 1:
                    continue
                elements.append(elem)
                elem_index[elem] = len(elements) - 1
    
    n = len(elements)
    if n != q * (q*q - 1):
        # Fallback or error
        pass

    identity = (1, 0, 0, 1)
    
    def mat_mul(e1, e2):
        a1, b1, c1, d1 = e1
        a2, b2, c2, d2 = e2
        return ((a1*a2 + b1*c2) % q, (a1*b2 + b1*d2) % q,
                (c1*a2 + d1*c2) % q, (c1*b2 + d1*d2) % q)
        
    # Find involutions
    involutions = []
    for elem in elements:
        if elem == identity:
            continue
        sq = mat_mul(elem, elem)
        if sq == identity:
            involutions.append(elem)
            
    if len(involutions) < 3:
        return None

    # Precompute all non-backtracking words of length < g
    # We need to check if any word evaluates to identity
    # Word is a tuple of generator indices
    # We can just BFS the Cayley graph up to depth g-1 and check for identity?
    # No, BFS from identity with non-backtracking constraint.
    # If we reach identity again at depth < g, girth < g.
    
    def check_girth(gens, g):
        # gens: list of 3 matrices
        # State: (elem_idx, prev_gen_idx)
        # Start: (identity_idx, -1)
        id_idx = elem_index[identity]
        visited = set()
        visited.add((id_idx, -1))
        queue = deque()
        queue.append((id_idx, -1, 0))
        
        while queue:
            idx, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for gi, gen in enumerate(gens):
                if gi == prev:
                    continue
                new_elem = mat_mul(elements[idx], gen)
                new_idx = elem_index[new_elem]
                new_state = (new_idx, gi)
                if new_state == (id_idx, -1):
                    # Found identity at depth+1
                    # Cycle length is depth+1
                    if depth + 1 < g:
                        return False
                if new_state in visited:
                    # Revisit means cycle
                    # Cycle length is at most depth + 1 + (something)
                    # Actually, if we revisit (new_idx, gi), we have a loop in the state space.
                    # The length of the loop is not necessarily the girth, but it implies a cycle.
                    # To be safe, if we revisit ANY state, we have a cycle of length <= depth+1 + dist(start, state).
                    # But we want to detect if girth < g.
                    # If we reach identity at depth < g, girth < g.
                    # If we revisit a state, it means there is a cycle in the non-backtracking walk space.
                    # This cycle lifts to a cycle in the graph.
                    # The length of the cycle in the graph is the length of the walk.
                    # So if we revisit, we have a cycle of length <= depth + 1.
                    # If depth + 1 < g, then girth < g.
                    if depth + 1 < g:
                        return False
                visited.add(new_state)
                queue.append((new_idx, gi, depth + 1))
        
        return True

    random.seed(42)
    # Try random triples
    # To ensure they generate the group, we can check if the BFS visits all n elements?
    # If not, the graph is disconnected.
    
    for trial in range(1000):
        if time.time() > deadline - 5:
            break
        idxs = random.sample(range(len(involutions)), 3)
        gens = [involutions[i] for i in idxs]
        
        # Quick check: generate group?
        # We can run the BFS and count visited elements.
        # Modify check_girth to also return visited count?
        
        # Let's just check girth first. If girth fails, skip.
        # If girth passes, check connectivity.
        
        if check_girth(gens, g):
            # Check connectivity
            id_idx = elem_index[identity]
            visited_elems = set([id_idx])
            queue = deque([id_idx])
            while queue:
                idx = queue.popleft()
                for gen in gens:
                    new_elem = mat_mul(elements[idx], gen)
                    new_idx = elem_index[new_elem]
                    if new_idx not in visited_elems:
                        visited_elems.add(new_idx)
                        queue.append(new_idx)
            if len(visited_elems) == n:
                # Valid
                edges = []
                for i, elem in enumerate(elements):
                    for gen in gens:
                        new_elem = mat_mul(elem, gen)
                        j = elem_index[new_elem]
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 3 or g != 16:
        return _luw_construct(k, g)
    
    result = _sl2_construct(k, g, deadline)
    if result is not None:
        return result
        
    return _luw_construct(k, g)
