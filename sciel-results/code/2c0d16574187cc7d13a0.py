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
    """
    Construct a 3-regular graph as a Cayley graph on SL(2,q) with 3 involutory generators.
    n = q*(q^2-1). Verify girth >= g via non-backtracking BFS.
    """
    if k != 3:
        return None
    
    # Try q = 16 first (n=4080), then q=32 (n=32256), then q=9 (n=720, but girth may be small)
    # q must be p^r. For girth 16, we need n >= 512 (Moore bound).
    # q=9: n=720, q=16: n=4080, q=25: n=6200, q=32: n=32256, q=49: n=117600
    
    candidate_qs = [16, 32, 9, 25, 49, 81, 121]
    
    for q in candidate_qs:
        if time.time() > deadline - 10:
            break
        n = q * (q * q - 1)
        if n < 512:  # below Moore bound, skip
            continue
        if n > 200000:  # too large to handle in time
            continue
        
        # Build SL(2,q) group
        # Element: (a, b, c, d) representing [[a,b],[c,d]] with ad-bc=1 mod q
        # Identity: (1, 0, 0, 1)
        # We need to enumerate all elements and build multiplication table (or use on-the-fly)
        
        # For efficiency, precompute all elements
        elements = []
        elem_index = {}
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # d = (1 + bc) * a^(-1) mod q, but a must be invertible
                    # Actually: ad - bc = 1 => d = (1 + bc) / a mod q
                    # a must be coprime to q for this to work
                    if a == 0:
                        continue
                    # Check if a is invertible mod q
                    # For prime powers, a is invertible iff gcd(a,q)=1
                    import math
                    if math.gcd(a, q) != 1:
                        continue
                    # a_inv = a^(q-2) mod q for prime q, but for prime powers use extended gcd
                    # For simplicity, only handle prime q
                    # Check if q is prime
                    def is_prime(n):
                        if n < 2:
                            return False
                        for i in range(2, int(n**0.5)+1):
                            if n % i == 0:
                                return False
                        return True
                    if not is_prime(q):
                        # For prime powers, use a different approach
                        # Skip for now, only handle prime q
                        break
                    a_inv = pow(a, q-2, q)
                    d = (1 + b * c) % q * a_inv % q
                    elem = (a, b, c, d)
                    elements.append(elem)
                    elem_index[elem] = len(elements) - 1
        
        # If q is not prime, try a different approach
        if not is_prime(q):
            continue
        
        n_actual = len(elements)
        if n_actual != n:
            # Mismatch, skip
            continue
        
        # Precompute multiplication: mul[i][j] = index of elements[i] * elements[j]
        # This is n^2 * 4 operations, for n=4080 that's ~66M — might be too slow
        # Instead, use on-the-fly multiplication in the BFS
        
        def mat_mul(e1, e2):
            a1, b1, c1, d1 = e1
            a2, b2, c2, d2 = e2
            return ((a1*a2 + b1*c2) % q, (a1*b2 + b1*d2) % q,
                    (c1*a2 + d1*c2) % q, (c1*b2 + d1*d2) % q)
        
        # Find involutory generators: trace = a+d = 0 mod q, A != I, A != -I
        # -I = (q-1, 0, 0, q-1)
        identity = (1, 0, 0, 1)
        neg_identity = ((q-1) % q, 0, 0, (q-1) % q)
        
        involutions = []
        for elem in elements:
            a, b, c, d = elem
            if (a + d) % q == 0 and elem != identity and elem != neg_identity:
                # Check it's actually an involution: A^2 = I
                sq = mat_mul(elem, elem)
                if sq == identity:
                    involutions.append(elem)
        
        if len(involutions) < 3:
            continue
        
        # We need to find 3 involutions that generate the group and give girth >= g
        # Strategy: try random triples of involutions, check girth
        
        def check_girth(gens, g):
            """
            Check girth >= g using non-backtracking BFS from identity.
            State: (current_elem, prev_gen_index) where prev_gen_index is which generator
            was used to arrive (to prevent backtracking), or -1 for start.
            """
            # BFS: (element, prev_gen) -> depth
            # We want to detect if we can return to identity with prev_gen != -1 
            # at depth < g/2 (for even girth) or... 
            # Actually for girth g, we need no cycle of length < g.
            # A cycle of length L corresponds to a non-backtracking walk of length L 
            # that returns to start.
            # Non-backtracking: consecutive generators are different.
            # For g=16, we need no non-backtracking closed walk of length < 16.
            
            half_g = g // 2  # 8
            
            # BFS from (identity, -1)
            # State: (elem_idx, prev_gen_idx)
            # Use dict for visited states
            visited = set()
            # Queue: (elem, prev_gen, depth)
            queue = deque()
            queue.append((identity, -1, 0))
            visited.add((identity, -1))
            
            while queue:
                elem, prev_gen, depth = queue.popleft()
                if depth >= half_g:
                    continue
                for gi, gen in enumerate(gens):
                    if gi == prev_gen:
                        continue  # non-backtracking
                    new_elem = mat_mul(elem, gen)
                    new_state = (new_elem, gi)
                    if new_state in visited:
                        # Found a cycle!
                        # The cycle length is depth + 1 + (distance from identity to new_elem via non-backtracking)
                        # Actually, this means there's a non-backtracking walk from identity 
                        # of length depth+1 that reaches new_elem with prev=gi, and new_elem 
                        # was previously reached with some prev. This indicates a cycle.
                        # The cycle length is at most depth + 1 + (something).
                        # More precisely: we have a non-backtracking walk of length depth+1 
                        # from identity to new_elem (ending with gen gi), and new_elem was 
                        # previously visited. The shortest such gives a cycle.
                        # For girth check, if we can reach (new_elem, gi) at depth+1 and 
                        # it was first reached at depth' < depth+1, then there's a cycle 
                        # of length (depth+1) - depth' + ... 
                        # Actually, the standard approach: if we revisit a state, the 
                        # cycle length is depth+1 - (first_depth_of_state).
                        # But we don't track first depth easily with a set.
                        # Let me use a different approach.
                        return False  # Conservative: any revisit means cycle < g
                    visited.add(new_state)
                    queue.append((new_elem, gi, depth + 1))
            
            return True
        
        # Try random triples
        random.seed(42)
        found = False
        max_trials = 200
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            if len(involutions) < 3:
                break
            idxs = random.sample(range(len(involutions)), 3)
            gens = [involutions[i] for i in idxs]
            
            # Quick check: do they generate the group? (optional, skip for speed)
            # Just check girth
            if check_girth(gens, g):
                # Verify more carefully and build the graph
                found = True
                break
        
        if not found:
            continue
        
        # Build the graph: n vertices, edges (i, i*gen) for each gen
        edges = []
        for i, elem in enumerate(elements):
            for gen in gens:
                new_elem = mat_mul(elem, gen)
                j = elem_index[new_elem]
                if i < j:
                    edges.append((i, j))
        
        return n_actual, edges
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    if k != 3 or g != 16:
        return _luw_construct(k, g)
    
    # Try SL(2,q) Cayley graph construction
    result = _sl2_construct(k, g, deadline)
    
    if result is not None:
        return result
    
    # Fallback to LUW
    return _luw_construct(k, g)
