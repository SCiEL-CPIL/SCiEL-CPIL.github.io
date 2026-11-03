import time
import random
from collections import deque
from math import gcd

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, systematically searches cyclic and bicyclic constructions.
    """
    if k != 3:
        return _luw_fallback(k, g)

    start_time = time.time()
    deadline = start_time + 110.0

    def check_girth_circulant(N, a, b, target_girth):
        """
        BFS on (position, last_edge_type) state space to find shortest non-backtracking cycle.
        Edge types: 0=+a, 1=-a, 2=+b, 3=-b. Inverse: 0<->1, 2<->3.
        Returns True if girth >= target_girth.
        """
        a_mod = a % N
        b_mod = b % N
        gens = (a_mod, -a_mod, b_mod, -b_mod)
        inv = (1, 0, 3, 2)
        
        # dist[pos][edge_type] = shortest non-backtracking walk length to reach pos via edge_type
        # -1 means unvisited
        dist = [[-1, -1, -1, -1] for _ in range(N)]
        
        queue = deque()
        # Start: from vertex 0, take each of the 3 edges (we have 4 directions but +a/-a are same edge)
        # Actually for non-backtracking: from 0 we can go via any of the 4 directed edges
        # But +a and -a lead to different vertices (unless a = N-a, which we exclude)
        for i in range(4):
            new_s = gens[i] % N
            dist[new_s][i] = 1
            queue.append((new_s, i, 1))
        
        while queue:
            s, last_gen, length = queue.popleft()
            
            if length >= target_girth:
                continue
            
            for i in range(4):
                if last_gen == inv[i]:
                    continue
                
                new_s = (s + gens[i]) % N
                new_len = length + 1
                
                # If we return to 0, we have a cycle
                if new_s == 0:
                    if new_len < target_girth:
                        return False
                    else:
                        return True
                
                if dist[new_s][i] == -1:
                    dist[new_s][i] = new_len
                    queue.append((new_s, i, new_len))
        
        return True

    def generate_cyclic_graph(N, a, b):
        """Generate edges for Z_N with generators {a, -a, b, -b} (3-regular)."""
        a = a % N
        b = b % N
        
        if a == 0 or b == 0:
            return None
        if 2 * a % N == 0 or 2 * b % N == 0:
            return None
        if a == b or a == (N - b) % N or (N - a) % N == b or (N - a) % N == (N - b) % N:
            return None
        if gcd(gcd(a, b), N) != 1:
            return None

        edges = set()
        for i in range(N):
            for j in [(i + a) % N, (i - a) % N, (i + b) % N]:
                u, v = min(i, j), max(i, j)
                edges.add((u, v))
        return N, list(edges)

    def check_girth_bicyclic(N, a, b, target_girth):
        """
        Bicyclic graph: vertices are (i, s) where i in Z_N, s in {0,1}.
        Edges: (i,s)-(i+a,s) via 'a-edges', (i,s)-(i+b,1-s) via 'b-edges' (crossing layers).
        This gives degree 3: two a-neighbors in same layer, one b-neighbor in other layer.
        """
        a_mod = a % N
        b_mod = b % N
        
        # State: (position, layer, last_edge_type)
        # Edge types: 0=+a same layer, 1=-a same layer, 2=+b cross layer
        # Inverses: 0<->1, 2<->2 (b-edge from layer s goes to layer 1-s, and back is same type)
        
        # Actually let's think more carefully.
        # From (i, s):
        #   -a edge: goes to (i-a, s)  [type 1]
        #   +a edge: goes to (i+a, s)  [type 0]
        #   +b edge: goes to (i+b, 1-s) [type 2]
        # Non-backtracking: can't use type 0 after type 1, type 1 after type 0, type 2 after type 2.
        
        # State space: N * 2 * 3 = 6N states
        # dist[pos][layer][edge_type]
        dist = [[[-1, -1, -1] for _ in range(2)] for _ in range(N)]
        
        queue = deque()
        # Start from (0, 0), take each edge
        for i in range(3):
            if i == 0:
                new_s, new_layer = a_mod, 0
            elif i == 1:
                new_s, new_layer = (-a_mod) % N, 0
            else:
                new_s, new_layer = b_mod, 1
            dist[new_s][new_layer][i] = 1
            queue.append((new_s, new_layer, i, 1))
        
        inv = (1, 0, 2)  # inverse of edge type
        
        while queue:
            s, layer, last_gen, length = queue.popleft()
            
            if length >= target_girth:
                continue
            
            for i in range(3):
                if last_gen == inv[i]:
                    continue
                
                if i == 0:
                    new_s = (s + a_mod) % N
                    new_layer = layer
                elif i == 1:
                    new_s = (s - a_mod) % N
                    new_layer = layer
                else:
                    new_s = (s + b_mod) % N
                    new_layer = 1 - layer
                
                new_len = length + 1
                
                if new_s == 0 and new_layer == 0:
                    if new_len < target_girth:
                        return False
                    else:
                        return True
                
                if dist[new_s][new_layer][i] == -1:
                    dist[new_s][new_layer][i] = new_len
                    queue.append((new_s, new_layer, i, new_len))
        
        return True

    def generate_bicyclic_graph(N, a, b):
        """Generate edges for bicyclic graph with 2N vertices."""
        a = a % N
        b = b % N
        
        if a == 0 or b == 0:
            return None
        if 2 * a % N == 0:
            return None
        
        # Check connectivity: the graph on 2N vertices should be connected
        # gcd(a, b, N) should be 1, and also need to check layer connectivity
        if gcd(gcd(a, b), N) != 1:
            return None
        
        edges = set()
        for i in range(N):
            for layer in range(2):
                v = layer * N + i
                # a-edges within same layer
                for j in [(i + a) % N, (i - a) % N]:
                    u = layer * N + j
                    edges.add((min(v, u), max(v, u)))
                # b-edge to other layer
                j = (i + b) % N
                u = (1 - layer) * N + j
                edges.add((min(v, u), max(v, u)))
        
        # Verify it's 3-regular
        if len(edges) != 3 * N:
            return None
        
        return 2 * N, list(edges)

    # Moore bound for k=3, g=16: 2 + 3*(1+2+4+8+16+32+64) = 2 + 3*127 = 383
    # But for girth 16 (even), Moore bound is 2*(2^(g/2-1) - 1) = 2*(128-1) = 254
    # Actually Moore bound for k-regular girth g:
    #   if g even: n >= 2*(k-1)^(g/2-1) + 2*(k-1)^(g/2-1) ... 
    #   Standard: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * ((k-1)^(g/2) - 1)/(k-2)
    #   For k=3, g=16: 2 * (2^8 - 1) = 2 * 255 = 510
    # So minimum is 512 (even) or 511.
    
    moore_bound = 2 * ((1 << (g // 2)) - 1)
    if moore_bound % 2 != 0:
        moore_bound += 1
    
    max_N_circ = 1024
    max_N_bicyclic = 512  # bicyclic has 2N vertices, so N up to 512 gives 1024 vertices
    
    # Phase 1: Systematic search over cyclic graphs
    for N in range(moore_bound, max_N_circ + 1):
        if time.time() > deadline - 10:
            break
        
        # For each N, enumerate valid (a, b) pairs
        # a in [1, N//2], b in [a+1, N//2] to avoid duplicates
        # But we need to be careful about the modular constraints
        
        found = False
        for a in range(1, N // 2 + 1):
            if time.time() > deadline - 10:
                break
            if 2 * a % N == 0:
                continue
            for b in range(a + 1, N // 2 + 1):
                if time.time() > deadline - 10:
                    break
                if 2 * b % N == 0:
                    continue
                if a == (N - b) % N or (N - a) % N == b:
                    continue
                if gcd(gcd(a, b), N) != 1:
                    continue
                
                # Fast pre-check: the sum of any two generators (with signs) 
                # shouldn't create short relations
                # A necessary condition: no non-trivial relation of length < g
                # Simple check: a+b, a-b, a+b+N, etc. shouldn't be 0 mod N for small combos
                # Actually the BFS will handle this, but let's add a quick check
                # The shortest possible cycle involves at most g/2 edges from each side
                # Quick necessary condition: 2a != 0, 2b != 0, a+b != 0, a-b != 0 mod N
                # These are already checked above.
                
                if check_girth_circulant(N, a, b, g):
                    res = generate_cyclic_graph(N, a, b)
                    if res is not None:
                        return res
        
        # Also try a in [1, N//2], b in [1, N//2] with b < a (we already did b > a)
        # Actually we covered all pairs with a < b < N/2, which covers all distinct pairs
        # up to sign. But we might miss cases where a > N/2.
        # Let's also try a in [N//2+1, N-1] but that's equivalent to -a in [1, N/2-1]
        # So we've covered all cases.

    # Phase 2: Bicyclic search
    for N in range(moore_bound // 2, max_N_bicyclic + 1):
        if time.time() > deadline - 5:
            break
        
        for a in range(1, N // 2 + 1):
            if time.time() > deadline - 5:
                break
            if 2 * a % N == 0:
                continue
            for b in range(1, N // 2 + 1):
                if time.time() > deadline - 5:
                    break
                if b == 0:
                    continue
                if gcd(gcd(a, b), N) != 1:
                    continue
                
                if check_girth_bicyclic(N, a, b, g):
                    res = generate_bicyclic_graph(N, a, b)
                    if res is not None:
                        return res

    # Fallback to LUW
    return _luw_fallback(k, g)

def _luw_fallback(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction fallback.
    """
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
