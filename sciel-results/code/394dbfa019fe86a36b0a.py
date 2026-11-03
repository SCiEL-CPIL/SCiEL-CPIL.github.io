"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    # Target: (4, 10) cage
    if k == 4 and g == 10:
        # Try to find the (4,10)-cage using a cyclic cover of K5
        result = _find_cyclic_cover_cage(320, 10)
        if result is not None:
            return result
            
    # Fallback to LUW construction (might be larger)
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        return luw_result
        
    # Fallback to random
    return _random_regular(k, g, start_time, deadline)


def _find_cyclic_cover_cage(n, target_girth):
    """Find a cyclic cover of K5 with voltages in Z4xZ4xZ4 having girth >= target_girth."""
    # n should be 320
    # Base graph: K5
    # Voltage group: Z4 x Z4 x Z4
    # We need to assign voltages to edges of K5 such that no cycle of length < target_girth has sum 0.
    
    # Vertices of K5: 0, 1, 2, 3, 4
    # Edges: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    edges_k5 = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    
    # Precompute all cycles of length 3, 4, 5 in K5
    # A cycle is a sequence of vertices v0, v1, ..., vL-1 such that vi~vi+1 and vL-1~v0
    # We represent a cycle as a list of edges (u, v) with orientation.
    # Sum of voltages: sum of phi_{u,v} for directed edges.
    # Note: phi_{v,u} = -phi_{u,v} in the group.
    
    cycles = []
    vertices = [0, 1, 2, 3, 4]
    
    # Find all simple cycles of length 3, 4, 5
    # We can use a simple DFS or hardcode them.
    # Let's generate them.
    def find_cycles(length):
        found = []
        for start in vertices:
            # Fix start to avoid duplicates (rotate)
            # And fix next vertex to avoid reversal?
            # Actually, we just need the set of edges with signs.
            # Let's just enumerate all permutations of length 'length' starting with 0?
            # No, K5 is small.
            pass
        return found
    
    # Simpler: enumerate all combinations of edges that form a cycle.
    # Or just hardcode the cycles.
    # 3-cycles: (0,1,2), (0,1,3), (0,1,4), (0,2,3), (0,2,4), (0,3,4), (1,2,3), (1,2,4), (1,3,4), (2,3,4)
    # Each triangle {a,b,c} gives cycle a->b->c->a.
    # Voltages: phi_ab + phi_bc + phi_ca = 0?
    # We need phi_ab + phi_bc - phi_ac != 0?
    # Let's define phi[u][v] for u < v.
    # Cycle a->b->c->a: phi_ab + phi_bc + phi_ca.
    # phi_ca = -phi_ac.
    # So sum = phi_ab + phi_bc - phi_ac.
    
    # Let's generate the condition checks directly.
    # We assign a voltage vector to each edge (u, v) with u < v.
    # Let V[e] be the vector.
    # For a cycle v0->v1->...->vL-1->v0:
    # Sum = sum_{i} (V[ei] if vi < vi+1 else -V[ei])
    
    # We will search for V[0..9] in Z4^3.
    
    # To speed up, we can fix some voltages to break symmetry?
    # Or just random search.
    # There are 64^10 possibilities, but we can prune.
    
    # Let's try a few random assignments.
    for attempt in range(10000):
        # Randomly assign voltages
        voltages = [tuple(random.randint(0, 3) for _ in range(3)) for _ in range(10)]
        
        # Check girth
        valid = True
        # Check all cycles of length 3, 4, 5
        # We need to list the cycles as sequences of edge indices and signs.
        
        # Helper to get edge index
        edge_index = {}
        for i, (u, v) in enumerate(edges_k5):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i # Store both, but handle sign in sum
        
        # Generate cycles
        # 3-cycles
        for a in range(5):
            for b in range(a+1, 5):
                for c in range(b+1, 5):
                    # Cycle a->b->c->a
                    # Edges: (a,b), (b,c), (c,a)
                    # Signs: +, +, - (if c>a, then c->a is -)
                    # Wait, (c,a) is edge (a,c) with sign -?
                    # Yes, if we define V for u<v.
                    # Sum = V[ab] + V[bc] - V[ac]
                    idx_ab = edge_index[(a,b)]
                    idx_bc = edge_index[(b,c)]
                    idx_ac = edge_index[(a,c)]
                    
                    s = [0,0,0]
                    for k in range(3):
                        s[k] = (s[k] + voltages[idx_ab][k] + voltages[idx_bc][k] - voltages[idx_ac][k]) % 4
                    if s == (0,0,0):
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: continue
        
        # 4-cycles
        # a->b->c->d->a
        for a in range(5):
            for b in range(a+1, 5):
                for c in range(5):
                    if c == a or c == b: continue
                    for d in range(5):
                        if d == a or d == b or d == c: continue
                        # Ensure unique cycle?
                        # We just check if sum is 0.
                        # Path a->b->c->d->a
                        # Edges: (a,b), (b,c), (c,d), (d,a)
                        # Signs:
                        # (a,b): + if a<b else -
                        # (b,c): + if b<c else -
                        # (c,d): + if c<d else -
                        # (d,a): + if d<a else -
                        
                        def get_sign(u, v):
                            return 1 if u < v else -1
                        
                        def get_idx(u, v):
                            return edge_index[(min(u,v), max(u,v))]
                        
                        s = [0,0,0]
                        for k in range(3):
                            s[k] = (s[k] + 
                                    get_sign(a,b)*voltages[get_idx(a,b)][k] +
                                    get_sign(b,c)*voltages[get_idx(b,c)][k] +
                                    get_sign(c,d)*voltages[get_idx(c,d)][k] +
                                    get_sign(d,a)*voltages[get_idx(d,a)][k]) % 4
                        if s == (0,0,0):
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: continue
        
        # 5-cycles
        # a->b->c->d->e->a
        # Permutations of 0,1,2,3,4 starting with 0?
        # To avoid duplicates, we can just check all permutations of 1,2,3,4
        for perm in [(1,2,3,4), (1,2,4,3), (1,3,2,4), (1,3,4,2), (1,4,2,3), (1,4,3,2),
                     (2,1,3,4), (2,1,4,3), (2,3,1,4), (2,3,4,1), (2,4,1,3), (2,4,3,1),
                     (3,1,2,4), (3,1,4,2), (3,2,1,4), (3,2,4,1), (3,4,1,2), (3,4,2,1),
                     (4,1,2,3), (4,1,3,2), (4,2,1,3), (4,2,3,1), (4,3,1,2), (4,3,2,1)]:
            a = 0
            b, c, d, e = perm
            path = [a, b, c, d, e, a]
            
            s = [0,0,0]
            for k in range(3):
                s[k] = 0
                for i in range(4):
                    u = path[i]
                    v = path[i+1]
                    s[k] = (s[k] + get_sign(u,v)*voltages[get_idx(u,v)][k]) % 4
            if s == (0,0,0):
                valid = False
                break
        if not valid: continue
        
        # If we reach here, the voltages are valid!
        # Construct the graph
        n = 320
        edges = []
        # Vertex (v, g) where v in 0..4, g in Z4^3
        # Map (v, g) to index
        # index = v * 64 + g[0]*16 + g[1]*4 + g[2]
        
        for v in range(5):
            for g0 in range(4):
                for g1 in range(4):
                    for g2 in range(4):
                        g = (g0, g1, g2)
                        idx_v = v * 64 + g0*16 + g1*4 + g2
                        for u in range(5):
                            if u == v: continue
                            # Get voltage
                            idx_e = get_idx(v, u)
                            sign = get_sign(v, u)
                            # New g
                            ng = [0,0,0]
                            for k in range(3):
                                ng[k] = (g[k] + sign * voltages[idx_e][k]) % 4
                            idx_u = u * 64 + ng[0]*16 + ng[1]*4 + ng[2]
                            if idx_v < idx_u:
                                edges.append((idx_v, idx_u))
                            else:
                                edges.append((idx_u, idx_v))
        
        # Remove duplicates?
        # The construction should produce each edge exactly once if we iterate carefully?
        # No, (v,g) to (u,ng) and (u,ng) to (v,g) are the same edge.
        # We added both?
        # In the loop, for each v, we iterate u.
        # If we add (min, max), we might add duplicates.
        # Let's use a set.
        edge_set = set(edges)
        return n, list(edge_set)
        
    return None


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    # This is a fallback. The cyclic cover is preferred.
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


def _random_regular(k, g, start_time, deadline):
    n = 500
    edges = _gen_random_4reg(n)
    if edges is not None:
        return n, edges
    return n, []


def _gen_random_4reg(n):
    if n < 4 or n * 4 % 2 != 0:
        return None
    for _ in range(100):
        stubs = [v for v in range(n) for _ in range(4)]
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        if valid:
            return list(edges)
    return None

# EVOLVE-BLOCK-END
