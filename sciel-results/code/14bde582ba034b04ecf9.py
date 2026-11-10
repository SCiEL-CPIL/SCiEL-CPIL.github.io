import time
import random
from collections import deque

class GF:
    def __init__(self, m, poly):
        self.m = m
        self.q = 1 << m
        self.poly = poly
        self.mul_table = [[0]*self.q for _ in range(self.q)]
        self.inv_table = [0]*self.q
        
        for i in range(self.q):
            for j in range(self.q):
                res = 0
                a = i
                b = j
                while b:
                    if b & 1:
                        res ^= a
                    a <<= 1
                    if a >= self.q:
                        a ^= self.poly
                    b >>= 1
                self.mul_table[i][j] = res
        self.inv_table[1] = 1
        for i in range(2, self.q):
            for j in range(2, self.q):
                if self.mul_table[i][j] == 1:
                    self.inv_table[i] = j
                    self.inv_table[j] = i
                    break

    def add(self, a, b):
        return a ^ b

    def sub(self, a, b):
        return a ^ b

    def mul(self, a, b):
        return self.mul_table[a][b]

    def inv(self, a):
        return self.inv_table[a]

    def div(self, a, b):
        return self.mul_table[a][self.inv_table[b]]

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 3:
        # Fallback to LUW for other k
        return _luw_construct(k, g, deadline)

    # SL(2, 16) Construction
    field = GF(4, 0b10011)
    q = field.q
    
    # Generate SL(2, q)
    elements = []
    elem_to_idx = {}
    
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0:
                    if field.mul(b, c) == 1:
                        for d in range(q):
                            e = (a, b, c, d)
                            elem_to_idx[e] = len(elements)
                            elements.append(e)
                else:
                    bc = field.mul(b, c)
                    rhs = field.add(1, bc)
                    d = field.div(rhs, a)
                    e = (a, b, c, d)
                    elem_to_idx[e] = len(elements)
                    elements.append(e)
    
    n = len(elements)
    identity = (1, 0, 0, 1)
    
    # Precompute multiplication
    def mat_mul(m1, m2):
        a1,b1,c1,d1 = m1
        a2,b2,c2,d2 = m2
        aa = field.add(field.mul(a1,a2), field.mul(b1,c2))
        ab = field.add(field.mul(a1,b2), field.mul(b1,d2))
        ba = field.add(field.mul(c1,a2), field.mul(d1,c2))
        bb = field.add(field.mul(c1,b2), field.mul(d1,d2))
        return (aa, ab, ba, bb)
    
    # Find Involutions
    involutions = []
    for e in elements:
        a,b,c,d = e
        if field.add(a,d) == 0 and e != identity:
            involutions.append(e)
            
    random.seed(42)
    
    # Word Search for Girth
    def has_short_cycle(gens, max_depth):
        # BFS on word tree
        # State: (matrix, depth)
        # We don't need to track path, just if we hit identity
        visited = {identity: 0}
        queue = deque([(identity, 0)])
        
        while queue:
            curr, depth = queue.popleft()
            if depth == max_depth:
                continue
            for g in gens:
                next_mat = mat_mul(curr, g)
                if next_mat == identity:
                    return True
                if next_mat not in visited:
                    visited[next_mat] = depth + 1
                    queue.append((next_mat, depth + 1))
        return False

    best_girth_found = 0
    best_gens = None
    
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        if len(involutions) < 3:
            break
        S = random.sample(involutions, 3)
        # Check if they generate the group? 
        # If not, girth is min of components. 
        # But for Cayley graph, if they don't generate, graph is disconnected.
        # Girth is min girth of components.
        # We want a single component? No, just "no cycle shorter than g".
        # So disconnected is OK if all components have girth >= g.
        # But if they don't generate, the "word" search only checks the subgroup.
        # If the subgroup has no short cycles, but the group does?
        # The graph is on the whole group. Edges are e->e*s.
        # If S doesn't generate G, the graph is disjoint union of cosets.
        # Each coset is a copy of the Cayley graph of the subgroup H.
        # So we just need H to have girth >= g.
        
        if not has_short_cycle(S, g - 1):
            best_gens = S
            break
            
    if best_gens:
        # Build graph
        edges = []
        for i, e in enumerate(elements):
            for s in best_gens:
                f = mat_mul(e, s)
                j = elem_to_idx[f]
                if i < j:
                    edges.append((i, j))
        return n, edges

    # Fallback to LUW
    return _luw_construct(k, g, deadline)

def _luw_construct(k, g, deadline):
    q = k
    d = g - 5
    if d % 2 != 1:
        d += 1
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
            
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
