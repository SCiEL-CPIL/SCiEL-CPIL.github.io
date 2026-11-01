import time
import random
import sys

def construct(k, g):
    # Fallback to the original seed logic if not the specific target case or if search fails
    if k != 4 or g != 10:
        return _seed_construct(k, g)
        
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Define Dihedral Group D20 (Order 40)
    # 0..19: rotations r^i
    # 20..39: reflections s*r^i
    def mul(a, b):
        if a < 20 and b < 20:
            return (a + b) % 20
        elif a < 20 and b >= 20:
            return 20 + (a + (b - 20)) % 20
        elif a >= 20 and b < 20:
            return 20 + (a - 20 + b) % 20
        else:
            return (a - 20 - (b - 20)) % 20

    def inv(a):
        if a < 20:
            return (-a) % 20
        else:
            return a

    # Base graph K4,4
    # Vertices 0..3 (U), 4..7 (V) in base, but we track indices i, j
    # Edges defined by (i, j)
    
    # Generate non-backtracking closed walks of length 4, 6, 8
    # Walk is a sequence of (u_idx, v_idx) pairs representing edges traversed.
    # Direction matters for voltage product.
    # Walk: u_i0 -> v_j0 -> u_i1 -> v_j1 -> ... -> u_i0
    # Edges: (i0, j0) [u->v], (i1, j0) [v->u], (i1, j1) [u->v], (i2, j1) [v->u] ...
    
    walks = []
    
    # Length 4
    for i0 in range(4):
        for j0 in range(4):
            for i1 in range(4):
                if i1 == i0: continue
                for j1 in range(4):
                    if j1 == j0: continue
                    # u_i0 -> v_j0 -> u_i1 -> v_j1 -> u_i0
                    # Check if u_i1 -> v_j1 -> u_i0 is valid (j1 != j0 is already checked)
                    # And u_i0 -> v_j0 -> u_i1 is valid (i1 != i0 is already checked)
                    # This forms a 4-cycle.
                    # Sequence of edges: (i0, j0), (i1, j0), (i1, j1), (i0, j1)
                    # Note: (i1, j0) is traversed v->u, so inv(X[i1, j0])
                    # (i0, j1) is traversed v->u? No, v_j1 -> u_i0. So inv(X[i0, j1])
                    # Wait, the last edge is v_j1 -> u_i0.
                    # So the sequence is:
                    # 1. u_i0 -> v_j0 : X[i0, j0]
                    # 2. v_j0 -> u_i1 : inv(X[i1, j0])
                    # 3. u_i1 -> v_j1 : X[i1, j1]
                    # 4. v_j1 -> u_i0 : inv(X[i0, j1])
                    walks.append([(i0, j0, 1), (i1, j0, -1), (i1, j1, 1), (i0, j1, -1)])
                    
    # Length 6
    for i0 in range(4):
        for j0 in range(4):
            for i1 in range(4):
                if i1 == i0: continue
                for j1 in range(4):
                    if j1 == j0: continue
                    for i2 in range(4):
                        if i2 == i1: continue
                        # if i2 == i0, it's allowed (non-backtracking only forbids immediate return)
                        # But if i2 == i0, the walk is u_i0->v_j0->u_i1->v_j1->u_i0->v_j2->u_i0?
                        # No, length 6 means 3 u's and 3 v's in between?
                        # u_i0 -> v_j0 -> u_i1 -> v_j1 -> u_i2 -> v_j2 -> u_i0
                        # Constraints:
                        # i1 != i0, i2 != i1.
                        # j1 != j0, j2 != j1.
                        # Also j2 != j0? No, non-backtracking doesn't forbid j2=j0 unless it creates a backtrack?
                        # If j2 == j0, step is u_i2 -> v_j0 -> u_i0.
                        # Previous step was u_i1 -> v_j1.
                        # So v_j1 != v_j0 is required (j1 != j0).
                        # So j2 can be j0.
                        for j2 in range(4):
                            if j2 == j1: continue
                            # Walk:
                            # 1. u_i0 -> v_j0 : X[i0, j0]
                            # 2. v_j0 -> u_i1 : inv(X[i1, j0])
                            # 3. u_i1 -> v_j1 : X[i1, j1]
                            # 4. v_j1 -> u_i2 : inv(X[i2, j1])
                            # 5. u_i2 -> v_j2 : X[i2, j2]
                            # 6. v_j2 -> u_i0 : inv(X[i0, j2])
                            walks.append([(i0, j0, 1), (i1, j0, -1), (i1, j1, 1), (i2, j1, -1), (i2, j2, 1), (i0, j2, -1)])
                            
    # Length 8
    for i0 in range(4):
        for j0 in range(4):
            for i1 in range(4):
                if i1 == i0: continue
                for j1 in range(4):
                    if j1 == j0: continue
                    for i2 in range(4):
                        if i2 == i1: continue
                        for j2 in range(4):
                            if j2 == j1: continue
                            for i3 in range(4):
                                if i3 == i2: continue
                                for j3 in range(4):
                                    if j3 == j2: continue
                                    # Walk:
                                    # u_i0 -> v_j0 -> u_i1 -> v_j1 -> u_i2 -> v_j2 -> u_i3 -> v_j3 -> u_i0
                                    # Edges:
                                    # 1. X[i0, j0]
                                    # 2. inv(X[i1, j0])
                                    # 3. X[i1, j1]
                                    # 4. inv(X[i2, j1])
                                    # 5. X[i2, j2]
                                    # 6. inv(X[i3, j2])
                                    # 7. X[i3, j3]
                                    # 8. inv(X[i0, j3])
                                    walks.append([(i0, j0, 1), (i1, j0, -1), (i1, j1, 1), (i2, j1, -1), (i2, j2, 1), (i3, j2, -1), (i3, j3, 1), (i0, j3, -1)])

    def check_valid(X):
        for walk in walks:
            prod = 0
            for i, j, sign in walk:
                val = X[i][j]
                if sign == -1:
                    val = inv(val)
                prod = mul(prod, val)
            if prod == 0: # Identity in D20 is 0
                return False
        return True

    # Search
    best_n = None
    best_edges = None
    
    while time.time() < deadline:
        # Random X
        X = [[random.randint(0, 39) for _ in range(4)] for _ in range(4)]
        if check_valid(X):
            # Build graph
            n_vertices = 320
            # Indices:
            # U: (i, g) -> i * 40 + g
            # V: (j, g) -> 200 + j * 40 + g
            edges = []
            for i in range(4):
                for j in range(4):
                    x_ij = X[i][j]
                    for g in range(40):
                        u_idx = i * 40 + g
                        v_idx = 200 + j * 40 + mul(g, x_ij)
                        edges.append((u_idx, v_idx))
            return n_vertices, edges
            
    # Fallback
    return _seed_construct(k, g)

def _seed_construct(k, g):
    # Original seed code
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
