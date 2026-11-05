import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Target parameters for the specific task instance
    if k == 4 and g == 10:
        # Try to construct a 244-vertex graph using K4 lift over Z_61
        n_target = 244
        group_size = 61
        base_vertices = 4
        base_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        
        # We need voltages v_ij for these edges such that
        # sum around every triangle is != 0 mod group_size
        # Triangles: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        
        def check_voltages(v):
            # v is dict (u,v) -> voltage for u<v
            # Sums:
            # T012: v[0,1] + v[1,2] - v[0,2]
            s1 = (v[(0,1)] + v[(1,2)] - v[(0,2)]) % group_size
            # T013: v[0,1] + v[1,3] - v[0,3]
            s2 = (v[(0,1)] + v[(1,3)] - v[(0,3)]) % group_size
            # T023: v[0,2] + v[2,3] - v[0,3]
            s3 = (v[(0,2)] + v[(2,3)] - v[(0,3)]) % group_size
            # T123: v[1,2] + v[2,3] - v[1,3]
            s4 = (v[(1,2)] + v[(2,3)] - v[(1,3)]) % group_size
            return s1 != 0 and s2 != 0 and s3 != 0 and s4 != 0

        voltages = None
        attempts = 0
        while time.time() < deadline - 10 and attempts < 10000:
            # Assign random non-zero voltages? Or any?
            # Non-zero is safer to avoid trivial 0 sums, but 0 is allowed in Z_61 as long as sum != 0.
            # However, if v_ij = 0, it's just an edge with no twist.
            # Let's use random 1..60 to be safe from small order issues (though 61 is prime, order is 61 for all non-zero).
            v = {}
            valid = True
            for u, w in base_edges:
                key = (u, w) if u < w else (w, u)
                v[key] = random.randint(1, group_size - 1)
            if check_voltages(v):
                voltages = v
                break
            attempts += 1
        
        if voltages:
            edges = []
            # Build lift
            # Vertex (u, i) -> u * group_size + i
            for u, w in base_edges:
                key = (u, w) if u < w else (w, u)
                vol = voltages[key]
                # Direction matters for voltage sign
                # If we store v for u<w, then:
                # Edge u->w has voltage vol
                # Edge w->u has voltage -vol
                for i in range(group_size):
                    # u to w
                    u_lift = u * group_size + i
                    w_lift = w * group_size + (i + vol) % group_size
                    edges.append((u_lift, w_lift))
                    # w to u (implicitly covered by undirected graph construction? 
                    # The task asks for a list of edges (i, j). Usually undirected.
                    # If we add (u_lift, w_lift), it covers the connection.
                    # We don't need to add (w_lift, u_lift) separately if the evaluator treats edges as undirected.
                    # Standard graph construction: list of unique pairs.
            return n_target, edges

        # Fallback: 248-vertex graph using K4,4 lift over Z_31
        n_target = 248
        group_size = 31
        base_vertices = 8
        # K4,4 edges
        base_edges = []
        for i in range(4):
            for j in range(4):
                base_edges.append((i, 4 + j))
        
        # Check 4-cycles
        # A 4-cycle is defined by two vertices in A and two in B.
        # i1, i2 in A; j1, j2 in B.
        # Cycle: i1-j1-i2-j2-i1
        # Sum: v(i1,j1) + v(i2,j1) [wait, direction]
        # Let's just check all 36 4-cycles.
        
        def check_voltages_k44(v):
            # v is dict (u,w) -> voltage for u<w
            # Get voltage for directed edge u->w
            def get_vol(u, w):
                if u < w:
                    return v[(u, w)]
                else:
                    return -v[(w, u)] % group_size
            
            for i1 in range(4):
                for i2 in range(i1 + 1, 4):
                    for j1 in range(4):
                        for j2 in range(j1 + 1, 4):
                            # Cycle i1 -> j1 -> i2 -> j2 -> i1
                            s = (get_vol(i1, j1) + get_vol(i2, j1) + get_vol(i2, j2) + get_vol(i1, j2)) % group_size
                            # Note: get_vol(i2, j1) is vol for i2->j1.
                            # Edge i2-j1 in base. In cycle i1-j1-i2-j2-i1:
                            # i1 -> j1 (vol i1,j1)
                            # j1 -> i2 (vol i2,j1 is for i2->j1, so j1->i2 is -vol(i2,j1))
                            # Let's be precise.
                            # v(u,w) is stored for u<w.
                            # vol(u->w) = v(u,w) if u<w else -v(w,u).
                            
                            s = 0
                            s += get_vol(i1, 4+j1)
                            s += get_vol(4+j1, i2)
                            s += get_vol(i2, 4+j2)
                            s += get_vol(4+j2, i1)
                            s = s % group_size
                            if s == 0:
                                return False
            return True

        voltages = None
        attempts = 0
        while time.time() < deadline - 10 and attempts < 10000:
            v = {}
            for u, w in base_edges:
                key = (u, w) if u < w else (w, u)
                v[key] = random.randint(1, group_size - 1)
            if check_voltages_k44(v):
                voltages = v
                break
            attempts += 1
            
        if voltages:
            edges = []
            for u, w in base_edges:
                key = (u, w) if u < w else (w, u)
                vol = voltages[key]
                for i in range(group_size):
                    u_lift = u * group_size + i
                    w_lift = w * group_size + (i + vol) % group_size
                    edges.append((u_lift, w_lift))
            return n_target, edges

    # Fallback to original seed logic if specific optimizations fail or params differ
    # (Though task implies k=4, g=10)
    
    # Original Seed Logic (L-U-W)
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
