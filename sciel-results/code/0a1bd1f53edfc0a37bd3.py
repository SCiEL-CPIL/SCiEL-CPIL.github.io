"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 115  # safety margin before 120s kill
    q = k
    
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    if q == 4:
        add_table = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        sub_table = add_table
    else:
        add_table = [[(i+j) % q for j in range(q)] for i in range(q)]
        mul_table = [[(i*j) % q for j in range(q)] for i in range(q)]
        sub_table = [[(i-j) % q for j in range(q)] for i in range(q)]
    
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        pairs = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule_a[c], rule_b[c] = pairs[r]
    
    # Precompute neighbor offsets for each "type" of vertex
    # A vertex is determined by (side, x[1], x[2], ..., x[d])
    # For each such vertex and each 'first' value, compute the neighbor's encoded value
    
    # Total possible vertices: 2 * q^d (since x[1..d] each have q values)
    # Precompute: for each x_tuple (length d) and each first, what is the neighbor offset
    
    # Encode: vertex_id = side * q^d + x[1]*q^(d-1) + x[2]*q^(d-2) + ... + x[d]*q^0
    q_pow = [1] * (d + 1)
    for i in range(1, d + 1):
        q_pow[i] = q_pow[i-1] * q
    base = q_pow[d]  # q^d
    
    # Precompute all transitions
    # trans[side][x_idx][first] = neighbor_x_idx (in the other side's space)
    # where x_idx = x[1]*q^(d-1) + ... + x[d]
    
    trans = [[None] * base for _ in range(2)]
    
    for side in range(2):
        for x_idx in range(base):
            # Decode x_idx to get x[1..d]
            x = [0] * (d + 1)
            rem = x_idx
            for i in range(d, 0, -1):
                x[i] = rem % q
                rem //= q
            
            trans[side][x_idx] = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    if side == 0:
                        y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                    else:
                        y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                
                # Encode y[1..d] to y_idx
                y_idx = 0
                for i in range(1, d + 1):
                    y_idx = y_idx * q + y[i]
                trans[side][x_idx].append(y_idx)
    
    # Now BFS using integer vertex IDs
    # vertex_id = side * base + x_idx
    start_id = 0  # side=0, x_idx=0
    
    index = {start_id: 0}
    order = [start_id]
    edges = []
    
    queue = deque([0])
    
    while queue:
        if time.time() > deadline - 3:
            break
        
        vi = queue.popleft()
        v = order[vi]
        side = v // base
        x_idx = v % base
        
        neighbors_idx = trans[side][x_idx]
        
        for first in range(q):
            y_idx = neighbors_idx[first]
            other = (1 - side) * base + y_idx
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            
            if side == 0:
                edges.append((vi, index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
