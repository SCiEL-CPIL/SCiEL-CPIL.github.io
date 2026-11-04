# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # --- LUW Construction ---
    def luw_construct(k, g):
        q = k
        if g <= 5:
            d = 3
        else:
            d = 3
            while True:
                girth_est = d + 5 if d % 2 == 1 else d + 4
                if girth_est >= g:
                    break
                d += 1
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        for c in range(2, d + 1):
            rule_a[c], rule_b[c] = rule[c]

        mul_table = [[mul(i, j) for j in range(q)] for i in range(q)]
        add_table = [[add(i, j) for j in range(q)] for i in range(q)]
        sub_table = [[sub(i, j) for j in range(q)] for i in range(q)]

        def compute_neighbors_point(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                result.append((1, tuple(y)))
            return result

        def compute_neighbors_line(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                result.append((0, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if time.time() > deadline - 10: # Reserve time for LS
                break
            side, x = vertex
            if side == 0:
                neighbors = compute_neighbors_point(x)
            else:
                neighbors = compute_neighbors_line(x)
            for other in neighbors:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    n_luw, edges_luw = luw_construct(k, g)
    
    # --- Local Search Attempt ---
    # Try to find a smaller graph
    # Moore Bound
    if g % 2 == 0:
        m = g // 2
        moore = k * sum((k-1)**i for i in range(m))
    else:
        m = (g - 1) // 2
        moore = 2 * (k-1) * sum((k-1)**i for i in range(m-1)) # Approx, actually (k-1)^m + ...
        # Correct formula for odd g: n >= 2 * (k-1) * sum_{i=0}^{g/2-1} ... 
        # Actually, for odd g, Moore bound is (k-1)^{(g-1)/2} + ... 
        # Let's just use a safe lower bound: k * sum((k-1)^i for i in range((g-1)//2))
        moore = k * sum((k-1)**i for i in range((g-1)//2))
        
    # Only attempt LS if n_luw is significantly larger than moore
    if n_luw > moore * 1.2:
        ls_result = local_search_construct(k, g, moore, n_luw, deadline - time.time() - 5)
        if ls_result:
            n_ls, edges_ls = ls_result
            if n_ls < n_luw:
                return n_ls, edges_ls
                
    return n_luw, edges_luw

def local_search_construct(k, g, n_min, n_max, time_budget):
    # Simplified LS: just try a few random graphs
    # This is a stub for now, to be filled if time permits
    # For safety, return None
    return None
# EVOLVE-BLOCK-END
