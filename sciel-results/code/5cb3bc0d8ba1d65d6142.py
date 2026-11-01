"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    if q == 4:
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
        is_gf4 = True
    else:
        mul_table = None
        is_gf4 = False

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rule_a = np.array([rule[c][0] for c in range(2, d + 1)], dtype=np.int64)
    rule_b = np.array([rule[c][1] for c in range(2, d + 1)], dtype=np.int64)
    d_range = np.arange(2, d + 1)
    num_steps = d - 1

    free_vals = np.arange(q, dtype=np.int64)

    def get_neighbours(side, x_arr):
        y_coords = np.empty((q, d + 1), dtype=np.int64)
        y_coords[:, 0] = 0
        y_coords[:, 1] = free_vals
        if side == 0:
            for idx in range(num_steps):
                c = 2 + idx
                a = rule_a[idx]
                b = rule_b[idx]
                if is_gf4:
                    prod = mul_table[y_coords[:, a], x_arr[b]]
                else:
                    prod = (y_coords[:, a] * x_arr[b]) % q
                y_coords[:, c] = (x_arr[c] + prod) % q
        else:
            for idx in range(num_steps):
                c = 2 + idx
                a = rule_a[idx]
                b = rule_b[idx]
                if is_gf4:
                    prod = mul_table[x_arr[a], y_coords[:, b]]
                else:
                    prod = (x_arr[a] * y_coords[:, b]) % q
                y_coords[:, c] = (x_arr[c] - prod) % q
        return y_coords

    start_key = (0, (0,) * (d + 1))
    index = {start_key: 0}
    order = [start_key]
    edges = []

    for i in range(len(order)):
        if time.time() > deadline - 5.0:
            break
        vertex = order[i]
        side, x_tuple = vertex
        x_arr = np.array(x_tuple, dtype=np.int64)
        y_coords = get_neighbours(side, x_arr)
        for j in range(q):
            other_key = (1 - side, tuple(y_coords[j]))
            if other_key not in index:
                index[other_key] = len(order)
                order.append(other_key)
            if side == 0:
                edges.append((index[vertex], index[other_key]))

    return len(order), edges
# EVOLVE-BLOCK-END
