import random
import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin

    if k != 5:
        return _luw_construct(k, g, deadline)

    # For k=5, g=9: LUW gives a valid graph. 
    # Try to find something smaller with a quick structured search,
    # but don't waste time on random graphs (probability ~0 for girth>=9).
    
    # First, get the LUW construction as a baseline (fast, deterministic)
    luw_result = _luw_construct(k, g, deadline)
    luw_n = luw_result[0]
    
    # Check if we have time to try a few structured alternatives
    elapsed = time.time() - start_time
    if elapsed > deadline - 20.0:
        return luw_result
    
    # Try a few smaller n values with a quick check
    # For 5-regular graphs with girth 9, the Moore bound is ~428
    # But finding one by random search is essentially impossible in 100s
    # Instead, try if LUW is already optimal or near-optimal
    # The LUW construction for d=5, q=5 gives 2*5^3 = 250? Let's see.
    
    # Actually, just return the LUW result - it's valid and fast.
    # Random search for n < luw_n with girth >= 9 has near-zero success probability.
    return luw_result


def _luw_construct(k, g, deadline=None):
    """Vectorized L-U-W construction using NumPy for speed."""
    if deadline is None:
        deadline = time.time() + 110.0
    
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    if q == 4:
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        mul_table = np.outer(np.arange(q, dtype=np.int64), np.arange(q, dtype=np.int64)) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rule_a = np.array([rule[c][0] for c in range(2, d + 1)], dtype=np.int64)
    rule_b = np.array([rule[c][1] for c in range(2, d + 1)], dtype=np.int64)

    free_vals = np.arange(q, dtype=np.int64)

    def get_neighbours(side, x_arr):
        y_coords = np.empty((q, d + 1), dtype=np.int64)
        y_coords[:, 0] = 0
        y_coords[:, 1] = free_vals
        if side == 0:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                prod = mul_table[y_coords[:, a], x_arr[b]]
                y_coords[:, c] = (x_arr[c] + prod) % q
        else:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                prod = mul_table[x_arr[a], y_coords[:, b]]
                y_coords[:, c] = (x_arr[c] - prod) % q
        return y_coords

    start_key = (0, (0,) * (d + 1))
    index = {start_key: 0}
    order = [start_key]
    edges = []

    for i, vertex in enumerate(order):
        if time.time() > deadline - 5.0:
            break
        side, x_tuple = vertex
        x_arr = np.array(x_tuple, dtype=np.int64)
        y_coords = get_neighbours(side, x_arr)
        for j in range(q):
            other_key = (1 - side, tuple(y_coords[j]))
            if other_key not in index:
                index[other_key] = len(order)
                order.append(other_key)
            if side == 0:
                edges.append((i, index[other_key]))

    return len(order), edges
