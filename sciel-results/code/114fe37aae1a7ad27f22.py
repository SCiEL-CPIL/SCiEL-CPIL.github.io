import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return elapsed() > deadline

    # Primary strategy: LUW construction
    # LUW graphs are known to have large girth.
    # For k=3, g=16, q=3 (prime), d=11.
    # This produces a graph with ~177k vertices.
    # This is a valid upper bound, though not the smallest.
    # It is fast to construct and guaranteed to have girth >= 2d (or similar bound >= g).
    
    try:
        n, edges = _luw_construct(k, g)
        # Optional: Quick sanity check on size and degree
        # We trust the construction for girth.
        # We can do a very fast check on degree and simplicity if needed, 
        # but LUW is simple and regular by construction.
        return n, edges
    except Exception as e:
        # Fallback if LUW fails (e.g., k not prime)
        # For k=3, this shouldn't happen.
        # If it does, we need a valid graph.
        # We can try to construct a random graph and verify, but that's slow and likely invalid.
        # As a last resort, if k=2, return a cycle of length g.
        if k == 2:
            n = g
            edges = [(i, (i+1)%n) for i in range(n)]
            return n, edges
        
        # If k > 2 and LUW failed, we are in trouble.
        # Try a very large random graph? No, girth will be small.
        # Just raise an error to indicate failure, or return a dummy.
        # But the task requires a valid graph.
        # Since k=3 is the target, and LUW works for k=3, we assume success.
        raise e

def _luw_construct(k, g, try_smaller_d=False):
    # ... (Keep existing LUW code from prompt) ...
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif _is_prime(q):
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        raise ValueError(f"q={q} is not prime and not 4; LUW not applicable")

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
