"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _smallest_prime_power(k):
    """Find the smallest prime power q >= k."""
    if k <= 1:
        return 2
    n = k
    while n < 100:
        is_pp = False
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                m = 0
                temp = n
                while temp % p == 0:
                    temp //= p
                    m += 1
                if temp == 1 and m > 1:
                    is_pp = True
                    break
        if is_pp or _is_prime(n):
            return n
        n += 1
    return n

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar
    construction CD(d, q) with q = smallest prime power >= k.
    """
    deadline = time.time() + 145

    q = _smallest_prime_power(k)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Determine d such that girth >= d + 5 >= g, with d odd
    if (g - 5) % 2 == 1:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)

    # Precompute the rule as a list of (a, b) pairs for faster access
    rule = [(None, None)] * (d + 1)
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

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []

    queue = deque([0])

    while queue:
        if time.time() > deadline:
            break
        i = queue.popleft()
        side, x = order[i]

        for first in range(q):
            y_list = [0] * (d + 1)
            y_list[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y_list[c] = (x[c] + y_list[a] * x[b]) % q
                else:
                    y_list[c] = (x[c] - x[a] * y_list[b]) % q
            other = (1 - side, tuple(y_list))

            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)

            if side == 0:
                edges.append((i, index[other]))

    return len(order), edges
# EVOLVE-BLOCK-END
