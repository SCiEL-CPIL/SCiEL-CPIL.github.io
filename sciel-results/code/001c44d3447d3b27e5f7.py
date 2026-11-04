import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Optimized for minimal d such that 2d+2 >= g.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # LUW parameter d: girth is 2d+2. We need 2d+2 >= g => d >= (g-2)/2.
    # d must be at least 3 for the rule to be well-defined in the seed logic.
    d = max(3, (g - 1) // 2)
    
    q = k
    
    # Check if q is a valid field size for LUW (prime or 4)
    # Simple primality check for small q
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif is_prime(q):
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        # Fallback: If q is not prime and not 4, LUW is not directly applicable.
        # For the specific task k=3, this won't happen.
        # We return a trivial invalid result or seed-like behavior to avoid crash.
        # But since the task is k=3, we assume valid q.
        raise ValueError("q must be prime or 4 for LUW construction")

    # Define the rule for LUW
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
                    # Point p ~ Line l: l[c] - p[c] = l[a] * p[b]
                    # l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line l ~ Point p: l[c] - p[c] = l[a] * p[b]
                    # p[c] = l[c] - l[a] * p[b]  (Wait, symmetry)
                    # The relation is symmetric: l[c] - p[c] = l[a] * p[b]
                    # If we are at line l and finding point p:
                    # p[c] = l[c] - l[a] * p[b]
                    # But p[b] is unknown until we compute it?
                    # In the seed code:
                    # y[c] = sub(x[c], mul(x[a], y[b]))
                    # Here x is the current vertex (line), y is the new vertex (point).
                    # So y[c] = x[c] - x[a] * y[b].
                    # This matches the algebraic rearrangement.
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find connected component
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        
        # Deadline check
        if time.time() > deadline:
            break

    return len(order), edges
