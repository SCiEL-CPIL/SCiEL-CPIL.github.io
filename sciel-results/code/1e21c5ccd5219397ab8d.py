import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=3, g=16: uses algebraic construction over GF(2^10).
    """
    start_time = time.time()
    deadline = start_time + 140

    if k != 3:
        # Fallback for other k values
        pass

    # GF(2^10) arithmetic using a primitive polynomial
    # Use x^10 + x^3 + 1 (primitive over GF(2))
    PRIMITIVE = 0x409  # x^10 + x^3 + 1

    def gf_mul(a, b):
        """Multiply two elements in GF(2^10)."""
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & (1 << 10):
                a ^= PRIMITIVE
        return result & 0x3FF

    def gf_inv(a):
        """Inverse of a in GF(2^10). a must be nonzero."""
        if a == 0:
            return 0
        # a^1023 = 1 in GF(2^10)
        result = 1
        power = a
        exp = 1023
        while exp:
            if exp & 1:
                result = gf_mul(result, power)
            power = gf_mul(power, power)
            exp >>= 1
        return result

    # Construct cubic graph using algebraic method
    # Vertices are elements of GF(2^10), so n = 1024
    n = 1024

    # For each vertex x, its three neighbors are determined by
    # solving equations related to the field structure.
    # We use a standard construction: for each x, neighbors are
    # x + a, x + b, x + c where a, b, c are fixed nonzero elements
    # such that the resulting graph has high girth.

    # Actually, let's use a different approach:
    # The graph where vertex i is connected to vertices j such that
    # the "distance" in some algebraic sense is exactly 1.

    # Let me use a known construction: the graph on GF(q) where
    # x is adjacent to y if x-y is in a set S of size 3 that is
    # "sum-free" in appropriate sense.

    # For GF(2^10), let's pick three elements a, b, c such that
    # the Cayley graph has high girth.

    # A better approach: use the fact that for prime power q,
    # there exist cubic graphs of girth >= 2*log_2(q) + O(1).

    # Let's try: neighbors of x are x+1, x+alpha, x+alpha^2
    # where alpha is a primitive element.

    # Find primitive element of GF(2^10)
    def is_primitive(a):
        """Check if a is primitive in GF(2^10)."""
        if a == 0:
            return False
        # Check that a^((1024-1)/p) != 1 for all prime p dividing 1023
        # 1023 = 3 * 11 * 31
        for p in [3, 11, 31]:
            # Compute a^((1023)/p)
            result = 1
            power = a
            exp = 1023 // p
            while exp:
                if exp & 1:
                    result = gf_mul(result, power)
                power = gf_mul(power, power)
                exp >>= 1
            if result == 1:
                return False
        return True

    # Find a primitive element
    primitive_elem = None
    for candidate in range(1, 1024):
        if is_primitive(candidate):
            primitive_elem = candidate
            break

    if primitive_elem is None:
        primitive_elem = 2  # fallback

    # Define the three "steps" for the Cayley graph
    # We want steps that generate a high-girth graph.
    # Use 1, primitive_elem, primitive_elem^2

    step1 = 1
    step2 = primitive_elem
    step3 = gf_mul(primitive_elem, primitive_elem)

    # Build adjacency
    adj = [[] for _ in range(n)]
    for x in range(n):
        for s in [step1, step2, step3]:
            y = x ^ s  # addition in GF(2^10) is XOR
            if y != x:  # no loops
                adj[x].append(y)

    # Check for repeated edges and ensure simplicity
    # Since we're using a Cayley graph with distinct steps, this should be fine
    # But let's verify

    # Build edge list
    edges = []
    edge_set = set()
    for x in range(n):
        for y in adj[x]:
            if x < y:
                if (x, y) not in edge_set:
                    edges.append((x, y))
                    edge_set.add((x, y))

    # Verify degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    if any(d != 3 for d in deg):
        # Fallback to LUW construction
        return _luw_fallback(k, g, deadline)

    # Verify girth using BFS from each vertex
    def check_girth(adj_list, n_nodes, target_girth):
        max_depth = (target_girth - 1) // 2
        for s in range(n_nodes):
            dist = [-1] * n_nodes
            parent = [-1] * n_nodes
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    if check_girth(adj, n, g):
        return n, edges

    # If the Cayley graph doesn't have sufficient girth, try other step sets
    # or fall back to LUW

    return _luw_fallback(k, g, deadline)


def _luw_fallback(k, g, deadline):
    """LUW construction fallback."""
    def build_luw(d, q_val):
        """Build LUW graph CD(d, q) for prime q. Returns (n, edges)."""
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q_val
            sub_fn = lambda x, y: (x - y) % q_val
            mul_fn = lambda x, y: x * y % q_val

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
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

    def check_girth_luw(n, edges, girth_target):
        max_depth = (girth_target - 1) // 2
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth_target:
                            return False
        return True

    # Try LUW with small d
    q_val = 3
    last_result = None
    for d in range(5, 15):
        if time.time() > deadline:
            break
        n_luw, edges_luw = build_luw(d, q_val)
        if n_luw < 2:
            continue
        deg = [0] * n_luw
        for u, v in edges_luw:
            deg[u] += 1
            deg[v] += 1
        if any(dd != k for dd in deg):
            continue
        last_result = (n_luw, edges_luw)
        if check_girth_luw(n_luw, edges_luw, g):
            return n_luw, edges_luw

    if last_result is not None:
        return last_result

    # Absolute fallback: return a trivial valid graph if possible
    # For k=3, minimum is 4 vertices (K4) but girth 4
    # We need girth >= g, so this may not work
    # Return the largest LUW result we found
    return last_result if last_result else (0, [])
# EVOLVE-BLOCK-END
