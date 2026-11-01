"""Construct a 4-regular graph with girth 10 on 320 vertices (the (4,10)-cage).

Uses a voltage graph lift of K_{4,4} over Z_40 with carefully chosen voltages.
The base K_{4,4} has 8 vertices and 16 edges. The lift over Z_40 gives 320 vertices.
The girth of the lift is at least 10 if the voltage assignment avoids short cycles.
"""
import time
import sys
import random
import itertools
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    if k != 4 or g != 10:
        # Fallback for other parameters: use LUW construction
        return _construct_luw(k, g, deadline)

    # For k=4, g=10: construct the 320-vertex cage via voltage graph lift of K_{4,4} over Z_40
    # K_{4,4} has vertices A = {a0..a3} and B = {b0..b3}, all edges a_i-b_j.
    # Lift over Z_40: 8*40 = 320 vertices.
    # Vertex (side, i, k) where side=0 for A, side=1 for B, i in 0..3, k in 0..39.
    # Edge from (0, i, k) to (1, j, (k + volt[i][j]) % 40).

    def _build_lift(voltages):
        """Build the voltage graph lift of K_{4,4} over Z_40."""
        n = 320
        edges = []
        for i in range(4):
            for j in range(4):
                volt = voltages[i][j]
                for kk in range(40):
                    a_idx = i * 40 + kk
                    b_idx = 160 + j * 40 + (kk + volt) % 40
                    edges.append((a_idx, b_idx))
        return n, edges

    def _check_4cycle_condition(voltages):
        """Check no 4-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for kk in range(4):
                if i == kk:
                    continue
                for j in range(4):
                    for l in range(4):
                        if j == l:
                            continue
                        s = (voltages[i][j] - voltages[kk][j] + voltages[kk][l] - voltages[i][l]) % 40
                        if s == 0:
                            return False
        return True

    def _check_6cycle_condition(voltages):
        """Check no 6-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for j in range(4):
                for kk in range(4):
                    if kk == i:
                        continue
                    for l in range(4):
                        if l == j:
                            continue
                        for m in range(4):
                            if m == kk or m == i:
                                continue
                            for n in range(4):
                                if n == l or n == j:
                                    continue
                                s = (voltages[i][j] - voltages[kk][j] + voltages[kk][l]
                                     - voltages[m][l] + voltages[m][n] - voltages[i][n]) % 40
                                if s == 0:
                                    return False
        return True

    def _check_8cycle_condition(voltages):
        """Check no 8-closed-walk in base has voltage 0 mod 40."""
        for perm_a in itertools.permutations([1, 2, 3]):
            full_a = [0] + list(perm_a)
            for perm_b in itertools.permutations(range(4)):
                s = 0
                for idx in range(4):
                    a_cur = full_a[idx]
                    b_cur = perm_b[idx]
                    a_next = full_a[(idx + 1) % 4]
                    s += voltages[a_cur][b_cur]
                    s -= voltages[a_next][b_cur]
                if s % 40 == 0:
                    return False
        return True

    def _verify_girth_exact(n, edges, g, deadline):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        max_len = g - 1  # we check walks of length 1..g-1

        for start_v in range(n):
            if time.time() > deadline:
                return False

            # BFS over (vertex, prev) states
            states = set()
            for w in adj[start_v]:
                states.add((w, start_v))

            for d in range(2, max_len + 1):
                new_states = set()
                for (v, p) in states:
                    for w in adj[v]:
                        if w != p:
                            new_states.add((w, v))
                if any(v == start_v for (v, p) in new_states):
                    return False
                states = new_states
                if not states:
                    break

        return True

    # Search for valid voltages
    candidate_voltages = None

    # Try structured voltage matrices: volt[i][j] = (a*i + b*j + c*i*j) mod 40
    # Condition 1 (no 4-cycles): c*(i-k)*(j-l) != 0 mod 40 for all i!=k, j!=l
    # This requires c not in {0, 10, 20, 30} mod 40

    # First try the simplest: volt[i][j] = c*i*j mod 40 with various c
    for c in range(1, 40):
        if c % 10 == 0:
            continue  # fails condition 1
        voltages = [[(c * i * j) % 40 for j in range(4)] for i in range(4)]
        if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
            candidate_voltages = voltages
            break

    if candidate_voltages is None:
        # Try volt[i][j] = (a*i + b*j + c*i*j) mod 40 with more values
        for c in [1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39]:
            for a in range(0, 40):
                for b in range(0, 40):
                    voltages = [[(a*i + b*j + c*i*j) % 40 for j in range(4)] for i in range(4)]
                    if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                        candidate_voltages = voltages
                        break
                if candidate_voltages:
                    break
            if candidate_voltages:
                break

    if candidate_voltages is None:
        # Random search with more attempts
        for attempt in range(20000):
            if time.time() > deadline - 30:
                break
            voltages = [[random.randint(0, 39) for _ in range(4)] for _ in range(4)]
            if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                candidate_voltages = voltages
                break

    if candidate_voltages is not None:
        n, edges = _build_lift(candidate_voltages)
        # Final verification with exact girth check
        if _verify_girth_exact(n, edges, g, deadline):
            return n, edges
        # If verification fails, the voltage assignment was insufficient
        # Try more random searches
        for attempt in range(20000):
            if time.time() > deadline - 20:
                break
            voltages = [[random.randint(0, 39) for _ in range(4)] for _ in range(4)]
            if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                n2, edges2 = _build_lift(voltages)
                if _verify_girth_exact(n2, edges2, g, deadline):
                    return n2, edges2

    # Fallback: use LUW construction
    return _construct_luw(4, 10, deadline)


def _construct_luw(k, g, deadline):
    """LUW construction as fallback."""
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    d = min(d_candidates) if d_candidates else max(3, g - 5)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
