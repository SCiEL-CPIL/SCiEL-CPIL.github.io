"""
Construct a 4-regular graph of girth >= 10 (the (4,10)-cage problem).

Strategy:
1. Try known good cyclic constructions on Z_320.
2. Time-budgeted search over cyclic groups Z_m with fast vectorized girth check.
3. Search over product groups Z_m x Z_n for potentially smaller graphs.
4. Fallback: LUW construction (guaranteed valid).

The evaluator measures n (lower is better); score = seed/n.
"""
import math
import time
from collections import deque
import numpy as np

_DEADLINE = 150.0
_MARGIN = 10.0


def construct(k, g):
    t0 = time.time()

    if k != 4:
        return _construct_fallback(k, g)

    deadline = t0 + _DEADLINE - _MARGIN

    # --- 1. Known good construction on Z_320 ---
    m0 = 320
    a0, b0 = 1, 13
    if _girth_check_fast(m0, a0, b0, g):
        edges = _build_cyclic_edges(m0, a0, b0)
        return m0, edges

    # Try other pairs on Z_320
    for (aa, bb) in [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
                     (1, 8), (1, 9), (1, 10), (1, 11), (1, 12),
                     (2, 3), (2, 5), (2, 7), (2, 9), (2, 11), (2, 13),
                     (3, 5), (3, 7), (3, 11), (3, 13), (4, 5), (4, 7),
                     (5, 7), (5, 9), (5, 11), (5, 13), (6, 7), (7, 9)]:
        if aa >= bb:
            continue
        if math.gcd(aa, m0) != 1 or math.gcd(bb, m0) != 1:
            continue
        if math.gcd(math.gcd(aa, bb), m0) != 1:
            continue
        if _girth_check_fast(m0, aa, bb, g):
            edges = _build_cyclic_edges(m0, aa, bb)
            return m0, edges

    # --- 2. Fallback: LUW construction (guaranteed valid) ---
    n_luw, edges_luw = _construct_fallback(4, g)
    best_n, best_edges = n_luw, edges_luw

    # --- 3. Time-budgeted search over cyclic groups Z_m ---
    # Search from Moore bound upward
    moore = _moore_bound(4, g)
    for m in range(moore, min(321, m0 + 1)):
        if time.time() > deadline:
            break
        if m == m0:
            continue
        if m % 2 != 0:
            continue
        found = False
        valid_a = [a for a in range(1, m // 2 + 1) if math.gcd(a, m) == 1]
        for a in valid_a:
            if time.time() > deadline:
                break
            valid_b = [b for b in range(a + 1, m // 2 + 1) if math.gcd(b, m) == 1]
            for b in valid_b:
                if time.time() > deadline:
                    break
                if math.gcd(math.gcd(a, b), m) != 1:
                    continue
                if not _quick_check_cyclic(m, a, b, g):
                    continue
                if _girth_check_fast(m, a, b, g):
                    edges = _build_cyclic_edges(m, a, b)
                    best_n, best_edges = m, edges
                    found = True
                    break
            if found:
                break
        if found:
            break

    # --- 4. Search over product groups Z_m x Z_n ---
    if time.time() < deadline:
        result = _search_product_groups(4, g, deadline)
        if result is not None:
            n_prod, edges_prod = result
            if n_prod < best_n:
                best_n, best_edges = n_prod, edges_prod

    return best_n, best_edges


def _moore_bound(k, g):
    """Lower bound on number of vertices for k-regular graph of girth g."""
    if g % 2 == 0:
        d = g // 2
        return 2 * (k - 1) * ((k - 1) ** (d - 1) - 1) // (k - 2) + 2
    else:
        d = (g - 1) // 2
        return k * (k - 1) ** (d - 1) + (k - 1) * ((k - 1) ** (d - 1) - 1) // (k - 2)


def _build_cyclic_edges(m, a, b):
    """Build edge list for Cayley graph of Z_m with S={a,-a,b,-b}."""
    edge_set = set()
    for i in range(m):
        for s in (a, b):
            j = (i + s) % m
            if i < j:
                edge_set.add((i, j))
            else:
                edge_set.add((j, i))
    return sorted(edge_set)


def _quick_check_cyclic(m, a, b, g):
    """Fast necessary conditions for girth >= g in cyclic group."""
    if (a + b) % m == 0:
        return False
    if (2 * a) % m == 0:
        return False
    if (2 * b) % m == 0:
        return False
    if a == b or a == m - b or b == m - a:
        return False
    if (2 * a + b) % m == 0:
        return False
    if (a + 2 * b) % m == 0:
        return False
    if (2 * a - b) % m == 0:
        return False
    if (a - 2 * b) % m == 0:
        return False
    if (3 * a + b) % m == 0:
        return False
    if (a + 3 * b) % m == 0:
        return False
    if (3 * a - b) % m == 0:
        return False
    if (a - 3 * b) % m == 0:
        return False
    return True


def _girth_check_fast(m, a, b, g):
    """
    Fast girth check using numpy for the cyclic group case.
    Uses BFS on state space (position, last_step_idx).
    State encoded as position*4 + last_step_idx, with -1 for no last step.
    """
    steps = np.array([a, -a, b, -b])
    inv_idx = np.array([1, 0, 3, 2])
    
    # Use a set of encoded integers for efficiency
    frontier = {0}  # position 0, no last step (special)
    
    for L in range(1, g):
        new_frontier = set()
        for state in frontier:
            if state < 0:
                pos = 0
                last = -1
            else:
                pos = state // 4
                last = state % 4
            
            for si in range(4):
                if last >= 0 and si == inv_idx[last]:
                    continue
                new_pos = (pos + steps[si]) % m
                new_frontier.add(new_pos * 4 + si)
        
        if L >= 2:
            for state in new_frontier:
                pos = state // 4
                if pos == 0:
                    return False
        frontier = new_frontier
        if not frontier:
            break
    return True


def _search_product_groups(k, g, deadline):
    """Search over product groups Z_m x Z_n for 4-regular graphs of girth >= g."""
    t0 = time.time()
    best_result = None
    
    # For product Z_m x Z_n, we need |m*n| >= moore_bound
    moore = _moore_bound(k, g)
    
    # Try small factors
    for m in range(2, min(50, moore)):
        if time.time() > deadline:
            break
        for n in range(2, min(50, moore // m + 1)):
            if time.time() > deadline:
                break
            total = m * n
            if total < moore:
                continue
            if total >= (best_result[0] if best_result else moore + 100):
                continue
            
            # Try generating sets of form {(a1,b1), (-a1,-b1), (a2,b2), (-a2,-b2)}
            # where (a1,b1) and (a2,b2) are in Z_m x Z_n
            valid_gens = []
            for a1 in range(1, m):
                for b1 in range(0, n):
                    if math.gcd(math.gcd(a1, m), math.gcd(b1, n)) == 1:
                        valid_gens.append((a1, b1))
            
            for i, (a1, b1) in enumerate(valid_gens):
                if time.time() > deadline:
                    break
                for j, (a2, b2) in enumerate(valid_gens[i+1:], i+1):
                    if time.time() > deadline:
                        break
                    
                    # Check if the two generators are independent
                    # and the group is generated
                    if not _check_independent_product(m, n, a1, b1, a2, b2):
                        continue
                    
                    # Quick check
                    if not _quick_check_product(m, n, a1, b1, a2, b2, g):
                        continue
                    
                    # Full girth check
                    if _girth_check_product(m, n, a1, b1, a2, b2, g):
                        edges = _build_product_edges(m, n, a1, b1, a2, b2)
                        if best_result is None or total < best_result[0]:
                            best_result = (total, edges)
    
    return best_result


def _check_independent_product(m, n, a1, b1, a2, b2):
    """Check if two generators generate Z_m x Z_n."""
    # The subgroup generated by (a1,b1) and (a2,b2) has size
    # m*n / gcd of the determinants in some sense
    # For Z_m x Z_n, we need that the ideal generated by all 2x2 minors is the whole ring
    # Simplified: check that the matrix [[a1,a2],[b1,b2]] has rank 2 over Z_m x Z_n
    # This is complex; use a simple necessary condition
    if a1 == 0 and a2 == 0:
        return False
    if b1 == 0 and b2 == 0:
        return False
    # Check that we can reach all residues
    # Use BFS to check connectivity (expensive but correct)
    # For now, use a necessary condition: gcd(a1, m) and gcd(a2, m) should not both be > 1
    # Similarly for b
    if math.gcd(a1, m) > 1 and math.gcd(a2, m) > 1:
        return False
    if math.gcd(b1, n) > 1 and math.gcd(b2, n) > 1:
        return False
    return True


def _quick_check_product(m, n, a1, b1, a2, b2, g):
    """Quick necessary conditions for product group girth."""
    # Check no short cycles of length 2, 3, 4
    # Length 2: (a1,b1) = -(a1,b1) => 2*(a1,b1) = (0,0) mod (m,n)
    if (2 * a1) % m == 0 and (2 * b1) % n == 0:
        return False
    if (2 * a2) % m == 0 and (2 * b2) % n == 0:
        return False
    # (a1,b1) = -(a2,b2)
    if (a1 + a2) % m == 0 and (b1 + b2) % n == 0:
        return False
    # Length 3: (a1,b1) + (a1,b1) + (a2,b2) = (0,0)
    if (2 * a1 + a2) % m == 0 and (2 * b1 + b2) % n == 0:
        return False
    if (2 * a2 + a1) % m == 0 and (2 * b2 + b1) % n == 0:
        return False
    # (a1,b1) + (a2,b2) + (a2,b2) = (0,0)
    if (a1 + 2 * a2) % m == 0 and (b1 + 2 * b2) % n == 0:
        return False
    # (a1,b1) + (a1,b1) + (-a1,-b1) = 0 (backtracking)
    # (a1,b1) + (a2,b2) + (-a1,-b1) = (a2,b2) != 0 (checked)
    return True


def _girth_check_product(m, n, a1, b1, a2, b2, g):
    """Full girth check for product group Z_m x Z_n."""
    # State: (pos_m, pos_n, last_step_idx)
    # last_step_idx: 0=(a1,b1), 1=(-a1,-b1), 2=(a2,b2), 3=(-a2,-b2)
    inv_idx = [1, 0, 3, 2]
    steps = [(a1, b1), (-a1, -b1), (a2, b2), (-a2, -b2)]
    
    # Encode state as (pos_m * n + pos_n) * 4 + last_step_idx
    # -1 for no last step
    frontier = {0}  # position (0,0), no last step
    
    for L in range(1, g):
        new_frontier = set()
        for state in frontier:
            if state < 0:
                pos_m, pos_n, last = 0, 0, -1
            else:
                last = state % 4
                rest = state // 4
                pos_n = rest % n
                pos_m = rest // n
            
            for si in range(4):
                if last >= 0 and si == inv_idx[last]:
                    continue
                new_pos_m = (pos_m + steps[si][0]) % m
                new_pos_n = (pos_n + steps[si][1]) % n
                new_frontier.add((new_pos_m * n + new_pos_n) * 4 + si)
        
        if L >= 2:
            for state in new_frontier:
                pos_m = (state // 4) // n
                pos_n = (state // 4) % n
                if pos_m == 0 and pos_n == 0:
                    return False
        frontier = new_frontier
        if not frontier:
            break
    return True


def _build_product_edges(m, n, a1, b1, a2, b2):
    """Build edge list for Cayley graph of Z_m x Z_n."""
    edge_set = set()
    for i in range(m):
        for j in range(n):
            for (da, db) in [(a1, b1), (a2, b2)]:
                ni = (i + da) % m
                nj = (j + db) % n
                v1 = i * n + j
                v2 = ni * n + nj
                if v1 < v2:
                    edge_set.add((v1, v2))
                else:
                    edge_set.add((v2, v1))
    return sorted(edge_set)


def _construct_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar construction (seed)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
