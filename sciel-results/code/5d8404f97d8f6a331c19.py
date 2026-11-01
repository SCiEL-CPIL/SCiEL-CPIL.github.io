"""Cage construction: 4-regular (general k) graphs of given girth, minimized vertices."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_START = time.time()
_DEADLINE = 130.0


def _elapsed():
    return time.time() - _START


def _time_ok():
    return _elapsed() < _DEADLINE


def _verify_simple_regular(edges, n, k):
    seen = set()
    deg = [0] * n
    for i, j in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    return True


def _min_girth_bfs(n, edges, k):
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    best = 10 ** 9
    for s in range(n):
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(nb, s)] = 1
            dq.append((nb, s))
        while dq:
            cur, prev = dq.popleft()
            d = dist[(cur, prev)]
            for nb in adj[cur]:
                if nb == prev:
                    continue
                key = (nb, cur)
                if key in dist:
                    L = d + 1
                    if L < best:
                        best = L
                else:
                    dist[key] = d + 1
                    dq.append((nb, cur))
        if best <= k:
            break
    return best if best < 10 ** 9 else 10 ** 9


def _luzebnik_ustimenko_woldar(k, g):
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
        if not _time_ok():
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _voltage_graph_4_10():
    """Construct (4,10)-cage via voltage graph lift over Z_80 with 4-vertex base.
    
    Base: 4-regular multigraph on vertices {0,1,2,3} with two parallel edges
    between each pair of adjacent vertices in a 4-cycle: (0,1)x2, (1,2)x2, (2,3)x2, (3,0)x2.
    Lift over Z_80 gives 320 vertices.
    
    We need voltages v0..v7 (one per edge) in Z_80 such that:
    1. No loop: v_i != 0 for all i
    2. No multi-edges: v_i != v_j for edges between same base vertices
    3. No 4-cycles: sum of voltages around any 4-cycle != 0 mod 80
    4. No 6-cycles: sum around any 6-cycle != 0 mod 80
    5. No 8-cycles: sum around any 8-cycle != 0 mod 80
    """
    m = 80
    # Edge indices: 0,1 = edges 0-1; 2,3 = edges 1-2; 4,5 = edges 2-3; 6,7 = edges 3-0
    # For a walk from vertex u to vertex w using edge idx with voltage v:
    #   forward (u->w): add v
    #   backward (w->u): subtract v
    
    # A non-backtracking closed walk of length L in the base corresponds to a sequence
    # of edge traversals. The total voltage is the signed sum of edge voltages.
    
    # For girth >= 10, we need no non-backtracking closed walk of length 4, 6, or 8
    # with total voltage 0 mod 80.
    
    # Let's enumerate all non-backtracking closed walks of length 4, 6, 8 in the base
    # and derive constraints.
    
    # Base adjacency: 0<->1 (edges 0,1), 1<->2 (edges 2,3), 2<->3 (edges 4,5), 3<->0 (edges 6,7)
    # Non-backtracking means we can't immediately reverse the same edge.
    
    # For length 4: The only non-backtracking 4-cycles in this multigraph are:
    # 0->1->2->3->0 and 0->3->2->1->0 (and their edge-variant versions)
    # With two parallel edges each, there are 2^4 = 16 such walks for each direction.
    # A walk 0->1(e_a)->2(e_b)->3(e_c)->0(e_d) has voltage: v_a + v_b + v_c + v_d
    # where a in {0,1}, b in {2,3}, c in {4,5}, d in {6,7}
    # We need: for all choices, v_a + v_b + v_c + v_d != 0 mod 80
    
    # For length 6: walks like 0->1->2->1->2->3->... wait, that's not closed.
    # Non-backtracking 6-cycles: 0->1->2->3->0->1->2? No, that's not closed either.
    # Actually in a 4-cycle base, non-backtracking walks of even length >= 4 that are closed:
    # Length 4: around the cycle
    # Length 6: go around 1.5 times? Not possible in a simple cycle.
    # Actually, in a multigraph with parallel edges, we can have walks that use different
    # parallel edges. E.g., 0->1(edge0)->2(edge2)->1(edge3)->0(edge7)? Wait, 1->0 isn't an edge.
    # The base is a 4-cycle, so from 1 you can only go to 0 or 2.
    # Non-backtracking 6-cycle: 0->1->2->3->0->1->2? No, not closed.
    # 0->1->2->3->0->3->2? No, 0->3 is valid but then 3->2 is backtracking if we came from 0... 
    # Wait, non-backtracking means the next edge is not the reverse of the previous edge.
    # 0->1(edge a): prev=1, can't go back to 0 via edge a
    # 1->2(edge b): prev=2, can't go back to 1 via edge b
    # 2->3(edge c): prev=3, can't go back to 2 via edge c
    # 3->0(edge d): prev=0, can't go back to 3 via edge d
    # Now at 0, prev=0 came from 3 via edge d. Can go to 1 via edge 0 or 1 (not d, which is 6 or 7)
    # 0->1(edge e): prev=1
    # 1->2(edge f): prev=2
    # Now at 2, need to close back to 0 in 0 more steps? We've used 6 steps: 0->1->2->3->0->1->2
    # That's not closed. We need to be back at 0.
    
    # Actually, let me reconsider. In a 4-cycle, a non-backtracking closed walk of length 6:
    # The walk must visit vertices in a pattern. Starting at 0:
    # Step 1: 0->1
    # Step 2: 1->2 (can't go back to 0)
    # Step 3: 2->3 (can't go back to 1)
    # Step 4: 3->0 (can't go back to 2)
    # Step 5: 0->1 (can't go back to 3)
    # Step 6: 1->? must be 2 (can't go back to 0). But we're at 2, not 0. Not closed.
    # 
    # Alternative: 
    # Step 1: 0->1
    # Step 2: 1->0? No, backtracking.
    # So from 1, must go to 2.
    # It seems like in a 4-cycle, all non-backtracking walks of length >= 4 that are closed
    # must have length a multiple of 4. So length 6 and 8:
    # Length 8: 0->1->2->3->0->1->2->3->0 (go around twice)
    # Voltage: v_a + v_b + v_c + v_d + v_e + v_f + v_g + v_h where a,e in {0,1}, b,f in {2,3}, c,g in {4,5}, d,h in {6,7}
    
    # Wait, I need to double-check: can we have a non-backtracking closed walk of length 6?
    # In a 4-cycle graph (even with parallel edges), the vertices alternate between {0,2} and {1,3}.
    # A closed walk of length 6 would need to return to start after 6 steps. Since 6 is even,
    # we'd be at the same parity class. But the 4-cycle has diameter 2, and...
    # Actually, 0->1->2->1->2->3->0? Let's check: 
    # 0->1: ok
    # 1->2: ok (not backtracking)
    # 2->1: backtracking! We just came from 1 to 2, so 2->1 is backtracking.
    # So that doesn't work.
    # 
    # What about 0->1->2->3->2->1->0? 
    # 0->1: ok
    # 1->2: ok
    # 2->3: ok
    # 3->2: backtracking! We just came from 2 to 3.
    # Doesn't work.
    #
    # It appears that in a 4-cycle (even with parallel edges), all non-backtracking closed walks
    # have length divisible by 4. So we only need to check lengths 4 and 8.
    
    # Constraints:
    # For all a in {0,1}, b in {2,3}, c in {4,5}, d in {6,7}:
    #   v[a] + v[b] + v[c] + v[d] != 0 mod 80  (no 4-cycles)
    # For all a,e in {0,1}, b,f in {2,3}, c,g in {4,5}, d,h in {6,7}:
    #   v[a]+v[b]+v[c]+v[d]+v[e]+v[f]+v[g]+v[h] != 0 mod 80  (no 8-cycles)
    # Also: v[i] != 0 for all i (no loops)
    # And: v[0] != v[1], v[2] != v[3], v[4] != v[5], v[6] != v[7] (no multi-edges)
    
    # Let's try a systematic search. We'll use a greedy approach with backtracking.
    # To make it feasible, we'll fix some voltages and search for others.
    
    # Symmetry: we can WLOG set v[0] = 1 (by scaling, since Z_80 isn't a field, but we can try)
    # Actually, Z_80 isn't a field, so scaling isn't straightforward. Let's just search.
    
    # Better approach: use the fact that if we set v[0]=a, v[1]=b, v[2]=c, v[3]=d, v[4]=e, v[5]=f, v[6]=g, v[7]=h
    # Then the 4-cycle constraints are:
    # a+c+e+g, a+c+e+h, a+c+f+g, a+c+f+h, a+d+e+g, a+d+e+h, a+d+f+g, a+d+f+h
    # b+c+e+g, b+c+e+h, b+c+f+g, b+c+f+h, b+d+e+g, b+d+e+h, b+d+f+g, b+d+f+h
    # All != 0 mod 80
    
    # And 8-cycle constraints (sum of two 4-cycle voltages):
    # (a+c+e+g) + (a+c+e+g) = 2(a+c+e+g) etc.
    # Actually, the 8-cycle voltage is the sum of two 4-cycle voltages (not necessarily the same).
    # So we need: for any two 4-cycle voltages s1, s2, s1+s2 != 0 mod 80.
    # This means no 4-cycle voltage is the negative of another 4-cycle voltage.
    
    # Let's try a random search with constraints.
    import random
    random.seed(42)
    
    def check_voltages(v):
        # v is a list of 8 voltages
        # Check no loops
        for x in v:
            if x % m == 0:
                return False
        # Check no multi-edges
        if v[0] % m == v[1] % m:
            return False
        if v[2] % m == v[3] % m:
            return False
        if v[4] % m == v[5] % m:
            return False
        if v[6] % m == v[7] % m:
            return False
        # Compute all 4-cycle voltages
        four_cycle_sums = []
        for a in [v[0], v[1]]:
            for b in [v[2], v[3]]:
                for c in [v[4], v[5]]:
                    for d in [v[6], v[7]]:
                        s = (a + b + c + d) % m
                        four_cycle_sums.append(s)
                        if s == 0:
                            return False
        # Check 8-cycle: no two 4-cycle sums add to 0
        sum_set = set(four_cycle_sums)
        for s in four_cycle_sums:
            if (-s) % m in sum_set:
                return False
        return True
    
    # Try random assignments
    best_result = None
    for attempt in range(50000):
        if not _time_ok():
            break
        v = [random.randint(1, m-1) for _ in range(8)]
        if check_voltages(v):
            # Build the graph and verify
            n = 4 * m
            edges = []
            base_edges = [
                (0, 1, 0), (0, 1, 1),
                (1, 2, 2), (1, 2, 3),
                (2, 3, 4), (2, 3, 5),
                (3, 0, 6), (3, 0, 7)
            ]
            for v_base in range(4):
                for g in range(m):
                    for u, w, idx in base_edges:
                        if u == v_base:
                            g2 = (g + v[idx]) % m
                            edges.append((v_base * m + g, w * m + g2))
            
            seen = set()
            unique_edges = []
            for i, j in edges:
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in seen:
                    seen.add((a, b))
                    unique_edges.append((a, b))
            
            deg = [0] * n
            for i, j in unique_edges:
                deg[i] += 1
                deg[j] += 1
            if max(deg) != 4 or min(deg) != 4:
                continue
            
            if _min_girth_bfs(n, unique_edges, 4) < 10:
                continue
            
            return n, unique_edges
    
    return None, None


def _petersen():
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (5, 6), (5, 7), (5, 8), (5, 9),
        (6, 7), (6, 8), (6, 9),
        (7, 8), (7, 9), (8, 9),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    return 10, edges


def _hoffman_singleton():
    n = 50
    edges = []
    def idx(a, b):
        return a * 10 + b
    for a in range(5):
        for b in range(10):
            v = idx(a, b)
            nbs = [
                idx(a, (b + 1) % 10), idx(a, (b - 1) % 10),
                idx((a + 1) % 5, (2 * b) % 10), idx((a + 1) % 5, (2 * b + 1) % 10),
                idx((a + 2) % 5, (4 * b) % 10), idx((a + 2) % 5, (4 * b + 1) % 10),
            ]
            for w in nbs:
                if v < w:
                    edges.append((v, w))
    return n, edges


def _moore_girth5(k):
    known = {3: 10, 4: 14, 5: 30, 6: 42, 7: 50}
    if k in known:
        if k == 7:
            return _hoffman_singleton()
        if k == 3:
            return _petersen()
    return _luzebnik_ustimenko_woldar(k, 5)


def construct(k, g):
    if k == 3 and g == 5:
        return _petersen()
    if k == 7 and g == 5:
        return _hoffman_singleton()
    if g == 5:
        return _moore_girth5(k)
    if k == 4 and g == 10:
        n, edges = _voltage_graph_4_10()
        if n is not None:
            return n, edges
        n, edges = _luzebnik_ustimenko_woldar(k, g)
        return n, edges
    n, edges = _luzebnik_ustimenko_woldar(k, g)
    return n, edges
# EVOLVE-BLOCK-END
