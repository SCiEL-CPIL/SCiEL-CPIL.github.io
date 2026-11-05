"""Cayley-graph construction for 4-regular girth-10 graph."""
import math
import numpy as np

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    best_n = None
    best_edges = None
    
    # Strategy 1: Cyclic groups Z_m with S = {a, -a, b, -b}
    # Moore bound for (4,10) is 243, so n >= 243
    # Use fast matrix-based girth check on Hashimoto (non-backtracking) graph
    
    for m in range(243, 400):
        found = False
        for a in range(1, m // 2 + 1):
            if 2 * a == m:
                continue
            for b in range(a + 1, m // 2 + 1):
                if 2 * b == m:
                    continue
                if a == b or a + b == m:
                    continue
                if math.gcd(math.gcd(a, b), m) != 1:
                    continue
                
                if _check_girth_fast(m, [a, -a, b, -b], g):
                    edge_set = set()
                    for i in range(m):
                        for s in [a, b]:
                            j = (i + s) % m
                            if i < j:
                                edge_set.add((i, j))
                            else:
                                edge_set.add((j, i))
                    edges = sorted(edge_set)
                    
                    if best_n is None or m < best_n:
                        best_n = m
                        best_edges = edges
                    found = True
                    break
            if found:
                break
        if found:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Strategy 2: Z_a x Z_b with S = {(a1,0),(−a1,0),(0,b1),(0,−b1)}
    for a_mod in range(15, 50):
        for b_mod in range(15, 50):
            n = a_mod * b_mod
            if n < 243 or n > 400:
                continue
            for a1 in range(1, a_mod // 2 + 1):
                if 2 * a1 == a_mod:
                    continue
                for b1 in range(1, b_mod // 2 + 1):
                    if 2 * b1 == b_mod:
                        continue
                    if _check_girth_ab(a_mod, b_mod, a1, b1, g):
                        edges = _build_edges_ab(a_mod, b_mod, a1, b1)
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                        break
                else:
                    continue
                break
    
    if best_n is not None:
        return best_n, best_edges
    
    return _construct_fallback(4, g)


def _check_girth_fast(m, S, g):
    """Check girth >= g for cyclic group Z_m with generating set S.
    Uses BFS in the non-backtracking state space (vertex, last_step).
    For each starting state, check if we can return to it in < g steps.
    Optimized: only need to check one representative per orbit.
    """
    S_list = list(S)
    nS = len(S)
    
    # State: (position, last_gen_idx) where last_gen_idx indicates which step we took
    # Non-backtracking: next step must not be the inverse of last step
    # We do BFS from each state, but we can be smart:
    # Just check: does there exist a non-backtracking closed walk of length L for any 2 <= L < g?
    
    # Use the transfer matrix approach but with sparse vectors
    # State space: m * nS states
    # From state (i, si), can go to ((i+S_list[si])%m, sj) for all sj where S_list[sj] != -S_list[si]
    
    # For efficiency, precompute transitions
    # trans[i][si] = list of (j, sj) reachable
    
    # Instead of full matrix, do iterative BFS from all states simultaneously
    # Count number of non-backtracking walks of length L from each state back to itself
    
    # Use numpy: represent the set of states as a 2D array or use the matrix approach
    # nH = m * nS <= 4*400 = 1600
    # H is 1600x1600. H^L for L up to 9.
    # numpy matmul of 1600x1600 is ~1600^3 = 4e9 ops... that's ~4 seconds per multiply in numpy.
    # 8 multiplications = 32 seconds. Too slow for many candidates.
    
    # Better approach: for cyclic groups, use the algebraic structure.
    # A non-backtracking walk is s1, s2, ..., sL with si != -s_{i+1} (cyclic) and sum = 0 mod m.
    # For L=2: s1+s2=0, s2=-s1, but s2 != -s1 violated. No.
    # For L=3: s1+s2+s3=0, s2!=-s1, s3!=-s2, s1!=-s3.
    #   Possible: (a,a,-2a), (a,b,-a-b), (a,-b,b-a), etc.
    #   Check if any sum to 0 mod m.
    
    # For general L, enumerate all valid sequences.
    # Number of non-backtracking sequences of length L: 4 * 3^(L-1)
    # For L=9: 4 * 3^8 = 26244. Times m candidates... 
    # But we can precompute all valid sequences once, then for each (m,a,b) check sums.
    
    # Precompute all non-backtracking cyclic sequences of length L for L=2..g-1
    # A sequence is "cyclic non-backtracking" if s_i != -s_{i+1} for all i (mod L)
    
    # For each L, enumerate non-backtracking cyclic sequences and check sum mod m
    for L in range(2, g):
        found = _has_closed_nonbacktracking(m, S_list, L)
        if found:
            return False
    return True


def _has_closed_nonbacktracking(m, S, L):
    """Check if there's a non-backtracking closed walk of length L in Z_m with steps from S."""
    # Enumerate all sequences s1,...,sL from S with s_i != -s_{i+1} for all i (cyclic)
    # and sum(s_i) = 0 mod m
    # Use DFS with pruning: track running sum mod m
    
    def dfs(pos, last, running_sum):
        if pos == L:
            # Cyclic: last != -S[0]... but we don't know S[0] here.
            # Actually the constraint is s_i != -s_{i+1} for i=1..L-1 and s_L != -s_1.
            # We track last = s_L, and need s_L != -s_1.
            # We need to pass s_1. Let's restructure.
            return running_sum % m == 0
        for s in S:
            if s == -last:
                continue
            if dfs(pos + 1, s, running_sum + s):
                return True
        return False
    
    # We need to handle the cyclic constraint s_L != -s_1
    # Do it properly: enumerate s_1, then DFS for s_2...s_L with constraint
    for s1 in S:
        def dfs2(pos, last, running_sum):
            if pos == L:
                if last != -s1 and running_sum % m == 0:
                    return True
                return False
            for s in S:
                if s == -last:
                    continue
                if dfs2(pos + 1, s, running_sum + s):
                    return True
            return False
        
        if dfs2(2, s1, s1):
            return True
    
    return False


def _check_girth_ab(a_mod, b_mod, a1, b1, g):
    """Check girth for Z_a x Z_b with S = {(a1,0),(-a1,0),(0,b1),(0,-b1)}."""
    S = [(a1, 0), (-a1 % a_mod, 0), (0, b1), (0, -b1 % b_mod)]
    
    for L in range(2, g):
        if _has_closed_nonbacktracking_ab(a_mod, b_mod, S, L):
            return False
    return True


def _has_closed_nonbacktracking_ab(a_mod, b_mod, S, L):
    """Check if there's a non-backtracking closed walk of length L in Z_a x Z_b."""
    for s1 in S:
        def dfs2(pos, last, rx, ry):
            if pos == L:
                if last != (-s1[0] % a_mod, -s1[1] % b_mod) and rx % a_mod == 0 and ry % b_mod == 0:
                    return True
                return False
            for s in S:
                if s == (-last[0] % a_mod, -last[1] % b_mod):
                    continue
                if dfs2(pos + 1, s, rx + s[0], ry + s[1]):
                    return True
            return False
        
        if dfs2(2, s1, s1[0], s1[1]):
            return True
    
    return False


def _build_edges_ab(a_mod, b_mod, a1, b1):
    """Build edge list for Z_a x Z_b Cayley graph with S = {(a1,0),(0,b1)} positive gens."""
    edge_set = set()
    for x in range(a_mod):
        for y in range(b_mod):
            for (sx, sy) in [(a1, 0), (0, b1)]:
                nx = (x + sx) % a_mod
                ny = (y + sy) % b_mod
                v1 = x * b_mod + y
                v2 = nx * b_mod + ny
                if v1 < v2:
                    edge_set.add((v1, v2))
                else:
                    edge_set.add((v2, v1))
    return sorted(edge_set)


def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
