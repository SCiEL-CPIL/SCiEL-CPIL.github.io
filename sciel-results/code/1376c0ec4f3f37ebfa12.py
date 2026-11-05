"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import numpy as np

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
        
    # Try Cayley graph of A6 (order 360)
    # A6 is the alternating group on 6 letters.
    
    def perm_sign(p):
        inv = 0
        for i in range(len(p)):
            for j in range(i+1, len(p)):
                if p[i] > p[j]:
                    inv += 1
        return 1 if inv % 2 == 0 else -1
        
    perms = list(itertools.permutations(range(6)))
    A6 = [p for p in perms if perm_sign(p) == 1]
    idx = {p: i for i, p in enumerate(A6)}
    
    def compose(p, q):
        return tuple(p[q[i]] for i in range(6))
        
    id_perm = (0, 1, 2, 3, 4, 5)
    
    order_3 = []
    order_5 = []
    
    for p in A6:
        if p == id_perm: continue
        cur = p
        for _ in range(1, 10):
            cur = compose(cur, p)
            if cur == id_perm:
                if _ == 3: order_3.append(p)
                if _ == 5: order_5.append(p)
                break
                
    def inv(p):
        r = [0]*6
        for i in range(6):
            r[p[i]] = i
        return tuple(r)
        
    candidates = order_3 + order_5
    
    for s1 in candidates:
        for s2 in candidates:
            s1_inv = inv(s1)
            s2_inv = inv(s2)
            
            if s1 == s2 or s1 == s2_inv or s2 == s1_inv or s1 == s1_inv or s2 == s2_inv:
                continue
            
            # Check girth >= g
            from collections import deque
            q = deque()
            q.append((0, id_perm, None))
            found_cycle = False
            visited = set()
            
            while q:
                l, v, prev = q.popleft()
                if l >= g:
                    continue
                if l > 0 and v == id_perm:
                    found_cycle = True
                    break
                
                for s in [s1, s1_inv, s2, s2_inv]:
                    if prev is not None and s == inv(prev):
                        continue
                    next_v = compose(v, s)
                    state = (next_v, s)
                    if state in visited:
                        continue
                    visited.add(state)
                    q.append((l+1, next_v, s))
                    
            if not found_cycle:
                edges = []
                for i, v in enumerate(A6):
                    for s in [s1, s2]:
                        u_idx = idx[v]
                        w = compose(v, s)
                        w_idx = idx[w]
                        if u_idx < w_idx:
                            edges.append((u_idx, w_idx))
                return len(A6), edges
                
    return _construct_fallback(4, g)


def _check_girth_cyclic(m, a, b, g):
    """Check that the Cayley graph of Z_m with S={a,-a,b,-b} has girth >= g.
    We check that there are no non-backtracking closed walks of length < g.
    """
    # A non-backtracking walk of length L is a sequence s1, s2, ..., sL from {a, -a, b, -b}
    # such that si != -s_{i+1} for all i (no immediate backtracking),
    # and sum(s1+...+sL) = 0 mod m.
    
    # For small L, we can enumerate.
    # L = 2: s1 + s2 = 0, s2 = -s1. But s2 != -s1 is violated. So no length-2 non-backtracking closed walks.
    # L = 3: s1+s2+s3=0, s2!=-s1, s3!=-s2. 
    #   Possibilities: (a, a, -2a), (a, b, -(a+b)), etc.
    #   We need to check if any such triple sums to 0 mod m with the non-backtracking constraint.
    
    S = [a, -a, b, -b]
    
    for L in range(2, g):
        # Enumerate all non-backtracking sequences of length L
        # This is 4 * 3^(L-1) possibilities
        # For L up to 9, that's 4 * 3^8 = 26244, which is fine.
        
        def _enumerate(start, length, current_sum, prev):
            if length == 0:
                return current_sum % m == 0
            for s in S:
                if s == -prev:
                    continue
                if _enumerate(s, length - 1, current_sum + s, s):
                    return True
            return False
        
        for s in S:
            if _enumerate(s, L - 1, s, s):
                return False
    
    return True


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
