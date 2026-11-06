"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3)."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # seconds; return with margin

def construct():
    """
    Returns a valid coloring for n vertices.
    Strategy:
    1. Try to construct a valid coloring for n=32 using a deterministic algebraic method (GF(32)).
    2. If successful, return n=32.
    3. If not, fall back to local search to find the best possible n within the time limit.
    """
    
    def _triples_to_idx(n):
        triples = list(combinations(range(n), 3))
        m = len(triples)
        idx = {}
        for i, t in enumerate(triples):
            idx[t] = i
        return triples, idx, m

    def _build_quads(n):
        m = len(list(combinations(range(n), 3)))
        def t_idx(a, b, c):
            cnt = 0
            for x in range(a):
                cnt += (n - x - 1) * (n - x - 2) // 2
            for y in range(a + 1, b):
                cnt += (n - y - 1)
            cnt += (c - b - 1)
            return cnt
        quads = []
        for q in combinations(range(n), 4):
            a, b, c, d = q
            idxs = [t_idx(a, b, c), t_idx(a, b, d), t_idx(a, c, d), t_idx(b, c, d)]
            quads.append(idxs)
        return quads

    def _build_pents(n):
        m = len(list(combinations(range(n), 3)))
        def t_idx(a, b, c):
            cnt = 0
            for x in range(a):
                cnt += (n - x - 1) * (n - x - 2) // 2
            for y in range(a + 1, b):
                cnt += (n - y - 1)
            cnt += (c - b - 1)
            return cnt
        pents = []
        for p in combinations(range(n), 5):
            a, b, c, d, e = p
            idxs = []
            for t in combinations(p, 3):
                idxs.append(t_idx(*t))
            pents.append(idxs)
        return pents

    def _count_violations(colors, quads, pents):
        red_k4 = 0
        for idxs in quads:
            if all(colors[i] == 0 for i in idxs):
                red_k4 += 1
        blue_k5 = 0
        for idxs in pents:
            if all(colors[i] == 1 for i in idxs):
                blue_k5 += 1
        return red_k4, blue_k5

    def _local_search(colors, quads, pents, time_budget):
        colors = colors.copy()
        m = len(colors)
        quad_red = np.zeros(len(quads), dtype=np.int8)
        for qi, idxs in enumerate(quads):
            for i in idxs:
                if colors[i] == 0:
                    quad_red[qi] += 1
        pent_blue = np.zeros(len(pents), dtype=np.int8)
        for pi, idxs in enumerate(pents):
            for i in idxs:
                if colors[i] == 1:
                    pent_blue[pi] += 1
        red_k4 = int(np.sum(quad_red == 4))
        blue_k5 = int(np.sum(pent_blue == 10))
        total_v = red_k4 + blue_k5
        if total_v == 0:
            return colors, red_k4, blue_k5

        tri_quads = [[] for _ in range(m)]
        for qi, idxs in enumerate(quads):
            for i in idxs:
                tri_quads[i].append(qi)
        tri_pents = [[] for _ in range(m)]
        for pi, idxs in enumerate(pents):
            for i in idxs:
                tri_pents[i].append(pi)

        t_end = time.time() + time_budget
        order = list(range(m))
        random.shuffle(order)
        improved = True
        while improved and time.time() < t_end:
            improved = False
            for t in order:
                if time.time() >= t_end:
                    break
                cur = colors[t]
                new = 1 - cur
                delta_red = 1 if new == 0 else -1
                delta_blue = 1 if new == 1 else -1
                dq = 0
                for qi in tri_quads[t]:
                    old = quad_red[qi]
                    new_q = old + delta_red
                    if old == 4:
                        dq -= 1
                    if new_q == 4:
                        dq += 1
                dp = 0
                for pi in tri_pents[t]:
                    old = pent_blue[pi]
                    new_p = old + delta_blue
                    if old == 10:
                        dp -= 1
                    if new_p == 10:
                        dp += 1
                if dq + dp < 0:
                    colors[t] = new
                    for qi in tri_quads[t]:
                        quad_red[qi] += delta_red
                    for pi in tri_pents[t]:
                        pent_blue[pi] += delta_blue
                    red_k4 += dq
                    blue_k5 += dp
                    improved = True
            random.shuffle(order)
        return colors, red_k4, blue_k5

    def _random_valid(n, time_budget):
        triples, idx, m = _triples_to_idx(n)
        quads = _build_quads(n)
        pents = _build_pents(n)
        attempts = 0
        while time.time() < _START + time_budget:
            attempts += 1
            p_red = 0.35
            colors = (np.random.rand(m) < p_red).astype(np.int8)
            colors = colors.tolist()
            colors, r4, b5 = _local_search(colors, quads, pents, 2.0)
            if r4 == 0 and b5 == 0:
                return colors
        return None

    # Strategy 1: Deterministic Algebraic Construction for n=32 (GF(32))
    # GF(32) is a field with 32 elements. We can represent elements as integers 0..31.
    # Multiplication in GF(32) is defined using a primitive polynomial, e.g., x^5 + x^2 + 1.
    # Let's define multiplication in GF(32).
    # Primitive polynomial: 0b100101 = 37 (x^5 + x^2 + 1)
    # We need a multiplication table or a fast function.
    
    def gf32_mul(a, b):
        if a == 0 or b == 0:
            return 0
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 32: # x^5
                a ^= 37  # reduce by x^5 + x^2 + 1
        return res

    def gf32_pow(a, e):
        res = 1
        while e:
            if e & 1:
                res = gf32_mul(res, a)
            a = gf32_mul(a, a)
            e >>= 1
        return res

    # Try to construct a coloring for n=32
    # A common construction for R(4, 5; 3) > 32 might involve:
    # Color (a,b,c) red if (a-b)(b-c)(c-a) is a non-zero square (or some specific property) in GF(32).
    # Or simpler: Color based on the value of a polynomial.
    # Let's try: Color(a,b,c) = 0 (Red) if a*b + b*c + c*a == 0, else 1 (Blue)?
    # Or: Color(a,b,c) = 0 if a+b+c == 0?
    
    # Let's try a few simple algebraic rules and check for violations.
    # Rule 1: Red if a*b + b*c + c*a == 0
    # Rule 2: Red if a+b+c == 0
    # Rule 3: Red if (a-b)(b-c)(c-a) is a quadratic residue (non-zero square)
    
    # Quadratic residues in GF(32):
    # The non-zero elements are a cyclic group of order 31.
    # Squares are elements of the form g^(2k). Since 31 is prime, there are 15 non-zero squares.
    # Let's find the squares.
    # Primitive element in GF(32) with x^5+x^2+1:
    # 1 = 1
    # 2 = x
    # 4 = x^2
    # 8 = x^3
    # 16 = x^4
    # 32 = x^5 = x^2+1 = 5
    # So 5 is a primitive element?
    # 5^1 = 5
    # 5^2 = 25
    # 5^3 = 25*5 = (x^4+x^2+x+1)*5 ... let's just compute the set of squares.
    
    squares = set()
    for i in range(1, 32):
        squares.add(gf32_pow(i, 15)) # i^((32-1)/2) is 1 if square, -1 if not? No, in char 2, -1=1.
        # Actually, i^15 is 1 if i is a square, 0 if i=0. But we want the set of squares.
        # Just compute i^2 for all i.
    squares = set(gf32_pow(i, 2) for i in range(32))
    
    # Let's test Rule 3: Red if (a-b)(b-c)(c-a) is in squares (and non-zero).
    # Note: In char 2, a-b = a+b.
    
    n_try = 32
    triples, idx, m = _triples_to_idx(n_try)
    quads = _build_quads(n_try)
    pents = _build_pents(n_try)
    
    # Try Rule 3
    colors = [0] * m
    for i, (a, b, c) in enumerate(triples):
        val = gf32_mul(gf32_mul(a+b, b+c), c+a)
        if val != 0 and val in squares:
            colors[i] = 0 # Red
        else:
            colors[i] = 1 # Blue
            
    r4, b5 = _count_violations(colors, quads, pents)
    if r4 == 0 and b5 == 0:
        return {'n': n_try, 'colors': colors}
        
    # Try Rule 1: Red if a*b + b*c + c*a == 0
    colors = [0] * m
    for i, (a, b, c) in enumerate(triples):
        val = gf32_mul(a,b) ^ gf32_mul(b,c) ^ gf32_mul(c,a)
        if val == 0:
            colors[i] = 0
        else:
            colors[i] = 1
    r4, b5 = _count_violations(colors, quads, pents)
    if r4 == 0 and b5 == 0:
        return {'n': n_try, 'colors': colors}

    # If algebraic construction fails, fall back to random search
    deadline = _START + _DEADLINE - 5.0
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    for n in range(10, 41):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if n <= 20:
            budget = min(remaining, 15.0)
        elif n <= 28:
            budget = min(remaining, 25.0)
        else:
            budget = min(remaining, 35.0)
        if budget < 3.0:
            break
        colors = _random_valid(n, budget)
        if colors is not None:
            best_n = n
            best_colors = colors
        if time.time() >= deadline:
            break
            
    return {'n': best_n, 'colors': best_colors}
