"""Block Strassen decomposition of the 4x4x5 matrix multiplication tensor."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    # Four 2x2x2 blocks: (row pair, col pair, k-pair)
    # Block (b1,b2,b3) covers rows 2*b1..2*b1+1, cols 2*b2..2*b2+1, k 2*b3..2*b3+1
    blocks = [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)]

    # Strassen 2x2x2 scheme: 5 terms.
    # Term t: for each (a,b,c) in {0,1}^3, U[a,b]=1 if a!=b or c!=0,
    #         V[b,c]=1 if b!=c or a!=0,
    #         W[c,a]=1 if c!=a or b!=0.
    # The 5th term is the negated sum of the first 4 (so each (a,b,c) sums to 0).
    strassen = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                if (a, b, c) == (1, 1, 1):
                    continue  # this is the "missing" position; term 5 handles it
                strassen.append((a, b, c))
    # strassen has 7 entries; we need 5. Use the standard 5-term scheme:
    # Terms (a,b,c) for t=0..3:
    #   t0: (0,0,0)
    #   t1: (0,1,0)
    #   t2: (1,0,1)
    #   t3: (1,1,0)
    #   t4: negated sum of t0..t3
    # Check coverage: each (a,b,c) in {0,1}^3 appears in exactly 4 of {t0,t1,t2,t3,t4}.
    terms = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 1),
        (1, 1, 0),
    ]
    # Build the 5th term as negated sum of the first 4.
    # For each (a,b,c), the sum of U[a,b] over t=0..3 must equal the sum over all 5 terms = 0.
    # So U5[a,b] = -sum_{t=0..3} U_t[a,b], similarly for V and W.
    U5 = {(a, b): -sum(1 for (ta, tb, tc) in terms if (a == ta or tb != a) and True)
          for a in range(2) for b in range(2)}
    # Recompute properly: U_t[a,b]=1 iff (a!=b or c!=0) for the term's (a_t,b_t,c_t).
    def Uval(term, a, b):
        ta, tb, tc = term
        return 1 if (a != b or tc != 0) else 0
    def Vval(term, b, c):
        ta, tb, tc = term
        return 1 if (b != c or ta != 0) else 0
    def Wval(term, c, a):
        ta, tb, tc = term
        return 1 if (c != a or tb != 0) else 0

    U5 = {(a, b): -sum(Uval(t, a, b) for t in terms) for a in range(2) for b in range(2)}
    V5 = {(b, c): -sum(Vval(t, b, c) for t in terms) for b in range(2) for c in range(2)}
    W5 = {(c, a): -sum(Wval(t, c, a) for t in terms) for c in range(2) for a in range(2)}

    all_terms = terms + [(None, None, None)]  # index 4 is the 5th term
    rank = 4 * 5
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for bi, (b1, b2, b3) in enumerate(blocks):
        for t in range(5):
            col = bi * 5 + t
            if t < 4:
                ta, tb, tc = terms[t]
                for a in range(2):
                    for b in range(2):
                        u = Uval(terms[t], a, b)
                        if u:
                            U[(2 * b1 + a) * m + (2 * b2 + b)][col] = u
                for b in range(2):
                    for c in range(2):
                        v = Vval(terms[t], b, c)
                        if v:
                            V[(2 * b2 + b) * p + (2 * b3 + c)][col] = v
                for c in range(2):
                    for a in range(2):
                        w = Wval(terms[t], c, a)
                        if w:
                            W[(2 * b3 + c) * n + (2 * b1 + a)][col] = w
            else:
                for a in range(2):
                    for b in range(2):
                        u = U5[(a, b)]
                        if u:
                            U[(2 * b1 + a) * m + (2 * b2 + b)][col] = u
                for b in range(2):
                    for c in range(2):
                        v = V5[(b, c)]
                        if v:
                            V[(2 * b2 + b) * p + (2 * b3 + c)][col] = v
                for c in range(2):
                    for a in range(2):
                        w = W5[(c, a)]
                        if w:
                            W[(2 * b3 + c) * n + (2 * b1 + a)][col] = w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
