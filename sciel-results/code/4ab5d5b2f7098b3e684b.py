"""
Mutually orthogonal Latin squares of order 10.

Uses the known finite-field construction: since 10 = 2 * 5 and both are prime,
Z_10 is not a field, but Z_5 x Z_2 is isomorphic to Z_10. However, the standard
construction for MOLS of order n = p^k uses the finite field GF(n). For n=10,
which is not a prime power, we cannot use a single finite field.

Wait - 10 is NOT a prime power. So the standard finite field construction does
NOT apply directly. The maximum number of MOLS of order 10 is known to be at
least 2 and at most 9. In fact, it is known that there exist at least 2 MOLS of
order 10, and it is an open question whether 9 exist. Actually, it's known that
there exist exactly... let me think.

Actually, for n = 10, it is known that there exist at least 2 mutually orthogonal
Latin squares. The maximum is not known to be 9. The Euler 36 officers problem
shows that no pair of MOLS exists for order 6, and it was long unknown for order
10. In 1959, it was shown that MOLS of order 10 exist (at least 2).

Actually, let me reconsider. The standard result: for prime power q, there exist
q-1 MOLS of order q. For n=10, which is not a prime power, the maximum number of
MOLS is not known in general, but it is known that at least 2 exist.

Wait, actually I need to be more careful. Let me think about what's actually
known. The task says "Maximize the number of pairwise orthogonal arrays" and
"Return between 1 and 9 squares". The seed program returns 1 square (the cyclic
one) with score 1.0.

Let me try to actually construct MOLS of order 10. One approach: use the fact
that 10 = 2 * 5, and construct squares based on the group Z_2 x Z_5 (which is
isomorphic to Z_10). For a group G of order n, we can construct a Latin square
L_g(h) = g * h (group operation). Two such squares from different groups or
different "directions" may be orthogonal.

Actually, a cleaner approach: since Z_10 ≅ Z_2 × Z_5, we can represent each
element as a pair (a, b) where a ∈ Z_2 and b ∈ Z_5. Then we can define Latin
squares using linear combinations over these components.

For a Latin square of order n, we need L(i, j) such that for each fixed i,
{L(i, j) : j} is all of Z_n, and for each fixed j, {L(i, j) : i} is all of Z_n.

If we identify Z_10 with Z_2 × Z_5, then an element is (a, b). We can define
squares as:
L_m(i, j) = (i_a + m_a * j_a, i_b + m_b * j_b) mod (2, 5)

Wait, this needs more care. Let me think differently.

Actually, the simplest correct approach: just use backtracking with the cyclic
square as a starting point, and try to add more squares. The current program
does this but may be too slow. Let me optimize the backtracking.

Alternatively, I recall that for order 10, it's actually possible to construct
a set of 2 MOLS easily. Let me try to construct them explicitly.

Cyclic square L0: L0[i][j] = (i + j) % 10

For L1, we need it to be a Latin square orthogonal to L0. One known construction:
L1[i][j] = (i + 2*j) % 10? Let me check if this is Latin:
Row i: {i + 2j mod 10 : j=0..9} = {2j mod 10 + i} - but 2 and 10 are not coprime,
so {2j mod 10} = {0,2,4,6,8,0,2,4,6,8} - only 5 distinct values! Not Latin.

So we need a multiplier coprime to 10: 1, 3, 7, 9.

L1[i][j] = (i + 3*j) % 10: Row i has {3j mod 10 + i} = all 10 values since gcd(3,10)=1. Good.
Column j: {i + 3j mod 10} = all 10 values. Good. So L1 is Latin.

Orthogonality of L0 and L1: pairs (L0[i][j], L1[i][j]) = ((i+j)%10, (i+3j)%10).
Are all 100 pairs distinct? Suppose (i+j, i+3j) = (i'+j', i'+3j') mod 10.
Then j-j' = 2(j'-j) mod 10, so 3(j-j') = 0 mod 10, so j=j' mod 10 (since gcd(3,10)=1).
Then i=i'. So yes, orthogonal!

L2[i][j] = (i + 7*j) % 10: similarly Latin and orthogonal to L0.
Check L1 and L2: (i+3j, i+7j). If equal, then 4(j-j')=0 mod 10, so 2(j-j')=0 mod 5,
j-j'=0 mod 5. So j-j' ∈ {0,5}. If j-j'=5, then i+3j vs i+3(j-5)=i+3j-15=i+3j-5 mod 10.
Not equal unless 0=5 mod 10. Wait, let me redo.

If (i+3j)%10 = (i'+3j')%10 and (i+7j)%10 = (i'+7j')%10, then:
3(j-j') = 0 mod 10 from first pair diff? No.

Let me set up properly. We need: i+j ≡ i'+j' (mod 10) and i+3j ≡ i'+3j' (mod 10) for L0,L1.
Subtract: 2j ≡ 2j' (mod 10), so j ≡ j' (mod 5). So j-j' ∈ {0,5}.
If j-j'=0, then i=i'. If j-j'=5, then i+5 ≡ i' (mod 10), so i' = i+5.
Check L1: i+3j vs i'+3j' = i+5+3(j-5) = i+5+3j-15 = i+3j-10 ≡ i+3j (mod 10). 
So (i,j) and (i+5, j-5) give the same pair! So L0 and L1 are NOT orthogonal.

My mistake. Let me reconsider. For cyclic squares L_a[i][j] = (i + a*j) % n, two such
squares L_a and L_b are orthogonal iff gcd(a-b, n) = 1. Here n=10, so we need
a-b coprime to 10. Possible differences: ±1, ±3, ±7, ±9.

So from L0 (a=1), we can have L3 (a=3, diff=2, gcd(2,10)=2≠1) - NOT orthogonal.
L3 with L0: diff = 2, gcd=2, not orthogonal. 

We need a such that gcd(a-1, 10) = 1. So a-1 ∈ {1,3,7,9}, a ∈ {2,4,8,0}.
a=0: L[i][j] = i, not a valid Latin square in the usual sense (each row is constant).
Actually L[i][j] = i is not Latin because column j is all i values... wait, column j
has values 0,1,...,9, so it is a permutation. Row i is all i's, not a permutation. So not Latin.

a=2: L[i][j] = (i+2j)%10. Row i: {i+2j mod 10} = only 5 distinct values. Not Latin.
a=4: similarly not Latin.
a=8: similarly not Latin.

So no cyclic square other than L0 itself is orthogonal to L0! The cyclic group Z_10
does not yield MOLS.

This is the fundamental issue: 10 is not a prime power, so the standard construction
fails. We need a different approach.

Let me use a computational search. I'll start with the cyclic square and use
backtracking to find orthogonal squares, with good heuristics.
"""
import time
import random
from collections import deque

N = 10

def _cyclic():
    return [[(i + j) % N for j in range(N)] for i in range(N)]

def _verify(S):
    for sq in S:
        if not isinstance(sq, (list, tuple)):
            return False
        if len(sq) != N:
            return False
        for row in sq:
            if not isinstance(row, (list, tuple)):
                return False
            if len(row) != N:
                return False
            for v in row:
                if type(v) is not int:
                    return False
                if not (0 <= v <= N - 1):
                    return False
        for i in range(N):
            if sorted(sq[i]) != list(range(N)):
                return False
        for j in range(N):
            col = [sq[i][j] for i in range(N)]
            if sorted(col) != list(range(N)):
                return False
    m = len(S)
    for a in range(m):
        for b in range(a + 1, m):
            pairs = set()
            for i in range(N):
                for j in range(N):
                    pairs.add((S[a][i][j], S[b][i][j]))
            if len(pairs) != N * N:
                return False
    return True

def _find_orthogonal(S, time_limit, t_start, max_nodes=2000000):
    """Backtrack a new Latin square orthogonal to every square in S."""
    K = len(S)
    row_used = [set() for _ in range(N)]
    col_used = [set() for _ in range(N)]
    pair_used = [set() for _ in range(K)]
    grid = [[-1] * N for _ in range(N)]
    state = {"count": 0, "found": None}

    # Precompute pair constraints as a 3D boolean array for speed
    # pair_forbidden[k][i][j][s] = True if (S[k][i][j], s) is already used
    pair_forbidden = [[[[False] * N for _ in range(N)] for _ in range(N)] for _ in range(K)]
    for k in range(K):
        for i in range(N):
            for j in range(N):
                for s in range(N):
                    if (S[k][i][j], s) in pair_used[k]:
                        pair_forbidden[k][i][j][s] = True

    def get_candidates(i, j):
        ru = row_used[i]
        cu = col_used[j]
        res = []
        for s in range(N):
            if s in ru or s in cu:
                continue
            ok = True
            for k in range(K):
                if pair_forbidden[k][i][j][s]:
                    ok = False
                    break
            if ok:
                res.append(s)
        return res

    def solve(idx):
        if state["found"] is not None:
            return
        if state["count"] > max_nodes or (time.time() - t_start) > time_limit:
            return
        if idx == N * N:
            state["found"] = [row[:] for row in grid]
            return
        i = idx // N
        j = idx % N
        cands = get_candidates(i, j)
        random.shuffle(cands)
        for s in cands:
            grid[i][j] = s
            row_used[i].add(s)
            col_used[j].add(s)
            for k in range(K):
                pair_forbidden[k][i][j][s] = True
            state["count"] += 1
            solve(idx + 1)
            if state["found"] is not None:
                return
            grid[i][j] = -1
            row_used[i].remove(s)
            col_used[j].remove(s)
            for k in range(K):
                pair_forbidden[k][i][j][s] = False

    solve(0)
    return state["found"]

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    random.seed()

    S = [_cyclic()]

    # Try to add squares one by one
    while time.time() < deadline and len(S) < 9:
        remaining = deadline - time.time()
        if remaining <= 0.1:
            break
        tl = min(remaining * 0.9, 5.0)
        newq = _find_orthogonal(S, tl, time.time())
        if newq is not None:
            S.append(newq)
        else:
            break

    if _verify(S):
        return S
    for k in range(len(S), 0, -1):
        sub = S[:k]
        if _verify(sub):
            return sub
    return [_cyclic()]
