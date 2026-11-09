import math
import itertools
import time

# EVOLVE-BLOCK-START
def _coprime_pairwise(vals):
    """True if all distinct vals are pairwise coprime."""
    v = sorted(set(vals))
    n = len(v)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(v[i], v[j]) != 1:
                return False
    return True

def _valid(A, B):
    """Full validity check for two 3-element sets."""
    A = list(A)
    B = list(B)
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    sums = [a + b for a in A for b in B]
    if not _coprime_pairwise(sums):
        return False
    return True

def construct():
    start = time.time()
    DEADLINE = 60.0
    best = ([1, 7, 13], [4, 10, 16])
    best_score = len({a + b for a in best[0] for b in best[1]})

    # Enumerate A (sorted triples) and B (sorted triples) over [1,128].
    # Use incremental coprimality pruning on the partial 3x3 grid.
    # To keep runtime bounded, we cap the search and rely on deadline guard.

    # Precompute: for speed, iterate A triples, then B triples, but prune
    # early. We'll use a smarter ordering: pick A, then for each B compute sums.

    # Full enumeration is C(128,3)^2 ~ 1e11, too big. Instead we do a
    # structured search: fix A, then search B with pruning.
    # We'll iterate over A triples (sorted) and for each, search B triples.

    # To make it tractable, we limit A to a reasonable subset first, then
    # extend. Actually, let's do a smarter approach:
    # Enumerate all A triples, and for each, enumerate B triples with
    # early termination on coprimality failure.

    # But even C(128,3) = 341,376 for A, times C(128,3) for B = ~1.17e11.
    # We need heavy pruning.

    # Strategy: For a fixed A, the sums with B are {a1+b, a2+b, a3+b} for each b in B.
    # All 9 sums must be pairwise coprime. This is very restrictive.

    # Let's do: for each A triple, for each B triple, check. But prune B
    # incrementally: pick b1, check {a1+b1, a2+b1, a3+b1} pairwise coprime
    # and coprime to each other. Then pick b2, check new sums against all.

    # Actually, let's just do a bounded random + local search hybrid with
    # the CSP structure, since full enumeration is infeasible in 60s.

    # Better: use the CSP backtracking on the 9 grid variables with the
    # rank-1 constraint. Variables: a1,a2,a3,b1,b2,b3.
    # Constraint: all 9 sums pairwise coprime.
    # We'll backtrack on a1,a2,a3,b1,b2,b3 in order, checking partial sums.

    # Order: a1, a2, a3, b1, b2, b3.
    # After assigning a1,a2,a3,b1: sums {a1+b1, a2+b1, a3+b1} must be pairwise coprime.
    # After b2: add {a1+b2, a2+b2, a3+b2}, check all 6 pairwise.
    # After b3: add {a1+b3, a2+b3, a3+b3}, check all 9 pairwise.

    # This is still large. Let's bound it with a time check and return best.

    # We'll use a randomized restart approach with the CSP structure.
    import random
    rng = random.Random(42)

    # Precompute coprime pairs for fast lookup? Not needed, gcd is fast.

    # Let's do a structured search: iterate a1 from 1..128, a2 from a1+1..128,
    # a3 from a2+1..128, then for b1 from 1..128, check partial sums.
    # This is 341376 * 128 = ~43M for the a's and b1, which might be tight.
    # Let's add the b1 check to prune heavily.

    # Actually, let's just do a time-bounded search with random sampling
    # of A triples and B triples, plus local improvement.

    # Simpler and effective: enumerate A triples (sorted), and for each,
    # enumerate B triples (sorted) with early pruning. Use a time check
    # every so often.

    # Let's estimate: for each A, the number of B triples that pass the
    # first b check is small. So the total work is manageable.

    # Let's implement with a time guard.

    a_range = range(1, 129)
    count = 0

    for a1 in a_range:
        if time.time() > DEADLINE - 5:
            break
        for a2 in range(a1 + 1, 129):
            if time.time() > DEADLINE - 5:
                break
            for a3 in range(a2 + 1, 129):
                # For this A, search B triples with pruning
                for b1 in a_range:
                    # Check {a1+b1, a2+b1, a3+b1} pairwise coprime
                    s1 = a1 + b1
                    s2 = a2 + b1
                    s3 = a3 + b1
                    if s1 > 256 or s2 > 256 or s3 > 256:
                        continue
                    if math.gcd(s1, s2) != 1 or math.gcd(s1, s3) != 1 or math.gcd(s2, s3) != 1:
                        continue
                    # Now search b2 > b1
                    for b2 in range(b1 + 1, 129):
                        s4 = a1 + b2
                        s5 = a2 + b2
                        s6 = a3 + b2
                        if s4 > 256 or s5 > 256 or s6 > 256:
                            continue
                        # Check new sums against each other and existing
                        if math.gcd(s4, s5) != 1 or math.gcd(s4, s6) != 1 or math.gcd(s5, s6) != 1:
                            continue
                        for s in (s4, s5, s6):
                            if math.gcd(s, s1) != 1 or math.gcd(s, s2) != 1 or math.gcd(s, s3) != 1:
                                break
                        else:
                            # Check b3 > b2
                            for b3 in range(b2 + 1, 129):
                                s7 = a1 + b3
                                s8 = a2 + b3
                                s9 = a3 + b3
                                if s7 > 256 or s8 > 256 or s9 > 256:
                                    continue
                                if math.gcd(s7, s8) != 1 or math.gcd(s7, s9) != 1 or math.gcd(s8, s9) != 1:
                                    continue
                                ok = True
                                for s in (s7, s8, s9):
                                    for t in (s1, s2, s3, s4, s5, s6):
                                        if math.gcd(s, t) != 1:
                                            ok = False
                                            break
                                    if not ok:
                                        break
                                if ok:
                                    A = [a1, a2, a3]
                                    B = [b1, b2, b3]
                                    score = len({x + y for x in A for y in B})
                                    if score > best_score:
                                        best_score = score
                                        best = (A, B)
                                if time.time() > DEADLINE - 5:
                                    break
                        if time.time() > DEADLINE - 5:
                            break
                    if time.time() > DEADLINE - 5:
                        break
            if time.time() > DEADLINE - 5:
                break
        if time.time() > DEADLINE - 5:
            break

    return [list(best[0]), list(best[1])]
# EVOLVE-BLOCK-END
