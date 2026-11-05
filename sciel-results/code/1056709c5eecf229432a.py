import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse elements (x^2 == 1 mod 1009): {1, 1008}
    self_inv = set()
    for x in range(1, MOD):
        if x * x % MOD == 1:
            self_inv.add(x)

    # Sliding-window minimum approach:
    # For each start S (from 2 to 1007, excluding self-inverses),
    # the max valid E is the smallest x >= S such that inv[x] is in [S, x].
    # Equivalently: E_max(S) = min{x >= S : inv[x] >= S and inv[x] <= x} - 1
    # But we also need E_max(S) < 1008 (self-inverse) and E_max(S) >= S.
    #
    # More precisely: for interval [S, E] to be valid, for every x in [S, E],
    # inv[x] must NOT be in [S, E]. Since inv[x] != x for non-self-inverse x,
    # inv[x] is either < S or > E. The constraint is:
    #   For all x in [S, E]: inv[x] < S OR inv[x] > E
    # The binding constraint as E grows: E must be < min{inv[x] : x in [S,E], inv[x] >= S}
    #
    # So E_max(S) = min{ inv[x] : x >= S, inv[x] >= S, inv[x] <= x } - 1
    # Wait, that's not quite right either. Let me think again.
    #
    # For a fixed S, as E increases from S-1 upward:
    # - At each step adding element E, we need inv[E] not in [S, E-1]
    # - inv[E] < S is fine (no conflict)
    # - inv[E] >= S means inv[E] is in [S, E-1] if inv[E] <= E-1, i.e., inv[E] < E
    #   OR inv[E] could be > E (no conflict yet, but future elements might conflict)
    #
    # Actually the cleanest formulation: interval [S,E] is valid iff
    # for all x in [S,E], inv[x] < S or inv[x] > E.
    #
    # For fixed S, define f_S(x) = inv[x] for x >= S.
    # E is valid iff for all x in [S, E], inv[x] < S or inv[x] > E.
    #
    # The first violation occurs at the smallest x >= S where inv[x] >= S and inv[x] <= x
    # ... no. Let's just say: E_max is the smallest x >= S such that there exists
    # some y in [S, x] with inv[y] in [S, x] and inv[y] != y.
    #
    # Simpler: E_max(S) = min { x >= S : inv[x] >= S and inv[x] <= x } - 1
    # But wait, if inv[x] > x, then inv[x] is not in [S, x] yet, so x itself doesn't
    # cause a violation. The violation would come later when we reach inv[x].
    #
    # Let me reconsider. For fixed S, the interval [S, E] is invalid iff there exists
    # x in [S, E] with inv[x] in [S, E]. Since inv is a permutation, this means
    # the pair {x, inv[x]} is fully contained in [S, E].
    #
    # So E_max(S) = min { max(x, inv[x]) : x >= S, x in [S, ...], inv[x] >= S } - 1
    # i.e., the smallest upper endpoint of any pair that has both elements >= S.
    #
    # For each pair {a, b} with a < b (a = min, b = max), it forbids intervals
    # with S <= a and E >= b. So for fixed S, E must be < b for all pairs with a >= S.
    # E_max(S) = min { b : pair {a,b}, a >= S } - 1
    #
    # This is a classic "for each S, find min of b over pairs with a >= S" problem.
    # We can precompute: for each possible a value, what's the min b among pairs with that a?
    # Then E_max(S) = min over a >= S of (min_b_for_a[a]) - 1

    # Build pairs: for each x in [2, 1007] (non-self-inverse), pair x with inv[x]
    # Each pair {x, inv[x]} where x != inv[x]. To avoid duplicates, only count when x < inv[x].
    # For each pair, a = min(x, inv[x]), b = max(x, inv[x])
    # For fixed S, E_max = min{b : a >= S} - 1, also capped by 1007 (can't include 1008)

    # Precompute: for each a in [2, 1007], min_b[a] = smallest b such that {a,b} is a pair
    # Actually, for a given a, there might be multiple b values? No!
    # For a given a, inv[a] is unique. So the pair containing a is {a, inv[a]}.
    # If a < inv[a], then b = inv[a]. If a > inv[a], then a is the larger element,
    # and the pair is {inv[a], a} with the "a" in our notation being inv[a].
    #
    # So for each x in [2, 1007]:
    #   a = min(x, inv[x]), b = max(x, inv[x])
    #   This pair contributes: for any S <= a, E must be < b.
    #
    # For fixed S: E_max(S) = min{ b : x in [2,1007], min(x,inv[x]) >= S } - 1
    # Also E_max(S) <= 1007 (to exclude self-inverse 1008)
    # And E_max(S) >= S for a valid non-empty interval.

    # Precompute min_b_for_start[S] = min{ b : min(x, inv[x]) >= S } for S in [2, 1007]
    # We can do this with a suffix minimum.

    # First, for each a in [2, 1007], compute the b associated with pairs where min(x,inv[x]) = a
    # Actually, for each x in [2, 1007], let a = min(x, inv[x]), b = max(x, inv[x])
    # For a given a, there's exactly one x that gives this a? No, potentially two:
    # x such that min(x, inv[x]) = a. This means either x = a and inv[a] > a,
    # or inv[x] = a and x > a, i.e., x = inv[a] and inv[a] > a.
    # Wait, if min(x, inv[x]) = a, then either x = a (and inv[a] >= a) or inv[x] = a (and x >= a).
    # If x = a: pair is {a, inv[a]}, b = inv[a] (assuming inv[a] != a, which is true for a not self-inverse)
    # If inv[x] = a: then x = inv[a], pair is {inv[a], a}, b = inv[a] (assuming inv[a] > a)
    # So actually for a given a (not self-inverse), the pair is always {a, inv[a]} with b = max(a, inv[a]).
    # There's only ONE pair per element a. So for each a in [2,1007] \ self_inv:
    #   b_a = max(a, inv[a])
    #   This pair forbids E >= b_a for any S <= a.

    # So: for fixed S, E_max(S) = min{ b_a : a in [S, 1007], a not self-inverse } - 1
    # And also E_max(S) <= 1007.

    # Precompute b_a for each a
    b_a = [0] * (MOD + 1)
    for a in range(2, MOD):  # 2 to 1008
        if a in self_inv:
            b_a[a] = MOD  # effectively infinite, but a=1008 is self-inverse
            continue
        b_a[a] = max(a, inv[a])

    # Suffix minimum: suff_min[S] = min{ b_a[a] : a >= S }
    suff_min = [0] * (MOD + 2)
    suff_min[MOD] = MOD  # sentinel
    for a in range(MOD - 1, 1, -1):
        suff_min[a] = min(b_a[a], suff_min[a + 1])

    # Now for each S in [2, 1007]:
    #   E_max(S) = min(suff_min[S], 1007) - 1  (cap at 1007 to exclude 1008)
    #   But also need E_max(S) >= S for non-empty interval
    #   And S must not be self-inverse (S != 1, S != 1008; S is in [2,1007] so S != 1, and S != 1008)
    #   Wait, S can be 1007 max in our range. 1008 is self-inverse so excluded.

    best_start = 2
    best_len = 0

    for S in range(2, MOD):  # S from 2 to 1007
        if S in self_inv:
            continue
        e_max = min(suff_min[S], 1007) - 1
        if e_max < S:
            continue
        current_len = e_max - S + 1
        if current_len > best_len:
            best_len = current_len
            best_start = S

        # Deadline guard
        if time.time() > DEADLINE:
            break

    # Verify the result
    S = best_start
    E = S + best_len - 1
    valid = True
    for x in range(S, E + 1):
        if x in self_inv:
            valid = False
            break
        if S <= inv[x] <= E:
            valid = False
            break

    if not valid:
        # Fallback: brute force for safety
        for S2 in range(2, MOD):
            if S2 in self_inv:
                continue
            E2 = S2 - 1
            while E2 < MOD - 1:
                new_E = E2 + 1
                if new_E in self_inv:
                    break
                partner = inv[new_E]
                if S2 <= partner <= E2:
                    break
                E2 = new_E
            if E2 >= S2:
                cl = E2 - S2 + 1
                if cl > best_len:
                    best_len = cl
                    best_start = S2
                    break
            if time.time() > DEADLINE:
                break

    return [best_start, best_len]
