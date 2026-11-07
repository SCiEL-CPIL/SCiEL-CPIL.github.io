import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n

    # Precompute comparability: comp_mask[a] is a Python int bitset where bit b is set
    # iff a and b are comparable (one is subset of the other).
    # Two masks a, b are comparable iff (a & b) == a or (a & b) == b.
    # We build this efficiently: for each a, the comparable set is
    #   {b : b subseteq a} ∪ {b : a subseteq b}
    # Using subset enumeration.

    # Actually, let's build a 2D boolean list or use a different approach.
    # For speed in the inner loop, we want: given a set of "active" masks (family),
    # and a candidate y, compute how many family members are comparable to y.
    #
    # Approach: precompute for each mask a, the list of masks comparable to a.
    # Then comp_in_F[y] = sum over z in family of [z in comp_list[y]]
    #
    # Even better: precompute comp_list[a] as a sorted list.
    # Then for a given family, comp_in_F[y] = number of z in family that are in comp_list[y].
    #
    # But we need to update comp_in_F when we swap. Let's think differently.
    #
    # Key insight: comp_in_F[y] = number of z in family such that z~y.
    # When we remove x and add w:
    #   new_comp_in_F[y] = comp_in_F[y] - [x~y] + [w~y]
    #
    # So we need to quickly check [x~y] and [w~y] for all y.
    #
    # Let's precompute for each mask a, a bitset (Python int) where bit y is set iff a~y.
    # Then [a~y] = (comp_bitset[a] >> y) & 1.
    #
    # But iterating over all y to update comp_in_F is O(N) per swap.
    #
    # Alternative: don't maintain comp_in_F. Instead, for each candidate position i,
    # compute the gain of replacing family[i] with y by directly counting.
    #
    # Gain of replacing x=family[i] with y:
    #   new_pairs = (pairs involving y with F\{x}) - (pairs involving x with F\{x})
    #   = (number of z in F\{x} with z~y) - (number of z in F\{x} with z~x)
    #
    # We can precompute for the current family F:
    #   deg[x] = number of z in F\{x} with z~x  (for x in F)
    #
    # Then gain = (number of z in F\{x} with z~y) - deg[x]
    #
    # To compute "number of z in F\{x} with z~y" quickly:
    #   = (number of z in F with z~y) - [x~y]
    #
    # So if we maintain for each y: count_comp[y] = number of z in F with z~y,
    # then gain = count_comp[y] - [x~y] - deg[x]
    #
    # And deg[x] = count_comp[x] - 1 (since x is in F and x~x is true... wait, is x~x?)
    #
    # Is a mask comparable to itself? The problem says "strict subset", so a set is NOT
    # a strict subset of itself. So comparability for the purpose of counting pairs:
    # two distinct sets A, B are comparable if A ⊂ B or B ⊂ A.
    #
    # So for distinct masks a ≠ b, they are comparable iff (a&b)==a or (a&b)==b.
    # For a == b, they are NOT comparable (since we need strict subset).
    #
    # Let me redefine: comp(a,b) for a≠b is (a&b)==a or (a&b)==b.
    #
    # So count_comp[y] = number of z in F, z≠y, such that z~y.
    # If y is in F, then deg[y] = count_comp[y].
    # If y is not in F, then "number of z in F with z~y" = count_comp[y] (since y not in F, no self-issue).
    #
    # Wait, let me be more careful. Let's define:
    #   For any mask y: C[y] = number of z in F such that z ≠ y and z~y.
    #
    # If y ∈ F: the number of other family members comparable to y is C[y].
    # If y ∉ F: the number of family members comparable to y is C[y].
    #
    # When we replace x (which is in F) with w (which is not in F):
    #   New score change = (number of z in F\{x} with z~w) - (number of z in F\{x} with z~x)
    #   = (C[w] - [x~w]) - (C[x] - [x~x])
    #   But x~x is false (strict subset), so [x~x] = 0.
    #   Also, C[w] counts z in F with z≠w and z~w. Since w∉F, all z in F have z≠w, so
    #   C[w] = number of z in F with z~w.
    #   We want number of z in F\{x} with z~w = C[w] - [x~w].
    #   And number of z in F\{x} with z~x = C[x] (since C[x] already excludes x itself).
    #
    # So gain = (C[w] - [x~w]) - C[x]
    #
    # To find the best w not in F, we want to maximize C[w] - [x~w].
    #
    # Now, maintaining C[y] for all y:
    #   Initially, C[y] = number of z in F with z~y.
    #   When we remove x and add w:
    #     For each y: C_new[y] = C[y] - [x~y] + [w~y]
    #
    # This is O(N) per swap. With N=4096, that's manageable.
    #
    # Let's precompute comp_bitset[a] = bitset where bit y is set iff a~y (and a≠y).
    # Then [a~y] = (comp_bitset[a] >> y) & 1.

    # Build comp_bitset
    comp_bitset = [0] * N
    for a in range(N):
        bits = 0
        for b in range(N):
            if a == b:
                continue
            if (a & b) == a or (a & b) == b:
                bits |= (1 << b)
        comp_bitset[a] = bits

    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                a, b = family[i], family[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count

    def local_search(family, deadline):
        # family is a list of masks
        # C[y] = number of z in family such that z != y and z~y
        C = [0] * N
        for y in range(N):
            c = 0
            for z in family:
                if z != y and ((z & y) == z or (z & y) == y):
                    c += 1
            C[y] = c

        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        fam_set = set(family)

        no_improve_count = 0
        max_no_improve = 30

        while time.time() < deadline - 5:
            if no_improve_count >= max_no_improve:
                break

            improved = False
            for i in range(m):
                x = family[i]
                # C[x] is the number of other family members comparable to x
                cx = C[x]

                # Find best w not in fam_set maximizing C[w] - [x~w]
                best_w = -1
                best_gain = -1
                # Iterate through all w not in fam_set
                for w in range(N):
                    if w in fam_set:
                        continue
                    gain = C[w]
                    if (comp_bitset[x] >> w) & 1:
                        gain -= 1
                    if gain > best_gain:
                        best_gain = gain
                        best_w = w

                if best_w == -1:
                    continue

                if best_gain > cx:
                    # Perform swap: remove x, add best_w
                    # Update C for all y: C[y] += [w~y] - [x~y]
                    cw = comp_bitset[best_w]
                    cx_bits = comp_bitset[x]
                    diff = cw ^ cx_bits  # bits where they differ
                    # For bits set in diff:
                    # If bit is set in cw but not cx_bits: C[y] += 1
                    # If bit is set in cx_bits but not cw: C[y] -= 1
                    # We need to iterate over set bits in diff.
                    d = diff
                    while d:
                        lb = d & (-d)
                        y = lb.bit_length() - 1
                        if (cw >> y) & 1:
                            C[y] += 1
                        else:
                            C[y] -= 1
                        d ^= lb

                    # Update family
                    fam_set.remove(x)
                    fam_set.add(best_w)
                    family[i] = best_w
                    current_score += best_gain - cx

                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]

                    improved = True
                    no_improve_count = 0

            if not improved:
                no_improve_count += 1

        return best_family, best_score

    best_overall = None
    best_score_overall = 0

    # Multiple restarts with different random seeds
    for restart in range(50):
        if time.time() > deadline - 10:
            break

        random.seed(42 + restart)
        family = random.sample(range(N), m)
        fam, score = local_search(family, deadline)

        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam

    # Chain-based initialization
    if time.time() < deadline - 15:
        chain = [0]
        cur = 0
        for b in range(n):
            cur |= (1 << b)
            chain.append(cur)
        remaining = [x for x in range(N) if x not in set(chain)]
        random.seed(999)
        extra = random.sample(remaining, m - len(chain))
        family = chain + extra
        fam, score = local_search(family, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam

    # Additional structured initializations
    if time.time() < deadline - 15:
        # Initialize with sets of size 0, 1, 2, 3
        from itertools import combinations
        small_sets = []
        for sz in range(4):
            for comb in combinations(range(n), sz):
                mask = 0
                for b in comb:
                    mask |= (1 << b)
                small_sets.append(mask)
        if len(small_sets) >= m:
            random.seed(12345)
            family = random.sample(small_sets, m)
            fam, score = local_search(family, deadline)
            if score > best_score_overall:
                best_score_overall = score
                best_overall = fam

    if best_overall is None:
        best_overall = list(range(m))

    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])

    return result
