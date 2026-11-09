import time

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # Safety margin before 90s kill

    p = 1009
    N = 1008  # residues 1..1008

    # Precompute modular inverses: inv[x] = x^{-1} mod p for x in 1..N
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, p - 2, p)

    # Sliding window: O(N) primary solver
    # Maintain [s, e] such that no pair (x, y) in the window satisfies x*y ≡ 1 (mod p)
    best_len = 0
    best_start = 1
    s = 1

    for e in range(1, N + 1):
        partner = inv[e]
        if partner == e:
            # Self-inverse: e cannot be in any valid interval
            s = e + 1
        else:
            # If partner is in the current window [s, e-1], advance s past it
            if s <= partner < e:
                s = partner + 1

        cur_len = e - s + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = s

        if time.time() - t0 > DEADLINE:
            return [best_start, best_len]

    # O(N^2) verification: independently confirm the global optimum
    # For each start i, find the maximum valid end j such that [i, j] contains
    # no inverse pair. An interval [i, j] is invalid if there exists some k in [i, j]
    # such that inv[k] is also in [i, j] (including k itself for self-inverses).
    #
    # Efficient check: for a fixed start i, as we extend j from i upward,
    # the interval becomes invalid at the first j where inv[j] >= i and inv[j] <= j
    # (i.e., inv[j] is within [i, j]). Since inv[j] != j for non-self-inverse,
    # this means inv[j] >= i. For self-inverse, inv[j] == j which is always in [i, j].
    #
    # Actually, the correct condition: interval [i, j] is valid iff for ALL k in [i, j],
    # inv[k] is NOT in [i, j]. Since inv is an involution, this is equivalent to:
    # for ALL k in [i, j], inv[k] < i OR inv[k] > j.
    #
    # For a fixed i, as j increases, the constraint tightens. The interval [i, j]
    # becomes invalid when we first encounter a k in [i, j] with i <= inv[k] <= j.
    #
    # Precompute: for each k, the "conflict" means inv[k] and k form a pair.
    # For a fixed i, the maximum valid j is the smallest value such that
    # no pair (k, inv[k]) with both elements in [i, j] exists, minus 1.
    #
    # Simpler: for each i, scan j from i to N. Track if valid.
    # An element j breaks validity if inv[j] is in [i, j] (and inv[j] != j handled by self-inverse).
    # But we also need to check: when we add j, does inv[j] fall in [i, j]?
    # If inv[j] >= i and inv[j] <= j, then adding j creates a conflict (unless inv[j]==j, self-inverse).
    # Wait - we also need to check that all previously added elements don't conflict with j.
    # But since inv is an involution, if inv[j] is in [i, j-1], that's the only new conflict.
    # And if inv[j] > j, then inv[j] isn't in [i, j] yet, but it will be when we reach inv[j].
    # At that point, j will be in [i, inv[j]], so the conflict is detected then.
    #
    # So: for fixed i, interval [i, j] is valid iff for all k in [i, j], inv[k] < i or inv[k] > j.
    # Equivalently: no k in [i, j] has i <= inv[k] <= j.
    #
    # For fixed i, the maximum valid end is: min over all pairs {a,b} with a<b, a in [i, N], b in [i, N]
    # of (min(a,b) - 1) where both a and b >= i... 
    #
    # Actually simplest: for each i, find the minimum value of max(a, b) over all inverse pairs {a,b}
    # where both a >= i and b >= i. Then the maximum valid end for start i is that minimum - 1.
    # If no such pair exists, the max valid end is N.

    # Precompute all inverse pairs (a, b) with a < b, a*b ≡ 1 mod p
    # Also handle self-inverse: {1, 1} and {1008, 1008}
    # For self-inverse at position x: any interval containing x is invalid.
    # So for start i <= x, the max valid end is x - 1.
    # For start i > x, x doesn't matter.

    # Collect all "obstruction" pairs: for each k in 1..N, the pair is (min(k, inv[k]), max(k, inv[k]))
    # For self-inverse k: pair is (k, k)
    # An interval [i, j] is valid iff no obstruction pair (a, b) has i <= a and b <= j.
    # (Since a <= b always, this means both a and b are in [i, j].)

    # For each start i, the max valid end = min over all obstruction pairs (a,b) with a >= i of (b - 1),
    # or N if no such pair exists.

    # Precompute: for each possible start i, what's the min b among all pairs with a >= i?
    # Let min_b_for_start[i] = min{b : (a,b) is an obstruction pair, a >= i}
    # Then max_valid_end[i] = min_b_for_start[i] - 1, or N if min_b_for_start[i] is infinity.

    # Build list of obstruction pairs
    # For k in 1..N: pair = (min(k, inv[k]), max(k, inv[k]))
    # But we need to be careful: the pair (a,b) means both a and b must not be simultaneously in the interval.
    # For self-inverse, a = b = k.

    # For each a (the smaller element of the pair), record the corresponding b.
    # Multiple pairs can have the same a? No, each element has exactly one inverse.
    # So for each a in 1..N, there's exactly one partner b = inv[a]. The pair is (min(a,b), max(a,b)).
    # Let's define: for each a, partner_b = inv[a]. The obstruction pair is (min(a, partner_b), max(a, partner_b)).
    # For a given start i, we need: no pair (a, b) with a >= i and b <= j.
    # Since a <= b (by construction), this is: no pair with a >= i and b <= j.
    # For fixed i, the tightest constraint is the minimum b among all pairs with a >= i.
    # max_valid_end[i] = (min b over pairs with a >= i) - 1

    # Precompute min_b_from[i] = min{b : a >= i, (a,b) is a pair}
    # Process i from N down to 1:
    #   min_b_from[i] = min(b_i, min_b_from[i+1]) where b_i is the b for the pair with a = i
    #   But wait: for element i, the pair is (min(i, inv[i]), max(i, inv[i])).
    #   If i <= inv[i], then a = i, b = inv[i].
    #   If i > inv[i], then a = inv[i], b = i. In this case, the pair's "a" is inv[i], not i.
    #   So for element i, it contributes to the pair with a = min(i, inv[i]).
    #
    # Let me restructure: for each a in 1..N, what is the b in the pair?
    # The pair containing element a is (a, inv[a]). The smaller is min(a, inv[a]), larger is max(a, inv[a]).
    # For the obstruction: interval [i, j] is invalid if there exists a pair (a, b) with a <= b such that i <= a and b <= j.
    # For fixed i, max valid j = min{b - 1 : a >= i, (a,b) is a pair} (where a is the smaller element).
    #
    # For each element k, its pair has smaller = min(k, inv[k]), larger = max(k, inv[k]).
    # Let's build: for each possible "smaller" value a, what is the corresponding "larger" value b?
    # Since inv is a bijection, each a appears as the smaller in exactly one pair (unless a is self-inverse, where a=b).
    # Actually: for a given a, the pair is (a, inv[a]) if a < inv[a], or (inv[a], a) if inv[a] < a, or (a,a) if a == inv[a].
    # So for each a, the pair where a is the smaller is: b = inv[a] if a <= inv[a], else a is the larger in the pair (inv[a], a).
    #
    # Let me just build an array: for each a in 1..N, pair_b[a] = the b such that (a, b) is the obstruction pair with a being the smaller.
    # If a < inv[a]: pair_b[a] = inv[a]
    # If a == inv[a]: pair_b[a] = a
    # If a > inv[a]: a is NOT the smaller in its pair, so pair_b[a] is undefined (a is the larger).
    #
    # Then for start i, max_valid_end[i] = min{pair_b[a] - 1 : a >= i, pair_b[a] is defined}
    # If no such a exists, max_valid_end[i] = N.

    pair_b = [None] * (N + 1)
    for k in range(1, N + 1):
        a = min(k, inv[k])
        b = max(k, inv[k])
        # a is the smaller element of the pair
        # We only set pair_b[a] if not already set (should be unique)
        if pair_b[a] is None or b < pair_b[a]:
            pair_b[a] = b

    # Now compute min_b_from[i] = min{pair_b[a] : a >= i, pair_b[a] is not None}
    min_b_from = [N + 1] * (N + 2)  # N+1 means "no pair", so max_valid_end = N
    for i in range(N, 0, -1):
        if pair_b[i] is not None:
            min_b_from[i] = min(pair_b[i], min_b_from[i + 1])
        else:
            min_b_from[i] = min_b_from[i + 1]

    # Verify: for each start i, max_valid_end[i] = min_b_from[i] - 1
    verified_best_len = 0
    verified_best_start = 1
    for i in range(1, N + 1):
        max_end = min_b_from[i] - 1
        if max_end < i:
            continue  # empty interval
        length = max_end - i + 1
        if length > verified_best_len:
            verified_best_len = length
            verified_best_start = i

    # The sliding window should agree with the verification
    if verified_best_len > best_len:
        best_len = verified_best_len
        best_start = verified_best_start

    return [best_start, best_len]
