"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill

    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)  # 255 elements

    # Prime factorization for each candidate
    def factorize(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors

    factorizations = [factorize(x) for x in candidates]

    # Precompute conflict pairs using bitmasks for fast membership
    # conflict_pairs[a] = list of (b, c) where b < c, b != a, c != a, and a | candidates[b]*candidates[c]
    conflict_pairs = [[] for _ in range(n)]

    for a in range(n):
        fa = factorizations[a]
        for b in range(n):
            if b == a:
                continue
            fb = factorizations[b]
            for c in range(b + 1, n):
                if c == a:
                    continue
                fc = factorizations[c]
                ok = True
                for p, ea in fa.items():
                    eb = fb.get(p, 0)
                    ec = fc.get(p, 0)
                    if ea > eb + ec:
                        ok = False
                        break
                if ok:
                    conflict_pairs[a].append((b, c))

    # For each a, precompute bitmask of elements that can pair with another to conflict with a
    # Actually, we need: for a to be in violation, there must exist b,c in S\{a} with a|b*c
    # Precompute for each a: conflict_mask[a] as a list of (b,c) pairs is already done.
    # For fast checking, let's precompute for each a a bitmask representation.
    # But pairs are 2D, so let's use a different approach.
    # For each a, and for each b, compute which c values would make a|b*c.
    # Then for a given set S, a is in violation if there exists b in S\{a} such that
    # the set of c in S\{a,b} intersects with the "bad c" set for (a,b).

    # Precompute: bad_c[a][b] = bitmask of c values such that a | candidates[b]*candidates[c]
    # c ranges over 0..n-1, c != a, c != b
    bad_c = [[0] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            mask = 0
            for (bb, cc) in conflict_pairs[a]:
                if bb == b:
                    mask |= (1 << cc)
                elif cc == b:
                    mask |= (1 << bb)
            # Remove a and b from mask
            mask &= ~(1 << a)
            mask &= ~(1 << b)
            bad_c[a][b] = mask

    # Greedy removal using bitmasks
    # S_mask: bitmask of elements currently in S
    S_mask = (1 << n) - 1

    def count_conflicts(a, smask):
        """Count how many conflict pairs for a are fully within smask (excluding a)."""
        count = 0
        for (b, c) in conflict_pairs[a]:
            if (smask >> b) & 1 and (smask >> c) & 1:
                count += 1
        return count

    def has_any_conflict(a, smask):
        """Check if a has any conflict pair within smask (excluding a)."""
        for (b, c) in conflict_pairs[a]:
            if (smask >> b) & 1 and (smask >> c) & 1:
                return True
        return False

    # Iteratively remove the most conflicted element
    changed = True
    while changed:
        changed = False
        worst = -1
        worst_count = 0
        for a in range(n):
            if not ((S_mask >> a) & 1):
                continue
            count = 0
            for (b, c) in conflict_pairs[a]:
                if (S_mask >> b) & 1 and (S_mask >> c) & 1:
                    count += 1
                    if count >= worst_count:
                        break
            if count > worst_count:
                worst_count = count
                worst = a
        if worst_count > 0 and worst >= 0:
            S_mask &= ~(1 << worst)
            changed = True

    # Convert to list for easier manipulation
    S_list = [i for i in range(n) if (S_mask >> i) & 1]
    S_set = set(S_list)

    # Local search: try to add back removed elements
    removed = [i for i in range(n) if i not in S_set]

    def is_valid_set(smask):
        """Check if the set represented by smask has no violations."""
        for a in range(n):
            if not ((smask >> a) & 1):
                continue
            for (b, c) in conflict_pairs[a]:
                if (smask >> b) & 1 and (smask >> c) & 1:
                    return False
        return True

    improved = True
    while improved:
        improved = False
        for r in removed[:]:
            new_mask = S_mask | (1 << r)
            if is_valid_set(new_mask):
                S_mask = new_mask
                S_set.add(r)
                removed.remove(r)
                improved = True

    # Swap local search: remove one from S, add one from removed
    for r in removed[:]:
        for s in S_list[:]:
            if r == s:
                continue
            new_mask = (S_mask & ~(1 << s)) | (1 << r)
            if is_valid_set(new_mask):
                S_mask = new_mask
                S_set.discard(s)
                S_set.add(r)
                S_list.remove(s)
                S_list.append(r)
                removed.remove(r)
                break

    # Multiple passes of swap local search
    for _ in range(5):
        found = False
        for r in removed[:]:
            for s in S_list[:]:
                if r == s:
                    continue
                new_mask = (S_mask & ~(1 << s)) | (1 << r)
                if is_valid_set(new_mask):
                    S_mask = new_mask
                    S_set.discard(s)
                    S_set.add(r)
                    S_list.remove(s)
                    S_list.append(r)
                    removed.remove(r)
                    found = True
                    break
            if found:
                break
        if not found:
            break

    # Convert back to actual values
    result = [candidates[i] for i in sorted(S_set)]

    # Verify
    def verify(result_list):
        for i in range(len(result_list)):
            for j in range(len(result_list)):
                if i == j:
                    continue
                for k in range(len(result_list)):
                    if i == k or j == k:
                        continue
                    a, b, c = result_list[i], result_list[j], result_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    if not verify(result):
        # Fallback to primes
        result = [a for a in range(2, N+1) if all(a % d for d in range(2, isqrt(a)+1))]

    if time.time() - start_time > DEADLINE:
        pass

    return result
# EVOLVE-BLOCK-END
