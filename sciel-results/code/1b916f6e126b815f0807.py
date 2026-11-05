"""Improved program: maximize pairwise-disjoint residue classes with distinct moduli 2..128."""
# EVOLVE-BLOCK-START
import math
import time

_MAX = 128
_DEADLINE = 80.0  # safety margin before 90s kill


def _prime_powers_upto(max_n):
    """Return dict prime -> list of prime powers p^a <= max_n (a>=1)."""
    primes = []
    is_prime = [True] * (max_n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, max_n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, max_n + 1, i):
                is_prime[j] = False
    pp = {}
    for p in primes:
        lst = []
        v = p
        while v <= max_n:
            lst.append(v)
            v *= p
        pp[p] = lst
    return pp


def _greedy():
    """Greedy: process prime-power families by smallest exponent (hardest first).
    For each prime, pick a set of exponents and residues so that within the family
    r_i not equiv r_j mod p^{min(a_i,a_j)}."""
    pp = _prime_powers_upto(_MAX)

    # Build candidate list: (min_exponent_of_family, prime, [powers])
    # We'll decide per prime which powers to take.
    # Strategy: for each prime, consider taking a prefix of powers [p^1, p^2, ..., p^k]
    # and assign residues greedily. More powers = more classes, but harder to satisfy.
    # We'll try for each prime, increasing k, and keep the max feasible set.

    # Group primes by their smallest power's exponent (all are 1 for p^1).
    # Actually all families start at exponent 1. The constraint difficulty grows with k.
    # Process primes in order of increasing p (smaller p => p^1 smaller => more residue options? no).
    # The gcd between two same-prime powers is the smaller power, which is independent of which prime.
    # So primes are independent! We can optimize each prime separately.

    result = []
    for p in sorted(pp.keys()):
        powers = pp[p]  # [p, p^2, p^3, ...]
        # Try to select a subset of these powers (sorted ascending) with residues.
        # Greedy: try to include as many as possible. Start with all, check feasibility.
        # Feasibility: for selected powers q_1 < q_2 < ... < q_k (q_i = p^{a_i}),
        # need r_i not equiv r_j mod q_i for i<j.
        # This is always satisfiable: pick r_1 = 0. For i>1, r_i must avoid r_j mod q_i for all j<i.
        # Number of forbidden classes for r_i is (i-1) (one per j<i, each a single class mod q_i).
        # q_i = p^{a_i} >= p >= 2. As long as q_i > (i-1), we can pick r_i.
        # So we can include powers as long as p^{a_i} >= i (roughly).
        # Let's greedily include powers while p^{a_i} >= i (the index).
        selected = []
        for idx, q in enumerate(powers, start=1):
            if q >= idx:  # enough residues to avoid idx-1 forbidden classes
                selected.append(q)
            else:
                # Could still work if q > idx-1, i.e., q >= idx. Same condition.
                # If q < idx, we might still fit if we skipped some earlier, but keep simple.
                break
        # Now assign residues to selected.
        residues = []
        for i, q in enumerate(selected):
            # forbidden: residues[j] mod q for j < i
            forbidden = set()
            for j in range(i):
                forbidden.add(residues[j] % q)
            r = 0
            while r in forbidden:
                r += 1
            residues.append(r)
        for r, q in zip(residues, selected):
            result.append([r, q])
    return result


def construct():
    start = time.time()
    # Greedy baseline
    best = _greedy()
    # Verify baseline validity
    def valid(lst):
        mods = [m for _, m in lst]
        if len(set(mods)) != len(mods):
            return False
        for m in mods:
            if m < 2 or m > _MAX:
                return False
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                r1, m1 = lst[i]
                r2, m2 = lst[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    if not valid(best):
        # Fallback to seed
        best = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    # Try to improve: for primes where we stopped early due to q < idx,
    # maybe we can still add by re-choosing residues with a smarter assignment.
    # But the constraint is tight; greedy is near-optimal per prime.
    # Attempt a local search: try adding one more power per prime with careful residue.
    # Given time budget, do a limited improvement pass.
    if time.time() - start < _DEADLINE - 5.0:
        # For each prime, try to see if we can add the next power.
        pp = _prime_powers_upto(_MAX)
        # Rebuild per-prime selected from best
        per_prime = {}
        for r, m in best:
            # find prime of m
            p = None
            for cand in pp:
                if m % cand == 0:
                    # check m is power of cand
                    t = m
                    while t % cand == 0:
                        t //= cand
                    if t == 1:
                        p = cand
                        break
            if p is not None:
                per_prime.setdefault(p, []).append((r, m))
        # For each prime, try to extend
        changed = True
        while changed and time.time() - start < _DEADLINE - 5.0:
            changed = False
            for p in list(per_prime.keys()):
                cur = sorted(per_prime[p], key=lambda x: x[1])
                powers = pp[p]
                # next power not in cur
                cur_mods = set(m for _, m in cur)
                next_q = None
                for q in powers:
                    if q not in cur_mods:
                        next_q = q
                        break
                if next_q is None:
                    continue
                # Try to add (r, next_q) for some r
                # It must be disjoint from all existing classes (all primes, not just p)
                added = False
                for r in range(next_q):
                    ok = True
                    for (r2, m2) in best:
                        g = math.gcd(next_q, m2)
                        if r % g == r2 % g:
                            ok = False
                            break
                    if ok:
                        # also need within-prime constraint already covered by above (gcd with same-prime powers is the smaller)
                        new_best = best + [[r, next_q]]
                        if valid(new_best):
                            best = new_best
                            per_prime[p].append((r, next_q))
                            added = True
                            changed = True
                            break
                if added:
                    continue
    return best
# EVOLVE-BLOCK-END
