import time
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

PRIMES = _sieve(1000)
CAP = 100000

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    best_L = 0
    best_classes = []

    def compute_L(covered):
        L = 0
        for i in range(1, CAP + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def make_coverage(prime_res):
        covered = bytearray(CAP + 1)
        for p, r in prime_res.items():
            s = r
            if s == 0:
                s = p
            for k in range(s, CAP + 1, p):
                covered[k] = 1
        return covered

    def build_classes(prime_res):
        return [[p, prime_res[p]] for p in sorted(prime_res.keys())]

    def greedy_init(window, eff, deadline):
        nonlocal best_L, best_classes
        covered = bytearray(CAP + 1)
        used = set()
        budget = Fraction(1)
        prime_res = {}

        n = 1
        while n <= CAP:
            if time.time() > deadline:
                break
            if covered[n]:
                n += 1
                continue

            best_p = None
            best_r = 0
            best_score = -1

            hi = min(n + window, CAP)

            for p in PRIMES:
                if p in used:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue

                r = n % p
                cnt = 0
                k = n
                while k <= hi:
                    if not covered[k]:
                        cnt += 1
                    k += p

                if eff:
                    score = cnt * p
                else:
                    score = cnt

                if score > best_score or (score == best_score and (best_p is None or p < best_p)):
                    best_score = score
                    best_p = p
                    best_r = r

            if best_p is None:
                break

            used.add(best_p)
            budget -= Fraction(1, best_p)
            prime_res[best_p] = best_r

            s = best_r
            if s == 0:
                s = best_p
            for k in range(s, CAP + 1, best_p):
                covered[k] = 1

            n += 1

        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)
        return L

    # Phase 1: Multiple greedy strategies with different windows and efficiency metrics
    strategies = []
    for w in [200, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 50000]:
        for eff in [True, False]:
            strategies.append((w, eff))

    # Sort to try mid-range windows with efficiency first
    strategies.sort(key=lambda x: (not x[1], abs(x[0] - 3000)))

    for w, eff in strategies:
        if time.time() > deadline - 30:
            break
        greedy_init(w, eff, deadline)

    # Phase 2: Local search refinement
    if time.time() < deadline - 10 and best_classes:
        prime_res = {p: r for p, r in best_classes}
        used = set(prime_res.keys())
        budget_used = sum(Fraction(1, p) for p in prime_res)
        remaining = Fraction(1) - budget_used

        covered = make_coverage(prime_res)
        L = compute_L(covered)

        no_improve = 0
        max_no_improve = 60

        while time.time() < deadline - 3 and no_improve < max_no_improve:
            gap = L + 1
            if gap > CAP:
                break
            improved = False

            # Move 1: Try reassigning residue of an existing prime to cover gap
            candidates = []
            for p, r in prime_res.items():
                new_r = gap % p
                if new_r != r:
                    candidates.append((p, r, new_r))

            # Try a limited number of reassignments
            for p, old_r, new_r in candidates[:15]:
                if time.time() > deadline - 3:
                    return best_classes

                trial_res = dict(prime_res)
                trial_res[p] = new_r
                trial_cov = make_coverage(trial_res)
                L2 = compute_L(trial_cov)

                if L2 > L:
                    prime_res = trial_res
                    covered = trial_cov
                    L = L2
                    no_improve = 0
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_classes = build_classes(prime_res)
                    if L >= CAP:
                        return best_classes
                    break

            # Move 2: Add an unused prime with residue covering gap
            unused = [p for p in PRIMES if p not in used and remaining >= Fraction(1, p)]
            # Prioritize smaller primes (more coverage per budget)
            unused.sort()
            for p in unused[:15]:
                if time.time() > deadline - 3:
                    return best_classes

                r = gap % p
                new_cov = covered[:]
                s = r
                if s == 0:
                    s = p
                for k in range(s, CAP + 1, p):
                    new_cov[k] = 1
                L2 = compute_L(new_cov)
                if L2 > L:
                    prime_res[p] = r
                    used.add(p)
                    remaining -= Fraction(1, p)
                    covered = new_cov
                    L = L2
                    no_improve = 0
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_classes = build_classes(prime_res)
                    if L >= CAP:
                        return best_classes
                    break

            # Move 3: Try swapping - remove one prime and add another
            if not improved:
                # Try removing a large prime (high cost) and adding a smaller one
                sorted_used = sorted(used, key=lambda p: -1.0 / p)
                for p_remove in sorted_used[:8]:
                    if time.time() > deadline - 3:
                        return best_classes

                    trial_res = dict(prime_res)
                    del trial_res[p_remove]
                    trial_used = set(used)
                    trial_used.remove(p_remove)
                    trial_remaining = remaining + Fraction(1, p_remove)
                    trial_cov = make_coverage(trial_res)
                    L_base = compute_L(trial_cov)

                    # Try adding a different prime
                    for p_add in PRIMES:
                        if p_add in trial_used:
                            continue
                        if trial_remaining < Fraction(1, p_add):
                            continue

                        r_add = L_base % p_add if L_base < CAP else 0
                        trial_res2 = dict(trial_res)
                        trial_res2[p_add] = r_add
                        trial_cov2 = make_coverage(trial_res2)
                        L2 = compute_L(trial_cov2)

                        if L2 > L:
                            prime_res = trial_res2
                            used = set(trial_res2.keys())
                            remaining = Fraction(1) - sum(Fraction(1, p) for p in prime_res)
                            covered = trial_cov2
                            L = L2
                            no_improve = 0
                            improved = True
                            if L > best_L:
                                best_L = L
                                best_classes = build_classes(prime_res)
                            if L >= CAP:
                                return best_classes
                            break
                    if improved:
                        break

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

    # Fallback if nothing found
    if not best_classes:
        # Simple valid solution: just use prime 2, residue 1
        best_classes = [[2, 1]]
        best_L = 1

    return best_classes

if __name__ == "__main__":
    classes = construct()
    # Verify
    from fractions import Fraction
    s = sum(Fraction(1, p) for p, r in classes)
    print(f"Sum={float(s)}, Classes={len(classes)}")
    # Quick L check
    covered = bytearray(CAP + 1)
    for p, r in classes:
        start = r
        if start == 0:
            start = p
        for k in range(start, CAP + 1, p):
            covered[k] = 1
    L = 0
    for i in range(1, CAP + 1):
        if covered[i]:
            L = i
        else:
            break
    print(f"L={L}")
