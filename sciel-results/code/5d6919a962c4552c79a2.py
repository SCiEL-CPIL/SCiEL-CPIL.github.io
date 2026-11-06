import time
from fractions import Fraction
import random
import math

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    primes = _sieve(1000)
    CAP = 100000
    random.seed(42)

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

    def build_classes(prime_res):
        return [[p, prime_res[p]] for p in sorted(prime_res.keys())]

    def make_coverage(prime_res):
        covered = bytearray(CAP + 1)
        for p, r in prime_res.items():
            s = r
            if s == 0:
                s = p
            for k in range(s, CAP + 1, p):
                covered[k] = 1
        return covered

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

            for p in primes:
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

    # Phase 1: Multiple greedy strategies
    strategies = []
    for w in [500, 1000, 2000, 3000, 5000, 8000, 12000, 20000]:
        for eff in [True, False]:
            strategies.append((w, eff))

    strategies.sort(key=lambda x: (not x[1], abs(x[0] - 3000)))

    for w, eff in strategies:
        if time.time() > deadline - 15:
            break
        greedy_init(w, eff, deadline)

    # Phase 2: Enhanced local search
    if time.time() < deadline - 8 and best_classes:
        prime_res = {p: r for p, r in best_classes}
        used = set(prime_res.keys())
        budget_used = sum(Fraction(1, p) for p in prime_res)
        remaining = Fraction(1) - budget_used

        covered = make_coverage(prime_res)
        L = compute_L(covered)

        if L < CAP:
            no_improve = 0
            max_no_improve = 150

            while time.time() < deadline - 3 and no_improve < max_no_improve:
                gap = L + 1
                improved = False

                # Move 1: Try reassigning residue of an existing prime to cover gap
                # Also try residues that cover gap AND nearby uncovered spots
                best_reassign = None
                best_reassign_L = L
                for p, r in prime_res.items():
                    # Primary: cover the gap
                    new_r = gap % p
                    if new_r != r:
                        trial_res = dict(prime_res)
                        trial_res[p] = new_r
                        trial_cov = make_coverage(trial_res)
                        L2 = compute_L(trial_cov)
                        if L2 > best_reassign_L:
                            best_reassign_L = L2
                            best_reassign = (p, r, new_r, trial_res, trial_cov)
                    
                    # Secondary: try nearby residues that might cover more
                    for offset in [-2, -1, 1, 2, -3, 3]:
                        nr = (new_r + offset) % p
                        if nr != r and nr != new_r:
                            trial_res = dict(prime_res)
                            trial_res[p] = nr
                            trial_cov = make_coverage(trial_res)
                            L2 = compute_L(trial_cov)
                            if L2 > best_reassign_L:
                                best_reassign_L = L2
                                best_reassign = (p, r, nr, trial_res, trial_cov)

                if best_reassign and best_reassign_L > L:
                    p, old_r, new_r, trial_res, trial_cov = best_reassign
                    prime_res = trial_res
                    covered = trial_cov
                    L = best_reassign_L
                    no_improve = 0
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_classes = build_classes(prime_res)
                    if L >= CAP:
                        return best_classes

                # Move 2: Add an unused prime with residue covering gap
                if not improved:
                    unused = [p for p in primes if p not in used and remaining >= Fraction(1, p)]
                    random.shuffle(unused)
                    unused = unused[:25]

                    best_add = None
                    best_add_L = L
                    for p in unused:
                        if time.time() > deadline - 3:
                            break
                        r = gap % p
                        new_cov = bytearray(covered)
                        s = r
                        if s == 0:
                            s = p
                        for k in range(s, CAP + 1, p):
                            new_cov[k] = 1
                        L2 = compute_L(new_cov)
                        if L2 > best_add_L:
                            best_add_L = L2
                            best_add = (p, r, new_cov)

                    if best_add and best_add_L > L:
                        p, r, new_cov = best_add
                        prime_res[p] = r
                        used.add(p)
                        remaining -= Fraction(1, p)
                        covered = new_cov
                        L = best_add_L
                        no_improve = 0
                        improved = True
                        if L > best_L:
                            best_L = L
                            best_classes = build_classes(prime_res)
                        if L >= CAP:
                            return best_classes

                # Move 3: Try swapping - remove one prime and add another
                if not improved:
                    sorted_used = sorted(used, key=lambda p: -1.0/p)
                    best_swap = None
                    best_swap_L = L

                    for p_remove in sorted_used[:15]:
                        if time.time() > deadline - 3:
                            break

                        trial_res = dict(prime_res)
                        del trial_res[p_remove]
                        trial_used = set(used)
                        trial_used.remove(p_remove)
                        trial_remaining = remaining + Fraction(1, p_remove)

                        unused_after_remove = [p for p in primes if p not in trial_used and trial_remaining >= Fraction(1, p)]
                        random.shuffle(unused_after_remove)

                        for p_add in unused_after_remove[:20]:
                            if time.time() > deadline - 3:
                                break
                            # Try multiple residue options for the new prime
                            for r_offset in [0, -1, 1, -2, 2]:
                                r = (gap + r_offset) % p_add
                                trial_res2 = dict(trial_res)
                                trial_res2[p_add] = r
                                trial_cov = make_coverage(trial_res2)
                                L2 = compute_L(trial_cov)
                                if L2 > best_swap_L:
                                    best_swap_L = L2
                                    best_swap = (p_remove, p_add, r, trial_res2, trial_used, trial_remaining, trial_cov)

                    if best_swap and best_swap_L > L:
                        p_remove, p_add, r, trial_res2, trial_used, trial_remaining, trial_cov = best_swap
                        prime_res = trial_res2
                        used = trial_used | {p_add}
                        remaining = trial_remaining - Fraction(1, p_add)
                        covered = trial_cov
                        L = best_swap_L
                        no_improve = 0
                        improved = True
                        if L > best_L:
                            best_L = L
                            best_classes = build_classes(prime_res)
                        if L >= CAP:
                            return best_classes

                # Move 4: Try multiple residue changes simultaneously (2-opt style)
                if not improved:
                    prime_list = list(prime_res.items())
                    random.shuffle(prime_list)
                    n_items = len(prime_list)
                    # Try pairs of primes with gap-covering residues
                    for i in range(min(10, n_items)):
                        if time.time() > deadline - 3:
                            break
                        p1, r1 = prime_list[i]
                        new_r1 = gap % p1
                        for j in range(i+1, min(i+10, n_items)):
                            p2, r2 = prime_list[j]
                            new_r2 = gap % p2
                            if new_r1 != r1 or new_r2 != r2:
                                trial_res = dict(prime_res)
                                trial_res[p1] = new_r1
                                trial_res[p2] = new_r2
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

                # Move 5: Try 3-opt residue changes
                if not improved and len(prime_res) >= 3:
                    prime_list = list(prime_res.items())
                    random.shuffle(prime_list)
                    n_items = len(prime_list)
                    for i in range(min(5, n_items)):
                        if time.time() > deadline - 3:
                            break
                        p1, r1 = prime_list[i]
                        new_r1 = gap % p1
                        for j in range(i+1, min(i+5, n_items)):
                            p2, r2 = prime_list[j]
                            new_r2 = gap % p2
                            for k in range(j+1, min(j+5, n_items)):
                                p3, r3 = prime_list[k]
                                new_r3 = gap % p3
                                if new_r1 != r1 or new_r2 != r2 or new_r3 != r3:
                                    trial_res = dict(prime_res)
                                    trial_res[p1] = new_r1
                                    trial_res[p2] = new_r2
                                    trial_res[p3] = new_r3
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

                # Move 6: Simulated annealing style - accept worse moves sometimes
                if not improved and no_improve > 15:
                    temperature = max(0.05, 0.8 - no_improve / 60.0)
                    prime_list = list(prime_res.items())
                    random.shuffle(prime_list)
                    # Try changing 1-2 random residues
                    n_changes = random.randint(1, 2)
                    changed = []
                    for _ in range(n_changes):
                        p1, r1 = random.choice(prime_list)
                        new_r1 = random.randint(0, p1 - 1)
                        if new_r1 != r1:
                            changed.append((p1, r1, new_r1))
                    
                    if changed:
                        trial_res = dict(prime_res)
                        for p, old_r, new_r in changed:
                            trial_res[p] = new_r
                        trial_cov = make_coverage(trial_res)
                        L2 = compute_L(trial_cov)
                        if L2 >= L or random.random() < temperature:
                            prime_res = trial_res
                            covered = trial_cov
                            L = L2
                            no_improve = 0
                            improved = True
                            if L > best_L:
                                best_L = L
                                best_classes = build_classes(prime_res)

                # Move 7: Gap-focused search - find all uncovered spots near gap and try to cover them
                if not improved and no_improve > 30:
                    # Find the next few uncovered positions after L
                    uncovered_spots = []
                    pos = L + 1
                    while pos <= min(L + 100, CAP) and len(uncovered_spots) < 5:
                        if not covered[pos]:
                            uncovered_spots.append(pos)
                        pos += 1
                    
                    if uncovered_spots:
                        target = uncovered_spots[0]
                        # Try to cover target by adjusting multiple primes
                        best_combo = None
                        best_combo_L = L
                        
                        # Try single prime adjustments targeting the first uncovered
                        for p, r in prime_res.items():
                            new_r = target % p
                            if new_r != r:
                                trial_res = dict(prime_res)
                                trial_res[p] = new_r
                                trial_cov = make_coverage(trial_res)
                                L2 = compute_L(trial_cov)
                                if L2 > best_combo_L:
                                    best_combo_L = L2
                                    best_combo = (trial_res, trial_cov, L2)
                        
                        if best_combo and best_combo_L > L:
                            prime_res = best_combo[0]
                            covered = best_combo[1]
                            L = best_combo[2]
                            no_improve = 0
                            improved = True
                            if L > best_L:
                                best_L = L
                                best_classes = build_classes(prime_res)
                            if L >= CAP:
                                return best_classes

                if not improved:
                    no_improve += 1

        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)

    # Phase 3: Final polish - try to extend coverage by fine-tuning residues
    if time.time() < deadline - 2 and best_classes:
        prime_res = {p: r for p, r in best_classes}
        covered = make_coverage(prime_res)
        L = compute_L(covered)
        
        if L < CAP:
            # Try all possible single-residue changes that improve L
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                gap = L + 1
                for p, r in list(prime_res.items()):
                    if time.time() > deadline - 1:
                        break
                    # Try the residue that covers gap
                    new_r = gap % p
                    if new_r != r:
                        trial_res = dict(prime_res)
                        trial_res[p] = new_r
                        trial_cov = make_coverage(trial_res)
                        L2 = compute_L(trial_cov)
                        if L2 > L:
                            prime_res = trial_res
                            covered = trial_cov
                            L = L2
                            improved = True
                            if L > best_L:
                                best_L = L
                                best_classes = build_classes(prime_res)
                            if L >= CAP:
                                return best_classes
                            break

    if not best_classes:
        small = [p for p in primes if p <= 97]
        total_cost = sum(Fraction(1, p) for p in small)
        while total_cost > Fraction(1) and small:
            small.pop()
            total_cost = sum(Fraction(1, p) for p in small)
        best_classes = [[p, 0] for p in small]

    return best_classes
