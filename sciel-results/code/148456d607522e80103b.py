import time
import random
from fractions import Fraction
import numpy as np

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 150s kill

    primes = _sieve(1000)
    prime_set = set(primes)
    CAP = 100000
    random.seed(42)

    best_L = 0
    best_classes = []

    def compute_L(covered):
        # covered is a numpy bool array or bytearray; find first False from 1
        # Use numpy for speed
        if isinstance(covered, np.ndarray):
            arr = covered[1:]
            if arr.all():
                return CAP
            idx = np.argmax(~arr)
            return int(idx)
        else:
            L = 0
            for i in range(1, CAP + 1):
                if covered[i]:
                    L = i
                else:
                    break
            return L

    def build_classes(prime_res):
        return [[p, prime_res[p]] for p in sorted(prime_res.keys())]

    def make_coverage_np(prime_res):
        covered = np.zeros(CAP + 1, dtype=np.uint8)
        for p, r in prime_res.items():
            s = r
            if s == 0:
                s = p
            covered[s::p] = 1
        return covered

    def make_coverage_byte(prime_res):
        covered = bytearray(CAP + 1)
        for p, r in prime_res.items():
            s = r
            if s == 0:
                s = p
            for k in range(s, CAP + 1, p):
                covered[k] = 1
        return covered

    def greedy_init(window, eff, deadline, rng=None):
        nonlocal best_L, best_classes
        if rng is None:
            rng = random.Random()
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

                if eff == 0:
                    score = cnt
                elif eff == 1:
                    score = cnt * p
                else:
                    # eff==2: density-weighted, favor primes where uncovered density is high
                    # cnt / (hi - n + 1) * p
                    span = hi - n + 1
                    score = (cnt / span) * p * 1000 if span > 0 else 0

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
        return L, prime_res

    def simulated_annealing(prime_res, deadline, rng=None):
        """
        Simulated annealing to escape local optima.
        State: dict of prime -> residue
        Moves: add, remove, swap, change residue
        Objective: maximize L (coverage length)
        """
        nonlocal best_L, best_classes
        if rng is None:
            rng = random.Random()

        current_res = dict(prime_res)
        current_L = compute_L(make_coverage_np(current_res))
        current_cost = sum(Fraction(1, p) for p in current_res)

        if current_L > best_L:
            best_L = current_L
            best_classes = build_classes(current_res)

        # Temperature schedule: start high, cool geometrically
        T_start = 50.0
        T_end = 0.1
        cooling_rate = 0.985
        T = T_start

        max_iter = 20000
        iter_count = 0
        no_improve = 0
        max_no_improve = 3000

        while time.time() < deadline - 3 and iter_count < max_iter and no_improve < max_no_improve:
            iter_count += 1
            if iter_count % 500 == 0:
                T = max(T_end, T * cooling_rate)

            # Choose a random move
            move_type = rng.random()
            new_res = dict(current_res)
            new_cost = current_cost

            if move_type < 0.35:
                # Change residue of an existing prime
                if not current_res:
                    continue
                p = rng.choice(list(current_res.keys()))
                old_r = current_res[p]
                new_r = rng.randrange(p)
                if new_r == old_r:
                    continue
                new_res[p] = new_r

            elif move_type < 0.60:
                # Add a new prime
                unused = [p for p in primes if p not in current_res]
                if not unused:
                    continue
                # Filter by budget
                available = [p for p in unused if new_cost + Fraction(1, p) <= Fraction(1)]
                if not available:
                    continue
                p = rng.choice(available)
                r = rng.randrange(p)
                new_res[p] = r
                new_cost += Fraction(1, p)

            elif move_type < 0.80:
                # Remove a prime
                if not current_res:
                    continue
                p = rng.choice(list(current_res.keys()))
                del new_res[p]
                new_cost -= Fraction(1, p)

            else:
                # Swap: remove one, add another
                if not current_res:
                    continue
                p_remove = rng.choice(list(current_res.keys()))
                del new_res[p_remove]
                new_cost -= Fraction(1, p_remove)

                unused = [p for p in primes if p not in new_res]
                available = [p for p in unused if new_cost + Fraction(1, p) <= Fraction(1)]
                if available:
                    p_add = rng.choice(available)
                    r_add = rng.randrange(p_add)
                    new_res[p_add] = r_add
                    new_cost += Fraction(1, p_add)

            # Evaluate new state
            new_L = compute_L(make_coverage_np(new_res))

            # Acceptance criterion
            delta_L = new_L - current_L
            if delta_L >= 0:
                # Always accept improvements
                current_res = new_res
                current_L = new_L
                current_cost = new_cost
                no_improve = 0
                if current_L > best_L:
                    best_L = current_L
                    best_classes = build_classes(current_res)
            else:
                # Accept worse solutions with probability exp(delta/T)
                if T > 0.01:
                    accept_prob = np.exp(delta_L / T)
                    if rng.random() < accept_prob:
                        current_res = new_res
                        current_L = new_L
                        current_cost = new_cost
                no_improve += 1

        # Final update
        if current_L > best_L:
            best_L = current_L
            best_classes = build_classes(current_res)

        return current_L, current_res

    # Phase 1: Multiple greedy strategies
    # eff: 0=count, 1=count*p, 2=density*p
    strategies = []
    for w in [500, 1000, 2000, 3000, 5000, 8000, 12000, 20000]:
        for eff in [0, 1, 2]:
            strategies.append((w, eff))

    # Sort: try mid-range windows with eff=1 first, then eff=2, then eff=0
    def strat_key(x):
        w, eff = x
        eff_rank = {1: 0, 2: 1, 0: 2}[eff]
        return (eff_rank, abs(w - 3000))

    strategies.sort(key=strat_key)

    # Run greedy strategies, keeping track of the best prime_res for local search
    best_prime_res_for_ls = None
    best_prime_res_for_sa = None
    for w, eff in strategies:
        if time.time() > deadline - 30:
            break
        L, prime_res = greedy_init(w, eff, deadline)
        if L > best_L - 10:  # keep if competitive
            if best_prime_res_for_ls is None or L > best_L - 50:
                best_prime_res_for_ls = dict(prime_res)
            if best_prime_res_for_sa is None or L > best_L - 100:
                best_prime_res_for_sa = dict(prime_res)

    # Phase 2: Simulated Annealing (primary optimization)
    if time.time() < deadline - 15 and best_prime_res_for_sa:
        sa_rng = random.Random(42)
        L, prime_res = simulated_annealing(best_prime_res_for_sa, deadline, rng=sa_rng)
        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)

    # Phase 3: Local search refinement on the SA result
    if time.time() < deadline - 8 and best_prime_res_for_ls:
        prime_res = dict(best_prime_res_for_ls)
        used = set(prime_res.keys())
        budget_used = sum(Fraction(1, p) for p in prime_res)
        remaining = Fraction(1) - budget_used

        covered = make_coverage_np(prime_res)
        L = compute_L(covered)

        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)

        if L < CAP:
            no_improve = 0
            max_no_improve = 60
            ls_rng = random.Random(123)

            while time.time() < deadline - 3 and no_improve < max_no_improve:
                gap = L + 1
                improved = False

                # Move 1: Try reassigning residue of an existing prime to cover gap
                candidates = []
                for p, r in prime_res.items():
                    new_r = gap % p
                    if new_r != r:
                        candidates.append((p, r, new_r))

                ls_rng.shuffle(candidates)
                # Try a limited number of reassignments
                for p, old_r, new_r in candidates[:12]:
                    if time.time() > deadline - 3:
                        break

                    trial_res = dict(prime_res)
                    trial_res[p] = new_r
                    trial_cov = make_coverage_np(trial_res)
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
                            break
                        break

                if L >= CAP:
                    break

                # Move 2: Add an unused prime with residue covering gap
                unused = [p for p in primes if p not in used and remaining >= Fraction(1, p)]
                ls_rng.shuffle(unused)
                unused = unused[:15]

                for p in unused:
                    if time.time() > deadline - 3:
                        break
                    r = gap % p
                    trial_res = dict(prime_res)
                    trial_res[p] = r
                    trial_cov = make_coverage_np(trial_res)
                    L2 = compute_L(trial_cov)
                    if L2 > L:
                        prime_res = trial_res
                        used.add(p)
                        remaining -= Fraction(1, p)
                        covered = trial_cov
                        L = L2
                        no_improve = 0
                        improved = True
                        if L > best_L:
                            best_L = L
                            best_classes = build_classes(prime_res)
                        if L >= CAP:
                            break
                        break

                if L >= CAP:
                    break

                # Move 3: Random perturbation - pick a random used prime, try random residue
                if not improved and used:
                    p_rand = ls_rng.choice(list(used))
                    r_rand = ls_rng.randrange(p_rand)
                    if r_rand != prime_res[p_rand]:
                        trial_res = dict(prime_res)
                        trial_res[p_rand] = r_rand
                        trial_cov = make_coverage_np(trial_res)
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

                # Move 4: Swap - remove one prime and try adding a different one
                if not improved and used:
                    # Sort used primes by cost descending (largest p first, since 1/p is smallest)
                    # Actually remove the one with largest p (smallest cost) to free least budget? 
                    # No, we want to free budget, so remove smallest p (largest cost) first
                    sorted_used = sorted(used, key=lambda p: p)  # smallest p first (highest cost)
                    for p_remove in sorted_used[:6]:
                        if time.time() > deadline - 3:
                            break

                        trial_res = dict(prime_res)
                        del trial_res[p_remove]
                        trial_remaining = remaining + Fraction(1, p_remove)

                        # Try adding several unused primes
                        unused_for_swap = [p for p in primes if p not in used and p != p_remove
                                           and trial_remaining >= Fraction(1, p)]
                        ls_rng.shuffle(unused_for_swap)
                        unused_for_swap = unused_for_swap[:10]

                        for p_add in unused_for_swap:
                            if time.time() > deadline - 3:
                                break
                            r_add = gap % p_add
                            trial_res2 = dict(trial_res)
                            trial_res2[p_add] = r_add
                            trial_cov = make_coverage_np(trial_res2)
                            L2 = compute_L(trial_cov)
                            if L2 > L:
                                prime_res = trial_res2
                                used.discard(p_remove)
                                used.add(p_add)
                                remaining = trial_remaining - Fraction(1, p_add)
                                covered = trial_cov
                                L = L2
                                no_improve = 0
                                improved = True
                                if L > best_L:
                                    best_L = L
                                    best_classes = build_classes(prime_res)
                                if L >= CAP:
                                    break
                                break
                            if L >= CAP:
                                break

                        if L >= CAP:
                            break

                if not improved:
                    no_improve += 1

        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)

    # Phase 4: If we have time, try a couple more randomized greedy restarts + quick local search
    if time.time() < deadline - 20:
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            rng = random.Random(1000 + restart * 7)
            w = rng.choice([2000, 3000, 5000, 8000])
            eff = rng.choice([1, 2])
            L, prime_res = greedy_init(w, eff, deadline, rng=rng)

            if L > best_L:
                best_L = L
                best_classes = build_classes(prime_res)

            # Quick local search on this new solution
            if time.time() < deadline - 10 and L >= best_L - 50:
                used = set(prime_res.keys())
                budget_used = sum(Fraction(1, p) for p in prime_res)
                remaining = Fraction(1) - budget_used
                covered = make_coverage_np(prime_res)
                L = compute_L(covered)

                for _ in range(15):  # limited iterations
                    if time.time() > deadline - 5:
                        break
                    if L >= CAP:
                        break
                    gap = L + 1
                    improved = False

                    # Try reassign
                    candidates = [(p, r, gap % p) for p, r in prime_res.items() if gap % p != r]
                    rng.shuffle(candidates)
                    for p, old_r, new_r in candidates[:8]:
                        trial_res = dict(prime_res)
                        trial_res[p] = new_r
                        trial_cov = make_coverage_np(trial_res)
                        L2 = compute_L(trial_cov)
                        if L2 > L:
                            prime_res = trial_res
                            covered = trial_cov
                            L = L2
                            improved = True
                            if L > best_L:
                                best_L = L
                                best_classes = build_classes(prime_res)
                            break

                    # Try add
                    if not improved:
                        unused = [p for p in primes if p not in used and remaining >= Fraction(1, p)]
                        rng.shuffle(unused)
                        for p in unused[:8]:
                            r = gap % p
                            trial_res = dict(prime_res)
                            trial_res[p] = r
                            trial_cov = make_coverage_np(trial_res)
                            L2 = compute_L(trial_cov)
                            if L2 > L:
                                prime_res[p] = r
                                used.add(p)
                                remaining -= Fraction(1, p)
                                covered = trial_cov
                                L = L2
                                improved = True
                                if L > best_L:
                                    best_L = L
                                    best_classes = build_classes(prime_res)
                                break

                    if not improved:
                        break

                if L > best_L:
                    best_L = L
                    best_classes = build_classes(prime_res)

    if not best_classes:
        # Fallback: Use small primes to ensure valid return
        small = [p for p in primes if p <= 97]
        total_cost = sum(Fraction(1, p) for p in small)
        while total_cost > Fraction(1) and small:
            small.pop()
            total_cost = sum(Fraction(1, p) for p in small)
        best_classes = [[p, 0] for p in small]

    return best_classes
