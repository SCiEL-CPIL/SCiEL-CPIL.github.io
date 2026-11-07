import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    candidates = list(range(2, 41))
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composites = [n for n in candidates if n not in primes]

    def check_add(products, x):
        """Check if adding x to current set keeps all subset products distinct.
        Returns (valid, new_products) where new_products is the updated set."""
        new_prods = []
        for p in products:
            np = p * x
            if np in products:
                return False, None
            new_prods.append(np)
        # Check new_prods for internal duplicates (shouldn't happen if products distinct and x>0)
        # But also check that new_prods don't overlap with products
        for np in new_prods:
            if np in products:
                return False, None
        # Check new_prods among themselves
        seen = set()
        for np in new_prods:
            if np in seen:
                return False, None
            seen.add(np)
        return True, products | set(new_prods)

    # Initial best: all primes
    best_set = primes[:]
    best_size = len(best_set)
    best_products = {1}
    for p in primes:
        best_products |= {p * x for x in best_products}

    def try_greedy(order):
        """Try greedy addition in given order, return (set, products, size)."""
        prods = {1}
        s = []
        for x in order:
            valid, new_prods = check_add(prods, x)
            if valid:
                s.append(x)
                prods = new_prods
        return s, prods, len(s)

    # Phase 1: Deterministic greedy with primes first, then composites sorted
    for comp_order in [sorted(composites), sorted(composites, reverse=True)]:
        order = primes + comp_order
        s, prods, sz = try_greedy(order)
        if sz > best_size:
            best_size = sz
            best_set = s[:]
            best_products = prods
        if best_size >= 18:
            break

    # Phase 2: Random restarts with greedy
    random.shuffle(composites)
    while time.time() < deadline:
        order = primes[:] + composites[:]
        random.shuffle(order)
        # Ensure primes are tried but not necessarily first - actually let's keep primes first
        # but shuffle composites
        order = primes + composites[:]
        random.shuffle(composites)
        s, prods, sz = try_greedy(order)
        if sz > best_size:
            best_size = sz
            best_set = s[:]
            best_products = prods
            if best_size >= 18:
                break

    # Phase 3: Local search - try to improve by swapping
    # Work from best_set, try adding/removing/swapping
    current_set = best_set[:]
    current_products = best_products
    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline and no_improve < max_no_improve:
        improved = False

        # Try adding a random element
        others = [x for x in candidates if x not in current_set]
        if others:
            random.shuffle(others)
            for cand in others[:20]:  # Try up to 20 random candidates
                if time.time() > deadline:
                    break
                valid, new_prods = check_add(current_products, cand)
                if valid:
                    current_set.append(cand)
                    current_products = new_prods
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                        best_products = current_products
                    improved = True
                    break

        if not improved:
            # Try swapping: remove one element, add another
            if current_set:
                swap_idx = random.randint(0, len(current_set) - 1)
                removed = current_set[swap_idx]
                # Recompute products without removed
                # This is expensive; instead, try a few swaps
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    cand = random.choice(others) if others else None
                    if cand is None or cand == removed:
                        continue
                    # Build set without removed, then try adding cand
                    trial_set = [x for i, x in enumerate(current_set) if i != swap_idx]
                    trial_prods = {1}
                    for x in trial_set:
                        valid_t, tp = check_add(trial_prods, x)
                        if not valid_t:
                            break
                        trial_prods = tp
                    else:
                        valid_t, tp = check_add(trial_prods, cand)
                        if valid_t:
                            current_set = trial_set + [cand]
                            current_products = tp
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = current_set[:]
                                best_products = current_products
                            improved = True
                            break

        if improved:
            no_improve = 0
        else:
            no_improve += 1

    # Final verification
    final_prods = {1}
    for x in best_set:
        final_prods |= {x * p for p in final_prods}
    if len(final_prods) != (1 << len(best_set)):
        # Fallback to primes if something went wrong
        best_set = primes[:]

    return [int(x) for x in best_set]
