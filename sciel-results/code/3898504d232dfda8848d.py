<<<<<<< SEARCH
# ---------- local search: mutate one element to reduce distinct primes ----------
def _score(nums):
    return _count_distinct_prime_factors_of_sums(nums)

def _local_search(nums, prime_target_set, budget_iters=2000):
    """Hill-climb single-element changes to try to reach a solution whose pairwise
    sums use only primes in prime_target_set. Returns (improved_list, current_score)."""
    current = sorted(nums)
    current_score = _score(current)
    target = set(prime_target_set)
    iters = 0
    improved = True
    rng = random.Random(1234567)
    while improved and iters < budget_iters and not _deadline_hit():
        improved = False
        iters += 1
        # try to fix the worst-sum first
        # find a pair whose sum has a prime outside target; perturb one element.
        n = len(current)
        bad_pair = None
        for i in range(n):
            for j in range(i + 1, n):
                s = current[i] + current[j]
                for p in _distinct_prime_factors(s):
                    if p not in target:
                        bad_pair = (i, j)
                        break
                if bad_pair:
                    break
            if bad_pair:
                break
        if bad_pair is None:
            break  # already S-smooth!
        i, j = bad_pair
        # candidate new values for current[i]
        tried = set()
        # small neighborhood
        for d in list(range(-6, 7)) + list(range(-20, 21, 5)):
            v = current[i] + d
            if 1 <= v <= MAX_VAL and v not in tried and v not in current:
                tried.add(v)
                new_list = current[:]
                new_list[i] = v
                new_list = sorted(new_list)
                ns = _score(new_list)
                if ns < current_score:
                    current = new_list
                    current_score = ns
                    improved = True
                    break
        if not improved:
            # try a few random values
            for _ in range(8):
                v = rng.randint(1, MAX_VAL)
                if v in tried or v in current:
                    continue
                tried.add(v)
                new_list = current[:]
                new_list[i] = v
                new_list = sorted(new_list)
                ns = _score(new_list)
                if ns < current_score:
                    current = new_list
                    current_score = ns
                    improved = True
                    break
    return current, current_score
=======
# ---------- Local search to refine ----------
def _score(nums):
    return _count_distinct_prime_factors_of_sums(nums)

def _local_search(nums, budget_iters=500):
    """Generic hill-climb to reduce the number of distinct primes in pairwise sums."""
    current = sorted(nums)
    current_score = _score(current)
    iters = 0
    improved = True
    rng = random.Random(1234567)
    while improved and iters < budget_iters and not _deadline_hit():
        improved = False
        iters += 1
        # Try to improve by changing one element
        n = len(current)
        # Find the worst pair
        worst_pair = None
        worst_primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                s = current[i] + current[j]
                pf = _distinct_prime_factors(s)
                if len(pf) > len(worst_primes):
                    worst_primes = pf
                    worst_pair = (i, j)
        
        if worst_pair is None:
            break
            
        i, j = worst_pair
        tried = set()
        # Try small perturbations
        for d in list(range(-20, 21)) + list(range(-50, 51, 10)) + list(range(-100, 101, 20)):
            v = current[i] + d
            if 1 <= v <= MAX_VAL and v not in tried and v not in current:
                tried.add(v)
                new_list = current[:]
                new_list[i] = v
                new_list = sorted(new_list)
                ns = _score(new_list)
                if ns < current_score:
                    current = new_list
                    current_score = ns
                    improved = True
                    break
        if not improved:
            # Try random values
            for _ in range(15):
                v = rng.randint(1, MAX_VAL)
                if v in tried or v in current:
                    continue
                tried.add(v)
                new_list = current[:]
                new_list[i] = v
                new_list = sorted(new_list)
                ns = _score(new_list)
                if ns < current_score:
                    current = new_list
                    current_score = ns
                    improved = True
                    break
    return current, current_score
>>>>>>> REPLACE

