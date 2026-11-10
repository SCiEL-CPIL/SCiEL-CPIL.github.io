# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill

    def compute_mobius(n_max):
        mu = np.ones(n_max + 1, dtype=np.int64)
        is_prime = np.ones(n_max + 1, dtype=bool)
        for i in range(2, n_max + 1):
            if is_prime[i]:
                for j in range(i, n_max + 1, i):
                    is_prime[j] = False
                    mu[j] *= -1
                i2 = i * i
                if i2 <= n_max:
                    for j in range(i2, n_max + 1, i2):
                        mu[j] = 0
        return mu

    mu = compute_mobius(200)

    def evaluate(domain, values):
        """Evaluate a candidate. Returns -inf if infeasible, else a."""
        if len(domain) == 0:
            return -np.inf

        domain = np.asarray(domain, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)

        if len(domain) != len(values):
            return -np.inf
        if len(np.unique(domain)) != len(domain):
            return -np.inf
        if np.any(domain <= 0) or np.any(domain > 200000):
            return -np.inf
        if not np.all(np.isfinite(values)):
            return -np.inf

        # Clip
        vals = np.clip(values, -10.0, 10.0)

        # Compute S = sum vals/k
        S = np.sum(vals / domain.astype(np.float64))

        # f(1) adjustment
        if 1 in domain:
            idx = np.where(domain == 1)[0][0]
            vals = vals.copy()
            vals[idx] = vals[idx] - S
        else:
            # Add 1 with value -S
            domain = np.append(domain, 1)
            vals = np.append(vals, -S)
            sort_idx = np.argsort(domain)
            domain = domain[sort_idx]
            vals = vals[sort_idx]

        # Now sum vals/k should be ~0
        # Check M(x) <= 1.0001 for x in [1, 10K]
        K = int(np.max(domain))
        max_x = 10 * K

        # Efficient M(x) computation
        # M(x) = sum_i vals[i] * floor(x / domain[i])
        # floor(x/k) changes only at multiples of k
        # For small K, just compute directly
        if max_x <= 200000:
            # Vectorized: for each domain element, add vals[i] * floor(x/domain[i])
            # This is O(m * max_x)
            m = len(domain)
            # Use the fact that floor(x/k) = x//k
            # Build M array
            M = np.zeros(max_x + 1, dtype=np.float64)
            for i in range(m):
                k = domain[i]
                v = vals[i]
                if v != 0.0:
                    # floor(x/k) for x=1..max_x
                    # This is a sawtooth: 0 for x<k, 1 for k<=x<2k, etc.
                    # We can use integer division
                    M[1:max_x+1] += v * (np.arange(1, max_x+1) // k)
            if np.max(M[1:max_x+1]) > 1.0001:
                return -np.inf
        else:
            # K is large, sample or skip - but K<=200000 so max_x<=2000000
            # For now, if K > 20000, we might need to be more careful
            # But our search uses small K
            if K > 2000:
                # Too expensive, skip or approximate
                # Actually 10*2000 = 20000, m is small, so m*20000 is fine
                # Let's just do it
                M = np.zeros(max_x + 1, dtype=np.float64)
                for i in range(len(domain)):
                    k = domain[i]
                    v = vals[i]
                    if v != 0.0:
                        M[1:max_x+1] += v * (np.arange(1, max_x+1) // k)
                if np.max(M[1:max_x+1]) > 1.0001:
                    return -np.inf

        # Compute a = -sum vals * log(k) / k
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(vals * log_k / domain.astype(np.float64))
        return a

    # Seed
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    best_score = evaluate(seed_domain, seed_values)
    best_domain = seed_domain.copy()
    best_values = seed_values.copy()

    # Systematic search over small domains
    # Candidate pool: small integers, especially primes and their products
    # Focus on domain size 3 to 10

    # Pool of interesting numbers
    pool = list(range(1, 60))
    # Emphasize primes and small composites
    primes_pool = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    composites_pool = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 40, 42, 44, 45, 48, 49, 50, 52, 54, 55, 56, 57, 58]

    def try_candidate(domain, values):
        nonlocal best_score, best_domain, best_values
        score = evaluate(domain, values)
        if score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()
        return score

    # Strategy 1: Try Möbius-based values with various domain subsets
    # For a given domain, try values = c * mu(domain) and optimize c
    # Also try other patterns

    # Strategy 2: Random search with local refinement

    # First, try all subsets of size 3-8 from a focused pool
    focused_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 40, 48, 60]

    for size in range(3, 9):
        if time.time() > deadline - 60:
            break
        for combo in combinations(focused_pool, size):
            if time.time() > deadline - 60:
                break
            domain = np.array(combo, dtype=np.int64)
            # Try Möbius values
            vals = mu[domain].astype(np.float64)
            # Also try with 1 included
            if 1 not in combo:
                domain2 = np.array([1] + list(combo), dtype=np.int64)
                vals2 = np.array([1.0] + list(mu[combo].astype(np.float64)))
                try_candidate(domain2, vals2)

            # Try different value patterns
            # Pattern 1: mu values
            try_candidate(domain, vals)

            # Pattern 2: -mu for k>1, adjust for 1
            if 1 in domain:
                vals_alt = vals.copy()
                vals_alt[domain == 1] = 0  # Will be adjusted
                try_candidate(domain, vals_alt)

            # Pattern 3: All -1 for primes, +1 for 1
            vals_p = np.zeros(len(domain))
            for i, k in enumerate(domain):
                if k == 1:
                    vals_p[i] = 1.0
                elif mu[k] == -1:
                    vals_p[i] = -1.0
                elif mu[k] == 1:
                    vals_p[i] = 1.0
                else:
                    vals_p[i] = 0.0
            try_candidate(domain, vals_p)

    # Strategy 3: Random search with value optimization
    np.random.seed(42)

    for iteration in range(50000):
        if time.time() > deadline - 30:
            break

        # Random domain size
        m = np.random.randint(3, 12)
        # Random domain from pool
        domain = np.random.choice(pool, size=m, replace=False)
        domain = np.sort(domain)

        # Ensure 1 is included with some probability
        if np.random.random() < 0.7:
            if 1 not in domain:
                domain = np.append(domain, 1)
                domain = np.sort(domain)
                m += 1

        # Try several value patterns
        # Pattern A: Möbius
        vals = mu[domain].astype(np.float64)
        try_candidate(domain, vals)

        # Pattern B: -1 for all k>1, 1 for k=1
        vals_b = np.zeros(len(domain))
        for i, k in enumerate(domain):
            vals_b[i] = 1.0 if k == 1 else -1.0
        try_candidate(domain, vals_b)

        # Pattern C: Random values in [-10, 10]
        vals_c = np.random.uniform(-10, 10, size=len(domain))
        try_candidate(domain, vals_c)

        # Pattern D: Scaled Möbius
        scale = np.random.uniform(0.5, 3.0)
        vals_d = mu[domain].astype(np.float64) * scale
        try_candidate(domain, vals_d)

        # Pattern E: -log(k) based
        vals_e = -np.log(domain.astype(np.float64))
        try_candidate(domain, vals_e)

        # Pattern F: 1/k based
        vals_f = 1.0 / domain.astype(np.float64)
        try_candidate(domain, vals_f)

    # Strategy 4: Local refinement around best found
    if best_score > -np.inf:
        for _ in range(5000):
            if time.time() > deadline - 10:
                break

            # Perturb domain
            new_domain = best_domain.copy()
            op = np.random.randint(0, 3)
            if op == 0 and len(new_domain) > 2:
                # Remove a random element (not 1)
                non_one = np.where(new_domain != 1)[0]
                if len(non_one) > 0:
                    idx = np.random.choice(non_one)
                    new_domain = np.delete(new_domain, idx)
                    new_values = np.delete(best_values, idx)
                    try_candidate(new_domain, new_values)
            elif op == 1:
                # Add a random element
                candidates = [k for k in range(1, 100) if k not in new_domain]
                if candidates:
                    new_k = np.random.choice(candidates)
                    new_domain = np.append(new_domain, new_k)
                    new_domain = np.sort(new_domain)
                    new_values = np.append(best_values, mu[new_k].astype(np.float64))
                    new_values = new_values[np.argsort(new_domain)]
                    new_domain = new_domain[np.argsort(new_domain)]
                    try_candidate(new_domain, new_values)
            elif op == 2:
                # Perturb values
                new_values = best_values + np.random.normal(0, 0.5, size=len(best_values))
                try_candidate(best_domain, new_values)

    return best_domain, best_values

# EVOLVE-BLOCK-END
