import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We aim to maximize the score L such that 1..L is doubly covered.
    # We use MCMC (Metropolis-Hastings) to explore the space of residues.
    # The "energy" is a function of the coverage profile.
    # We want to minimize the "badness" of the coverage.
    
    # Evaluation range: We check up to MAX_EVAL to find the score.
    # If we find a gap before MAX_EVAL, the score is the gap index - 1.
    # If no gap, score is MAX_EVAL.
    MAX_EVAL = 5000
    
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def compute_coverage(residues):
        """
        Returns an array of length MAX_EVAL where coverage[i] is the number of primes
        covering xs[i].
        """
        # residues: (n_primes,)
        # xs: (MAX_EVAL,)
        # We want to count how many p satisfy xs[i] % p == residues[p]
        # Equivalent to (xs[i] - residues[p]) % p == 0
        
        # Vectorized calculation might be memory heavy: 168 * 5000 = 840k elements.
        # This is fine.
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    def get_score_and_energy(coverage):
        """
        Returns (score, energy).
        score: Largest L such that 1..L has coverage >= 2.
        energy: A scalar to minimize.
                Primary: -score (we want high score)
                Secondary: Deficit at the first failure point (if score < MAX_EVAL)
                Tertiary: Number of points with coverage < 2 (global)
        """
        # Find first index where coverage < 2
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL, 0.0
        
        first_fail_idx = np.argmax(mask)
        score = int(xs[first_fail_idx] - 1)
        
        # Energy calculation
        # We want to minimize energy.
        # Higher score -> lower energy.
        # If score is the same, we want to reduce the "deficit" at the next step
        # or increase coverage of the failed step to make it easier to fix.
        
        # Let's define energy as:
        # E = -score * 1000 + deficit_at_fail + 0.1 * total_deficit
        # deficit_at_fail = 2 - coverage[first_fail_idx] (if score < MAX_EVAL)
        # total_deficit = sum(max(0, 2 - coverage[i]) for all i)
        
        if score < MAX_EVAL:
            fail_x = int(xs[first_fail_idx])
            deficit_at_fail = max(0, 2 - coverage[first_fail_idx])
        else:
            deficit_at_fail = 0
            
        # Calculate total deficit for the whole range?
        # This might be expensive if done every step.
        # Let's approximate or just use the first failure.
        # Actually, for MCMC, a smooth energy landscape is better.
        # Let's use:
        # E = -score * 1000 + (2 - coverage[first_fail_idx]) if score < MAX_EVAL else 0
        
        # To make it smoother, let's add a small penalty for low coverage in the prefix
        # This helps "push" the coverage up.
        # But computing sum of deficits is O(N).
        # Let's stick to the primary driver: -score.
        # To break ties or guide towards fixing the immediate next number:
        # If score < MAX_EVAL, add penalty for the specific number that failed.
        
        energy = -score * 1000.0
        
        if score < MAX_EVAL:
            # Penalize the depth of the hole at the first failure
            # If coverage is 0, penalty is higher than if coverage is 1
            energy += (2 - coverage[first_fail_idx]) * 10.0
            
        return score, energy

    # Initialize residues
    # A good initialization might be random, or based on a heuristic.
    # Let's start with random.
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    current_coverage = compute_coverage(residues)
    current_score, current_energy = get_score_and_energy(current_coverage)
    
    best_score = current_score
    best_residues = residues.copy()
    
    # MCMC Parameters
    # Temperature T starts high and decays.
    # We accept moves if delta_E < 0 or with probability exp(-delta_E / T).
    # delta_E = E_new - E_old.
    # Since we minimize E, E_new < E_old is good.
    
    T_start = 500.0
    T_end = 1.0
    # We run for a fixed number of steps or until time limit.
    # Let's estimate steps per second.
    # compute_coverage is ~0.1ms?
    # get_score is ~0.01ms?
    # So ~1ms per step. 100 steps/sec?
    # 115 seconds -> 10,000 steps.
    
    # Let's do a time-budgeted loop.
    step_count = 0
    last_time = time.time()
    
    while time.time() < deadline:
        # Annealing schedule
        # Linear decay or exponential?
        # Let's use exponential decay to T_end.
        # Or simply: T = T_start * (T_end / T_start) ** (step_count / max_steps)
        # But we don't know max_steps.
        # Let's use a time-based decay.
        elapsed = time.time() - start_time
        remaining_ratio = max(0.0, 1.0 - elapsed / 115.0)
        # T decays from T_start to T_end as remaining_ratio goes 1 -> 0
        # T = T_end + (T_start - T_end) * remaining_ratio
        T = T_end + (T_start - T_end) * remaining_ratio
        
        # Pick a random prime index to perturb
        i = random.randint(0, n_primes - 1)
        old_res = residues[i]
        
        # Pick a new residue
        # Uniform random is standard, but we can bias towards residues that cover the current failure point
        # Let's stick to uniform for simplicity and correctness of MH.
        new_res = random.randint(0, primes[i] - 1)
        
        if new_res == old_res:
            continue
            
        residues[i] = new_res
        
        # Compute new coverage
        # Optimization: We can update coverage incrementally.
        # Old contribution of prime i: 1 if (x - old_res) % p_i == 0 else 0
        # New contribution of prime i: 1 if (x - new_res) % p_i == 0 else 0
        # Delta = New - Old
        # This is O(MAX_EVAL) instead of O(N * MAX_EVAL).
        # MAX_EVAL = 5000. N=168.
        # Full recompute: 168 * 5000 = 840k ops.
        # Incremental: 5000 ops.
        # Huge speedup!
        
        p_i = primes[i]
        # Calculate where old_res covered
        # x % p_i == old_res  =>  x = k * p_i + old_res
        # We can construct a boolean mask for old and new.
        
        # Vectorized mask creation
        # old_mask: True if xs % p_i == old_res
        # new_mask: True if xs % p_i == new_res
        
        # Using modulo is fast enough.
        old_mask = (xs % p_i == old_res)
        new_mask = (xs % p_i == new_res)
        
        # Update coverage
        # coverage += new_mask - old_mask
        # Since they are booleans, convert to int or use np.add/subtract with where
        # Note: booleans are 0/1 in arithmetic
        
        # We need to be careful with types. coverage is int.
        current_coverage += (new_mask.astype(np.int8) - old_mask.astype(np.int8))
        
        new_score, new_energy = get_score_and_energy(current_coverage)
        
        # Metropolis-Hastings Acceptance
        delta_E = new_energy - current_energy
        
        if delta_E < 0:
            # Accept
            current_energy = new_energy
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
        else:
            # Accept with probability exp(-delta_E / T)
            if T > 1e-9:
                prob = np.exp(-delta_E / T)
                if random.random() < prob:
                    current_energy = new_energy
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    # Reject: revert residues and coverage
                    residues[i] = old_res
                    current_coverage += (old_mask.astype(np.int8) - new_mask.astype(np.int8))
            else:
                # T is very small, reject
                residues[i] = old_res
                current_coverage += (old_mask.astype(np.int8) - new_mask.astype(np.int8))
        
        step_count += 1
        
        # Periodic full recompute to correct any floating point or logic drift?
        # Not needed for integers, but good for debugging.
        # We'll skip it for speed, but ensure the incremental update is correct.
        # The incremental update is exact for integers.
        
    return best_residues.tolist()
