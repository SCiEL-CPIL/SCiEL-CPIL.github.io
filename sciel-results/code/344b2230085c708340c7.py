import time
import numpy as np

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _first_start(r, p):
    if r == 0:
        return p
    return r

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    def _compute_counts(residues, max_check):
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = _first_start(r, p)
            if start <= max_check:
                counts[start::p] += 1
        return counts

    def _find_L(counts):
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return len(counts) - 1
        return int(below[0])

    def _update_counts(counts, old_r, new_r, p, max_check):
        old_start = _first_start(old_r, p)
        if old_start <= max_check:
            counts[old_start::p] -= 1
        new_start = _first_start(new_r, p)
        if new_start <= max_check:
            counts[new_start::p] += 1

    def _greedy_setcover(residues, max_check, time_limit):
        """
        Greedy set-cover: process primes in increasing order, for each prime
        select the residue that maximizes the number of positions in [1, current_L]
        that are covered by exactly one congruence (converting them to double-covered).
        """
        residues = list(residues)
        counts = _compute_counts(residues, max_check)
        best_L = _find_L(counts)
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < time_limit and no_improve < max_no_improve:
            if best_L >= max_check:
                break
            
            # Find positions with count == 1 in [1, best_L]
            single_positions = np.where((counts[1:best_L+1] == 1))[0] + 1
            if len(single_positions) == 0:
                no_improve += 1
                continue
            
            # For each prime, find the residue that covers the most single-covered positions
            best_gain = 0
            best_i = -1
            best_new_r = -1
            best_old_r = -1
            
            for i, p in enumerate(primes):
                old_r = residues[i]
                
                # Group single positions by their residue mod p
                residues_mod_p = single_positions % p
                
                # Count how many single positions each residue would cover
                # We need to find which residue class covers the most single positions
                unique_residues, counts_per_residue = np.unique(residues_mod_p, return_counts=True)
                
                # The best residue is the one with the highest count
                best_idx = np.argmax(counts_per_residue)
                candidate_r = int(unique_residues[best_idx])
                candidate_gain = int(counts_per_residue[best_idx])
                
                if candidate_r == old_r:
                    # Already at this residue, try second best
                    if len(unique_residues) > 1:
                        # Remove the best and find next
                        sorted_idx = np.argsort(-counts_per_residue)
                        for idx in sorted_idx:
                            cr = int(unique_residues[idx])
                            cg = int(counts_per_residue[idx])
                            if cr != old_r:
                                candidate_r = cr
                                candidate_gain = cg
                                break
                        else:
                            continue
                    else:
                        continue
                
                if candidate_gain > best_gain:
                    best_gain = candidate_gain
                    best_i = i
                    best_new_r = candidate_r
                    best_old_r = old_r
            
            if best_i != -1 and best_gain > 0:
                _update_counts(counts, best_old_r, best_new_r, primes[best_i], max_check)
                residues[best_i] = best_new_r
                new_L = _find_L(counts)
                if new_L > best_L:
                    best_L = new_L
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        return residues, best_L

    def _greedy_critical_optimize(residues, max_check, time_limit):
        """
        Greedy: find first position with count < 2 and try to fix it.
        """
        residues = list(residues)
        counts = _compute_counts(residues, max_check)
        best_L = _find_L(counts)
        
        no_improve = 0
        max_no_improve = 300
        
        while time.time() < time_limit and no_improve < max_no_improve:
            if best_L >= max_check:
                break
            
            target_x = best_L + 1
            
            best_delta = 0
            best_i = -1
            best_new_r = -1
            best_old_r = -1
            
            for i, p in enumerate(primes):
                old_r = residues[i]
                new_r = target_x % p
                
                if new_r == old_r:
                    continue
                
                _update_counts(counts, old_r, new_r, p, max_check)
                new_L = _find_L(counts)
                _update_counts(counts, new_r, old_r, p, max_check)
                
                delta = new_L - best_L
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_r = new_r
                    best_old_r = old_r
            
            if best_i != -1:
                _update_counts(counts, best_old_r, best_new_r, primes[best_i], max_check)
                residues[best_i] = best_new_r
                best_L += best_delta
                no_improve = 0
            else:
                no_improve += 1
        
        return residues, best_L

    def _local_search(residues, max_check, time_limit, accept_neutral=False):
        """
        Local search with optional neutral move acceptance to escape local minima.
        """
        residues = list(residues)
        counts = _compute_counts(residues, max_check)
        best_L = _find_L(counts)
        
        no_improve = 0
        max_no_improve = 400
        
        while time.time() < time_limit and no_improve < max_no_improve:
            if best_L >= max_check:
                break
            
            # Try random single-residue changes
            best_delta = 0
            best_i = -1
            best_new_r = -1
            best_old_r = -1
            
            # Number of random trials per iteration
            num_trials = min(50, n)
            
            for _ in range(num_trials):
                i = int(np.random.randint(0, n))
                p = primes[i]
                old_r = residues[i]
                new_r = int(np.random.randint(0, p))
                
                if new_r == old_r:
                    continue
                
                _update_counts(counts, old_r, new_r, p, max_check)
                new_L = _find_L(counts)
                _update_counts(counts, new_r, old_r, p, max_check)
                
                delta = new_L - best_L
                if delta > best_delta or (accept_neutral and delta == 0 and best_i == -1):
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_new_r = new_r
                        best_old_r = old_r
                    elif accept_neutral and delta == 0 and best_i == -1:
                        best_i = i
                        best_new_r = new_r
                        best_old_r = old_r
            
            if best_i != -1:
                _update_counts(counts, best_old_r, best_new_r, primes[best_i], max_check)
                residues[best_i] = best_new_r
                best_L += best_delta
                if best_delta > 0:
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        return residues, best_L

    def _random_restarts(residues, max_check, time_limit, num_restarts=10):
        best_residues = list(residues)
        best_L = _find_L(_compute_counts(residues, max_check))
        
        for r in range(num_restarts):
            if time.time() >= time_limit:
                break
            
            random_residues = [int(np.random.randint(0, p)) for p in primes]
            
            # Use greedy set-cover first, then local search
            res, L = _greedy_setcover(random_residues, max_check, time_limit)
            
            # Polish with local search
            if time.time() < time_limit and L < max_check:
                res, L = _local_search(res, max_check, time_limit, accept_neutral=True)
            
            if L > best_L:
                best_residues = res
                best_L = L
        
        return best_residues, best_L

    # Start with r_p = 1 for all primes
    init_residues = [1] * n
    current_M = 2000
    
    # Initial optimization
    current_residues, current_L = _greedy_setcover(init_residues, current_M, deadline)
    
    # Polish
    if current_L < current_M and time.time() < deadline:
        current_residues, current_L = _local_search(current_residues, current_M, deadline, accept_neutral=True)
    
    # Try to extend if we hit the boundary
    while time.time() < deadline:
        if current_L < current_M:
            # Stuck - try random restarts
            current_residues, current_L = _random_restarts(
                current_residues, current_M, deadline, num_restarts=8
            )
            if current_L < current_M:
                # Final local search
                current_residues, current_L = _local_search(
                    current_residues, current_M, deadline, accept_neutral=True
                )
                if current_L < current_M:
                    break
        
        # Extend the range
        current_M = min(current_M * 2, 10000)
        current_residues, current_L = _greedy_setcover(current_residues, current_M, deadline)
        if current_L < current_M and time.time() < deadline:
            current_residues, current_L = _local_search(current_residues, current_M, deadline, accept_neutral=True)
    
    return [int(r) for r in current_residues]
