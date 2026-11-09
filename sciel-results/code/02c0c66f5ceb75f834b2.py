import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill
    
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
    
    # We evaluate up to a large limit to ensure we can find improvements
    # The score is the largest L such that 1..L is doubly covered.
    MAX_EVAL = 20000
    
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def evaluate(residues):
        """
        Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
        Returns L (capped at MAX_EVAL).
        """
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        first_fail_idx = np.argmax(mask)
        return int(xs[first_fail_idx] - 1)

    def get_coverage_matrix(residues):
        """
        Returns the boolean matrix where hits[i, j] is True if prime i covers integer j.
        Shape: (n_primes, MAX_EVAL)
        """
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        return (mods == 0)

    # Initial solution: Random residues
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    # Precompute the "coverage matrix" for the current best state
    hits = get_coverage_matrix(residues)
    coverage = np.sum(hits, axis=0)
    
    # Main Loop
    while time.time() < deadline:
        # Update current score
        current_score = evaluate(residues)
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        if current_score >= MAX_EVAL:
            break
            
        target_x = current_score + 1
        target_idx = target_x - 1 # 0-indexed in xs
        
        # Identify primes NOT covering target_x
        covers_target = hits[:, target_idx]
        not_covers_target = np.where(~covers_target)[0]
        
        if len(not_covers_target) == 0:
            break
            
        # Phase 1: 1-Opt Repair
        found_repair = False
        candidates_1 = list(not_covers_target)
        random.shuffle(candidates_1)
        
        for idx in candidates_1:
            old_res = residues[idx]
            new_res = target_x % primes[idx]
            if old_res == new_res: continue
            
            # Simulate change
            residues[idx] = new_res
            hits = get_coverage_matrix(residues)
            coverage = np.sum(hits, axis=0)
            
            new_score = evaluate(residues)
            if new_score >= current_score:
                found_repair = True
                current_score = new_score
                break
            else:
                residues[idx] = old_res
                # Revert hits/coverage? We'll recompute if no repair found.
                
        if found_repair:
            continue
            
        # Revert hits/coverage if 1-opt failed
        hits = get_coverage_matrix(residues)
        coverage = np.sum(hits, axis=0)
        
        # Phase 2: 2-Opt Swap
        found_2opt = False
        candidates_2_i = list(not_covers_target)
        random.shuffle(candidates_2_i)
        
        # Limit the search for 2-opt to keep it fast
        max_i_tries = 5
        
        for i in candidates_2_i[:max_i_tries]:
            old_res_i = residues[i]
            new_res_i = target_x % primes[i]
            if old_res_i == new_res_i: continue
            
            # Simulate change of i
            residues[i] = new_res_i
            hits_sim = get_coverage_matrix(residues)
            cov_sim = np.sum(hits_sim, axis=0)
            
            # Find broken integers in 1..target_x
            # An integer y is broken if cov_sim[y] < 2
            # We specifically want to fix the ones that were 2 and are now 1.
            # But any broken integer in the prefix is a problem.
            broken_indices = np.where((cov_sim < 2) & (np.arange(MAX_EVAL) < target_idx + 1))[0]
            
            # Revert i
            residues[i] = old_res_i
            
            if len(broken_indices) == 0:
                # i alone fixed it? Then 1-opt should have worked.
                # Let's just accept it if it improves score.
                # But we are in 2-opt phase.
                # Let's just skip.
                continue
                
            # Pick a random broken integer y
            y_idx = random.choice(broken_indices)
            y = xs[y_idx]
            
            # Find a prime j to cover y
            # j should not be i.
            # Ideally, j is not covering y currently.
            covers_y = hits_sim[:, y_idx]
            not_covers_y = np.where(~covers_y)[0]
            
            if len(not_covers_y) == 0:
                continue
                
            # Try a few random j's
            j_candidates = list(not_covers_y)
            random.shuffle(j_candidates)
            
            for j in j_candidates[:3]:
                if j == i: continue
                
                old_res_j = residues[j]
                new_res_j = y % primes[j]
                if old_res_j == new_res_j: continue
                
                # Simulate the 2-move
                residues[i] = new_res_i
                residues[j] = new_res_j
                
                hits_final = get_coverage_matrix(residues)
                cov_final = np.sum(hits_final, axis=0)
                
                new_score = evaluate(residues)
                
                if new_score > current_score:
                    found_2opt = True
                    # Accept the move
                    hits = hits_final
                    coverage = cov_final
                    current_score = new_score
                    break
                else:
                    # Revert j
                    residues[j] = old_res_j
                    # Revert i? No, we will revert i after the j loop.
                    
            if found_2opt:
                break
                
            # Revert i
            residues[i] = old_res_i
            
        if found_2opt:
            continue
            
        # Phase 3: Kick
        # If 2-opt failed, do a random kick to escape local minimum.
        # Change 2 random primes to random residues.
        for _ in range(2):
            idx = random.randint(0, n_primes - 1)
            residues[idx] = random.randint(0, primes[idx] - 1)
            
        # Update hits and coverage
        hits = get_coverage_matrix(residues)
        coverage = np.sum(hits, axis=0)
        
    return best_residues.tolist()
