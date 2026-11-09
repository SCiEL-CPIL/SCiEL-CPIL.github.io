import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    
    # We will search for L up to M. 
    M = 2000
    counts = np.zeros(M + 1, dtype=np.int64)
    
    # Greedy initialization: build coverage incrementally
    # For each prime, pick the residue that maximizes coverage of currently weak numbers
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Start with all zeros, then greedily assign
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        for r in range(p):
            # Count how many numbers in [1, M] with x % p == r currently have count < 2
            # and how many have count == 1 (would become 2)
            idx = r
            if idx < 1:
                idx += p
            if idx > M:
                continue
            mask = counts[idx::p] < 2
            score = np.sum(mask)
            # Bonus for numbers that would go from 1 to 2
            mask1 = counts[idx::p] == 1
            score += np.sum(mask1)
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        counts[best_r::p] += 1
    
    def calc_L(counts, M):
        # Returns the largest L such that counts[1..L] >= 2
        if counts[1] < 2:
            return 0
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()
    
    # Helper to check if a move is valid (doesn't break prefix [1, L])
    def is_valid_move(counts, i, r_old, r_new, L, p):
        if r_old == r_new:
            return True
        # Check if any y in [1, L] congruent to r_old mod p has count 2
        start_y = r_old
        if start_y == 0:
            start_y = p
        for y in range(start_y, L + 1, p):
            if counts[y] == 2:
                return False
        return True

    # Helper to count risk: how many x in [1, L] with count==2 would be broken
    def count_risk(counts, r_old, L, p):
        risk = 0
        start_y = r_old
        if start_y == 0:
            start_y = p
        for y in range(start_y, L + 1, p):
            if counts[y] == 2:
                risk += 1
        return risk

    # Helper to apply move
    def apply_move(counts, i, r_old, r_new, p):
        if r_old != r_new:
            counts[r_old::p] -= 1
            counts[r_new::p] += 1

    # Helper to revert move
    def revert_move(counts, i, r_old, r_new, p):
        if r_old != r_new:
            counts[r_new::p] -= 1
            counts[r_old::p] += 1

    # Extend M if needed
    def extend_M(counts, residues, primes, M, new_M):
        old_counts = counts
        counts = np.zeros(new_M + 1, dtype=np.int64)
        counts[:M+1] = old_counts
        for j, pj in enumerate(primes):
            rj = residues[j]
            start_m = max(0, (M + 1 - rj + pj - 1) // pj)
            first_idx = rj + start_m * pj
            if first_idx <= new_M:
                counts[first_idx::pj] += 1
        return counts

    # 2-step repair: find two primes to change to cover a gap with count 0
    def try_2step_repair(counts, residues, primes, gap_x, current_L, M):
        # Find all primes that don't currently cover gap_x
        candidates = []
        for i, p in enumerate(primes):
            r_old = residues[i]
            if r_old != gap_x % p:
                risk = count_risk(counts, r_old, current_L, p)
                candidates.append((risk, i, p, r_old))
        
        if len(candidates) < 2:
            return None
        
        # Sort by risk, try the two lowest-risk candidates
        candidates.sort()
        
        # Try pairs from the top candidates
        for a in range(min(5, len(candidates))):
            for b in range(a+1, min(5, len(candidates))):
                risk_a, i_a, p_a, r_old_a = candidates[a]
                risk_b, i_b, p_b, r_old_b = candidates[b]
                
                if risk_a > 0 or risk_b > 0:
                    continue
                
                r_new_a = gap_x % p_a
                r_new_b = gap_x % p_b
                
                # Apply both moves
                apply_move(counts, i_a, r_old_a, r_new_a, p_a)
                apply_move(counts, i_b, r_old_b, r_new_b, p_b)
                residues[i_a] = r_new_a
                residues[i_b] = r_new_b
                
                new_L = calc_L(counts, M)
                
                if new_L > current_L:
                    return (new_L, i_a, i_b)
                else:
                    # Revert
                    revert_move(counts, i_a, r_new_a, r_old_a, p_a)
                    revert_move(counts, i_b, r_new_b, r_old_b, p_b)
                    residues[i_a] = r_old_a
                    residues[i_b] = r_old_b
        
        return None

    # Optimization Loop
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9998
    
    stuck_counter = 0
    repair_attempts = 0
    
    while time.time() - start_time < deadline:
        # 1. Try to repair the gap
        gap_x = current_L + 1
        
        if gap_x > M:
            # Extend M
            new_M = M * 2
            counts = extend_M(counts, residues, primes, M, new_M)
            M = new_M
            current_L = calc_L(counts, M)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            continue
        
        found_valid_move = False
        
        # Shuffle prime indices to avoid bias
        prime_indices = list(range(n_primes))
        random.shuffle(prime_indices)
        
        # Find the best single-prime repair with minimum risk
        best_risk = float('inf')
        best_i = -1
        best_p = -1
        best_r_old = -1
        
        for i in prime_indices:
            p = primes[i]
            r_old = residues[i]
            
            # If this prime already covers the gap, skip
            if r_old == gap_x % p:
                continue
                
            risk = count_risk(counts, r_old, current_L, p)
            
            if risk < best_risk:
                best_risk = risk
                best_i = i
                best_p = p
                best_r_old = r_old
        
        if best_i != -1 and best_risk == 0:
            # Apply the best repair move
            i = best_i
            p = best_p
            r_old = best_r_old
            r_new = gap_x % p
            
            apply_move(counts, i, r_old, r_new, p)
            residues[i] = r_new
            
            new_L = calc_L(counts, M)
            
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
            
            current_L = new_L
            found_valid_move = True
            stuck_counter = 0
            repair_attempts = 0
        else:
            # No zero-risk single move. Try 2-step repair
            result = try_2step_repair(counts, residues, primes, gap_x, current_L, M)
            if result is not None:
                new_L = result[0]
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                current_L = new_L
                found_valid_move = True
                stuck_counter = 0
                repair_attempts = 0
            else:
                # Use Simulated Annealing to escape
                repair_attempts += 1
                
                # Pick a random prime
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                r_old = residues[i]
                
                # Random new residue
                r_new = random.randint(0, p - 1)
                
                if r_new == r_old:
                    continue
                
                # Apply move
                apply_move(counts, i, r_old, r_new, p)
                
                new_L = calc_L(counts, M)
                
                if new_L >= current_L:
                    current_L = new_L
                    residues[i] = r_new
                    if new_L > best_L:
                        best_L = new_L
                        best_residues = residues.copy()
                    stuck_counter = 0
                else:
                    # SA acceptance
                    delta = current_L - new_L
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_L = new_L
                        residues[i] = r_new
                        stuck_counter = 0
                    else:
                        # Revert
                        revert_move(counts, i, r_old, r_new, p)
                        stuck_counter += 1
                
                # Cool down
                T = max(T_min, T * cooling_rate)
                
                # If stuck for a long time, do a more aggressive perturbation
                if stuck_counter > 50:
                    # Make 3 random changes
                    for _ in range(3):
                        i2 = random.randint(0, n_primes - 1)
                        p2 = primes[i2]
                        r_old2 = residues[i2]
                        r_new2 = random.randint(0, p2 - 1)
                        if r_new2 != r_old2:
                            apply_move(counts, i2, r_old2, r_new2, p2)
                            residues[i2] = r_new2
                    current_L = calc_L(counts, M)
                    stuck_counter = 0
                    T = max(T_min, T * 0.9)

    return [int(r) for r in best_residues]
