# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin (150s limit, leave buffer)

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_list = primes
    primes_arr = np.array(primes_list, dtype=np.int64)

    # Evaluation bound
    EVAL_BOUND = 10000

    def compute_counts_fast(residues):
        """Compute coverage counts for positions 1..EVAL_BOUND using vectorized ops."""
        counts = np.zeros(EVAL_BOUND + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_list[i]
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= EVAL_BOUND:
                idx = np.arange(first, EVAL_BOUND + 1, p)
                counts[idx] += 1
        return counts

    def get_L_from_counts(counts):
        """Find largest L such that counts[1..L] >= 2."""
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return EVAL_BOUND
        return int(below[0])

    def evaluate(residues):
        counts = compute_counts_fast(residues)
        return get_L_from_counts(counts)

    def compute_local_counts(residues, lo, hi):
        """Compute counts only in range [lo, hi] for faster local evaluation."""
        if hi > EVAL_BOUND:
            hi = EVAL_BOUND
        if lo < 1:
            lo = 1
        counts = np.zeros(hi - lo + 2, dtype=np.int32)  # indices 0..hi-lo+1, where offset lo maps to 0
        for i in range(n_primes):
            p = primes_list[i]
            r = residues[i]
            # Find first position >= lo that is congruent to r mod p
            # x = r (mod p), x >= lo
            # x = r + k*p >= lo => k >= ceil((lo - r)/p) if lo > r, else k >= 0
            if r >= lo:
                first = r
            else:
                # Need smallest k >= 0 such that r + k*p >= lo
                k = (lo - r + p - 1) // p
                first = r + k * p
            if first <= hi:
                idx = np.arange(first - lo, hi - lo + 1, p)
                counts[idx] += 1
        return counts

    def get_local_L(residues, start_pos, lookahead=50):
        """Get L by checking from start_pos with limited lookahead for speed."""
        check_end = min(start_pos + lookahead, EVAL_BOUND)
        counts = compute_local_counts(residues, start_pos, check_end)
        # Find first position where count < 2
        below = np.where(counts < 2)[0]
        if len(below) == 0:
            return check_end
        return start_pos + int(below[0]) - 1

    # Greedy initialization: maximize coverage density
    def greedy_init():
        residues = [0] * n_primes
        # Sort primes by size (already sorted)
        # For each prime, choose residue that covers the most uncovered positions in [1, 200]
        coverage = np.zeros(201, dtype=np.int32)
        for i in range(n_primes):
            p = primes_list[i]
            best_r = 0
            best_score = -1
            # Try all residues
            for r in range(p):
                # Count how many new positions this covers in [1, 200]
                first = r if r > 0 else p
                if first > 200:
                    continue
                cnt = 0
                for x in range(first, 201, p):
                    if coverage[x] < 2:
                        cnt += 1
                if cnt > best_score:
                    best_score = cnt
                    best_r = r
            residues[i] = best_r
            # Update coverage
            r = best_r
            first = r if r > 0 else p
            for x in range(first, 201, p):
                coverage[x] += 1
        return residues

    # Simulated Annealing
    random.seed(42)
    
    # Start with greedy init
    best_residues = greedy_init()
    best_L = evaluate(best_residues)
    
    current = best_residues[:]
    current_L = best_L
    
    # SA parameters
    T0 = 10.0  # Initial temperature
    T_min = 0.1
    alpha = 0.995  # Cooling rate
    T = T0
    
    iteration = 0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # Choose a random prime to modify
        i = random.randint(0, n_primes - 1)
        p = primes_list[i]
        old_r = current[i]
        
        # Choose new residue: either random or targeted
        if random.random() < 0.3 and current_L > 0:
            # Targeted: try to cover the gap at current_L + 1
            gap_pos = current_L + 1
            new_r = gap_pos % p
        else:
            new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        current[i] = new_r
        
        # Evaluate with local lookahead for speed
        # Check from max(1, current_L - 5) to current_L + 100
        check_start = max(1, current_L - 5)
        new_L = get_local_L(current, check_start, lookahead=200)
        
        # If local evaluation shows improvement, do full evaluation
        if new_L > current_L:
            full_L = evaluate(current)
            if full_L > current_L:
                current_L = full_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current[:]
                    no_improve = 0
                    # Cool down slower when improving
                    T = max(T * 0.999, T_min)
                else:
                    # Local was optimistic, revert
                    current[i] = old_r
            else:
                # Local was optimistic, revert
                current[i] = old_r
        else:
            # Accept worse move with probability exp(-delta/T)
            delta = current_L - new_L
            if delta > 0:
                accept_prob = math.exp(-delta / max(T, 0.01))
                if random.random() < accept_prob:
                    # Accept, but don't update best
                    pass
                else:
                    current[i] = old_r
            else:
                # Same or better, accept
                pass
        
        # Cool down
        T = max(T * alpha, T_min)
        
        # Reset if stuck too long
        no_improve += 1
        if no_improve > max_no_improve:
            # Restart from best with perturbation
            current = best_residues[:]
            for _ in range(n_primes // 5):
                idx = random.randint(0, n_primes - 1)
                current[idx] = random.randint(0, primes_list[idx] - 1)
            current_L = evaluate(current)
            T = T0 * 0.5
            no_improve = 0
        
        if best_L >= 1000:
            break
    
    # Final polish: targeted repair on best solution
    current = best_residues[:]
    current_L = best_L
    
    polish_steps = 0
    max_polish = 500
    while current_L < EVAL_BOUND and polish_steps < max_polish and time.time() < deadline - 2.0:
        gap_pos = current_L + 1
        if gap_pos > EVAL_BOUND:
            break
        
        # Find primes not covering gap_pos
        candidates = []
        for i in range(n_primes):
            if current[i] != gap_pos % primes_list[i]:
                candidates.append(i)
        
        if not candidates:
            break
        
        random.shuffle(candidates)
        improved = False
        
        for i in candidates[:30]:
            p = primes_list[i]
            old_r = current[i]
            new_r = gap_pos % p
            
            current[i] = new_r
            new_L = evaluate(current)
            
            if new_L > current_L:
                current_L = new_L
                improved = True
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current[:]
                break
            else:
                current[i] = old_r
        
        if not improved:
            break
        
        polish_steps += 1
    
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
