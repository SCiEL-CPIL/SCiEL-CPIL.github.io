import time
import random
import math
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 118  # Safety margin

    # Generate primes up to 1000
    def _primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    all_primes = _primes_upto(1000)
    
    # Select primes to maximize density while staying within budget
    # We want to keep small primes as they provide high coverage density.
    # So we start with all primes and remove the largest ones if budget is exceeded.
    selected_primes = all_primes[:]
    
    # Calculate sum of reciprocals exactly
    def calc_cost(primes):
        return sum(Fraction(1, p) for p in primes)
    
    cost = calc_cost(selected_primes)
    
    # If cost > 1, remove largest primes until cost <= 1
    # selected_primes is sorted ascending. Remove from end.
    while cost > Fraction(1, 1) and len(selected_primes) > 0:
        # Remove the largest prime
        p_to_remove = selected_primes[-1]
        selected_primes.pop()
        cost -= Fraction(1, p_to_remove)
        
    # If we still have cost > 1 (unlikely given the distribution), 
    # we might need to remove small primes, but that would be bad for coverage.
    # However, sum of 1/p for p<=1000 is ~1.26. 
    # Removing largest primes (1/997, 1/991...) reduces cost slowly.
    # We might need to remove many large primes.
    
    # Let's verify the count. 
    # If we remove all primes > 100, the sum of removed is small.
    # We need to remove enough to drop from ~1.26 to 1.0.
    # The sum of 1/p for p in [100, 1000] is approx 0.1-0.2?
    # If that's not enough, we remove primes in [50, 100], etc.
    
    # This strategy keeps the smallest primes, which is optimal for coverage.
    
    N = len(selected_primes)
    MAX_L = 100000  # Cap at 10^5 as specified

    # Initialize residues with diagonal strategy: r_i = i mod p
    # This ensures good initial coverage
    residues = [i % p for i, p in enumerate(selected_primes)]
    
    # Initialize counts using NumPy
    counts = np.zeros(MAX_L + 1, dtype=np.int16)
    
    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        counts[r::p] += 1

    def get_L():
        # Find first index i >= 1 where counts[i] == 0
        # np.argmax returns 0 if all are non-zero (no gap found in 1..MAX_L)
        # But we want the index of the first 0.
        # If counts[1:] == 0 has any True, argmax gives the first index.
        # If no True, argmax gives 0 (which is incorrect for "no gap").
        # We need to handle the case where all are covered.
        gap_mask = counts[1:] == 0
        if not np.any(gap_mask):
            return MAX_L
        idx = np.argmax(gap_mask)
        return idx

    L = get_L()
    best_L = L
    best_residues = residues[:]
    
    # Greedy prefix extension
    # Try to extend coverage one step at a time
    # This is fast and gets us to a high L quickly.
    
    # We will run this greedy phase for a limited time, then switch to SA.
    # Actually, the greedy loop in the original code was infinite until stuck.
    # Let's run it for a short time or until it gets stuck.
    
    greedy_deadline = t0 + 5.0
    while time.time() < greedy_deadline:
        if L >= MAX_L:
            break
        
        first_gap = L + 1
        found = False
        for i in range(N):
            p = selected_primes[i]
            old_r = residues[i]
            new_r = first_gap % p
            
            if new_r == old_r:
                continue
            
            # Check if change is beneficial
            counts[old_r::p] -= 1
            counts[new_r::p] += 1
            
            new_L = get_L()
            
            if new_L >= L:
                residues[i] = new_r
                L = new_L
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
                found = True
                break
            else:
                counts[new_r::p] -= 1
                counts[old_r::p] += 1
        
        if not found:
            break # Stuck in greedy
    
    # Now use Multi-Start Simulated Annealing
    # We will run SA in blocks. If no improvement in global best, restart.
    
    def run_sa(start_residues, time_limit):
        """
        Runs SA starting from start_residues for time_limit seconds.
        Returns best_L and best_residues found in this run.
        """
        # Copy state
        cur_residues = start_residues[:]
        cur_counts = np.zeros(MAX_L + 1, dtype=np.int16)
        for i in range(N):
            p = selected_primes[i]
            r = cur_residues[i]
            cur_counts[r::p] += 1
            
        cur_L = get_L()
        run_best_L = cur_L
        run_best_res = cur_residues[:]
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.999
        stuck_count = 0
        max_stuck = 5000 # Restart SA within the block if stuck too long
        
        sa_start = time.time()
        
        while time.time() < sa_start + time_limit:
            if cur_L >= MAX_L:
                break
                
            i = random.randint(0, N - 1)
            p = selected_primes[i]
            old_r = cur_residues[i]
            new_r = random.randint(0, p - 1)
            
            if old_r == new_r:
                continue
            
            cur_counts[old_r::p] -= 1
            cur_counts[new_r::p] += 1
            
            new_L = get_L()
            
            if new_L > cur_L:
                cur_L = new_L
                cur_residues[i] = new_r
                if cur_L > run_best_L:
                    run_best_L = cur_L
                    run_best_res = cur_residues[:]
                stuck_count = 0
            else:
                delta = cur_L - new_L
                if delta == 0:
                    if random.random() < 0.5:
                        cur_L = new_L
                        cur_residues[i] = new_r
                    else:
                        cur_counts[new_r::p] -= 1
                        cur_counts[old_r::p] += 1
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        cur_L = new_L
                        cur_residues[i] = new_r
                    else:
                        cur_counts[new_r::p] -= 1
                        cur_counts[old_r::p] += 1
                        stuck_count += 1
                        if stuck_count > max_stuck:
                            # Restart temperature or break? 
                            # Let's just cool down faster or accept worse?
                            # For now, just continue.
                            pass
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return run_best_L, run_best_res

    # Main SA Loop
    # We will run SA blocks of 10 seconds.
    # If global best doesn't improve, we perturb the global best and start a new block.
    
    current_best_res = best_residues[:]
    
    while time.time() < deadline:
        # Determine time for this SA block
        block_time = min(10.0, deadline - time.time())
        if block_time < 1.0:
            break
            
        # Start from current best
        # Perturb slightly to escape local minima if we've been stuck
        # For the first block, we use the greedy result.
        # Subsequent blocks, we perturb the global best.
        
        start_res = current_best_res[:]
        
        # Perturb 1-3 random residues
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, N - 1)
            p = selected_primes[idx]
            start_res[idx] = random.randint(0, p - 1)
            
        run_L, run_res = run_sa(start_res, block_time)
        
        if run_L > best_L:
            best_L = run_L
            best_residues = run_res[:]
            current_best_res = best_residues[:]
            
    return [[selected_primes[i], best_residues[i]] for i in range(N)]
