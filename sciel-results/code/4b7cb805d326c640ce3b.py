"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Generate primes up to 1000 using sieve
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
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Fast evaluation: compute coverage counts up to a bound
    # and find the largest L where all 1..L have count >= 2
    EVAL_BOUND = 5000  # evaluate up to this
    
    def compute_counts(residues):
        """Compute coverage counts for positions 1..EVAL_BOUND."""
        counts = np.zeros(EVAL_BOUND + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= EVAL_BOUND:
                # Add 1 to all positions first, first+p, first+2p, ...
                idx = np.arange(first, EVAL_BOUND + 1, p)
                counts[idx] += 1
        return counts
    
    def get_L_from_counts(counts):
        """Find largest L such that counts[1..L] >= 2."""
        # Find first position where count < 2
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return EVAL_BOUND
        return below[0]  # 0-indexed, so position is below[0] + 1, but L is below[0]
    
    def evaluate(residues):
        counts = compute_counts(residues)
        return get_L_from_counts(counts), counts
    
    # Initial assignment: seed strategy
    current = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    current_L, current_counts = evaluate(current)
    
    best = current[:]
    best_L = current_L
    
    # Simulated annealing with targeted moves
    random.seed(12345)
    
    # Temperature schedule
    T0 = 5.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T0
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5.0 and no_improve < max_no_improve and T > T_end:
        # Identify the first gap
        if current_L >= EVAL_BOUND:
            break
        gap_pos = current_L + 1  # first position not covered enough
        
        # Strategy: find primes that could cover gap_pos by changing their residue
        # A prime p covers gap_pos if residue = gap_pos % p
        # But we want to change residues to FIX the gap, not just cover it
        # We need to find which primes currently don't cover gap_pos, and whether
        # changing their residue would help cover gap_pos AND not create new gaps
        
        # Targeted move: pick a random prime and try changing its residue
        # to one that covers the gap position
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = current[i]
        
        # The residue that would make prime i cover gap_pos
        target_r = gap_pos % p
        if target_r == 0:
            target_r = p  # but residues are 0..p-1, so 0 means multiples of p
            # Actually residue 0 means x ≡ 0 (mod p), so gap_pos is covered if gap_pos % p == 0
            # Let me reconsider: residue r means x ≡ r (mod p)
            # So gap_pos is covered by prime i iff gap_pos % p == r
            # If gap_pos % p == 0, then r should be 0
            target_r = 0
        
        # Try the target residue first
        new_r = target_r
        if new_r == old_r:
            # Already covers the gap? Then the gap is due to other primes
            # Try a random residue
            new_r = random.randint(0, p - 1)
        
        current[i] = new_r
        new_L, new_counts = evaluate(current)
        
        # Accept or reject based on simulated annealing
        delta = new_L - current_L
        if delta > 0 or (T > 0 and random.random() < np.exp(delta / T)):
            current_L = new_L
            current_counts = new_counts
            no_improve = 0
            if new_L > best_L:
                best_L = new_L
                best = current[:]
        else:
            current[i] = old_r
            no_improve += 1
        
        # Cool down
        T *= cooling_rate
        
        # Occasionally do a more aggressive move: change multiple residues
        if no_improve % 100 == 0 and time.time() < deadline - 10.0:
            # Try to fix the gap by changing residues of several small primes
            # that are "close" to covering the gap
            small_primes_idx = [j for j in range(min(30, n_primes))]
            random.shuffle(small_primes_idx)
            for j in small_primes_idx[:5]:
                pj = primes[j]
                old_rj = current[j]
                target_rj = gap_pos % pj
                if target_rj == 0:
                    target_rj = 0
                current[j] = target_rj
            
            new_L, new_counts = evaluate(current)
            delta = new_L - current_L
            if delta > 0 or (T > 0 and random.random() < np.exp(delta / T)):
                current_L = new_L
                current_counts = new_counts
                no_improve = 0
                if new_L > best_L:
                    best_L = new_L
                    best = current[:]
            else:
                # Revert
                for j in small_primes_idx[:5]:
                    current[j] = old_rj
                no_improve += 1
    
    # Final local search: try to improve by targeting specific gaps
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            if best_L >= EVAL_BOUND:
                break
            
            counts = compute_counts(best)
            L = get_L_from_counts(counts)
            if L >= EVAL_BOUND:
                break
            gap_pos = L + 1
            
            # Find all primes that don't cover gap_pos
            uncovered = []
            for i, p in enumerate(primes):
                if best[i] != gap_pos % p:
                    uncovered.append(i)
            
            if not uncovered:
                break
            
            # Try changing one of the uncovered primes to cover the gap
            random.shuffle(uncovered)
            improved = False
            for i in uncovered[:10]:
                p = primes[i]
                old_r = best[i]
                target_r = gap_pos % p
                best[i] = target_r
                new_L, _ = evaluate(best)
                if new_L > L:
                    L = new_L
                    improved = True
                    break
                else:
                    best[i] = old_r
            
            if improved:
                best_L = L
            else:
                break
    
    return best
# EVOLVE-BLOCK-END
