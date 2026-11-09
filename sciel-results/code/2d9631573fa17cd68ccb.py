import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

PRIMES = _primes_upto(1000)
N_PRIMES = len(PRIMES)
# Precompute prime indices for quick access
PRIME_LIST = PRIMES

def _score(residues, max_check=10000):
    """
    Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
    Uses difference array for O(N + L) complexity.
    """
    # diff[i] holds the change in coverage at i
    # We need to cover 1..max_check
    # diff size max_check + 2 to handle start+p <= max_check+1
    diff = np.zeros(max_check + 2, dtype=np.int32)
    
    for i, p in enumerate(PRIME_LIST):
        r = residues[i]
        # x = r (mod p)
        # We want x >= 1.
        # The first positive integer x such that x % p == r is:
        # if r == 0, x = p
        # else x = r
        start = r
        if start == 0:
            start = p
            
        if start <= max_check:
            diff[start] += 1
            if start + p <= max_check + 1:
                diff[start + p] -= 1
            # Note: if start+p > max_check+1, we don't need to decrement within bounds
            # as we only compute prefix sums up to max_check.
            # However, to be safe and correct for prefix sums, we only care about indices <= max_check.
            # The decrement at start+p is only relevant if start+p <= max_check.
            # If start+p > max_check, the coverage continues to the end of our range.
            
    # Compute prefix sums to get coverage counts
    # cov[x] = sum(diff[1..x])
    # We can use np.cumsum
    cov = np.cumsum(diff)
    
    # cov[0] is 0. cov[1] is coverage of 1.
    # We need cov[x] >= 2 for all x in 1..L.
    # Find first x >= 1 where cov[x] < 2.
    
    # Check if all are >= 2
    # valid[x] = cov[x] >= 2
    valid = cov >= 2
    
    # We care about indices 1..max_check
    # If valid[1] is False, L=0.
    # If valid[max_check] is True, L=max_check.
    
    # Find first index i in 1..max_check where valid[i] is False.
    # If none, return max_check.
    
    # np.argmax on a boolean array returns the first True.
    # We want the first False in valid[1:].
    # So we look for first True in ~valid[1:].
    
    if not np.any(~valid[1:max_check+1]):
        return max_check
        
    first_fail_idx = np.argmax(~valid[1:max_check+1])
    # first_fail_idx is 0-based relative to valid[1:]
    # So the failing index in original array is first_fail_idx + 1
    # The score L is (failing index) - 1 = first_fail_idx
    
    return int(first_fail_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to be safe
    
    # Initial residues: 
    # A good heuristic is to set r_p = 0 for all p? 
    # If r_p = 0, x is covered if p|x.
    # 1 is not covered by any. So L=0.
    # Better: Random start is okay, but let's try a slightly smarter start.
    # Or just random.
    
    residues = np.array([random.randint(0, p - 1) for p in PRIME_LIST], dtype=np.int32)
    
    best_residues = residues.copy()
    best_score = _score(best_residues, 1000) # Start with small check for speed
    
    current_residues = residues.copy()
    current_score = best_score
    
    # Local Search
    # We will iterate, finding the first failure point, and trying to fix it.
    
    max_check_for_score = 2000 # Increase this as we get better? No, keep it fixed for speed, or dynamic.
    # Let's use a fixed max_check for scoring to ensure consistency and speed.
    # If we find L > 1000, we can expand. But for now, 2000 is fast enough.
    
    while time.time() < deadline:
        # Determine the gap
        # current_score is the largest L such that 1..L is covered >= 2.
        # The number current_score + 1 is the first one NOT covered >= 2.
        gap = current_score + 1
        
        # If gap is very large, we might have hit the limit of our check.
        # If current_score == max_check_for_score, we should expand max_check.
        if current_score >= max_check_for_score:
             max_check_for_score *= 2
             current_score = _score(current_residues, max_check_for_score)
             if current_score > best_score:
                 best_score = current_score
                 best_residues = current_residues.copy()
             continue
        
        # Try to fix the gap
        # We want to increase coverage of `gap` to at least 2.
        # Currently, `gap` has coverage count C < 2.
        # We need to add 2-C more covers.
        # We can change residues of primes that do NOT currently cover `gap`.
        
        # Calculate current coverage of `gap`
        # This is expensive to do from scratch. 
        # However, we can just try moves and score them.
        
        # Candidate moves:
        # 1. Pick a prime p that does not divide (gap - r_p) i.e., gap % p != r_p.
        #    Set r_p_new = gap % p.
        #    This guarantees `gap` gets +1 coverage from this prime.
        
        # Prioritize small primes? 
        # Small primes cover `gap` frequently in the future? 
        # Actually, changing a small prime affects many nearby numbers.
        # Changing a large prime affects fewer nearby numbers.
        # To maximize L, we want to "push" the failure point forward.
        # Often, fixing the immediate gap with a small prime is good because it also helps subsequent numbers.
        
        # Let's try a few random primes and some small primes.
        candidates = []
        
        # Add small primes (first 20)
        for i in range(min(20, N_PRIMES)):
            p = PRIME_LIST[i]
            if current_residues[i] != gap % p:
                candidates.append((i, gap % p))
                
        # Add random primes
        for _ in range(20):
            i = random.randint(0, N_PRIMES - 1)
            p = PRIME_LIST[i]
            target_r = gap % p
            if current_residues[i] != target_r:
                candidates.append((i, target_r))
        
        # Also try random perturbations if stuck? 
        # For now, just try the gap-fixing moves.
        
        best_move_score = current_score
        best_move_idx = -1
        best_move_r = -1
        
        for i, new_r in candidates:
            old_r = current_residues[i]
            current_residues[i] = new_r
            
            # Score with current max_check
            s = _score(current_residues, max_check_for_score)
            
            current_residues[i] = old_r
            
            if s > best_move_score:
                best_move_score = s
                best_move_idx = i
                best_move_r = new_r
        
        if best_move_idx != -1:
            # Apply the best move
            current_residues[best_move_idx] = best_move_r
            current_score = best_move_score
            
            if current_score > best_score:
                best_score = current_score
                best_residues = current_residues.copy()
        else:
            # No improvement found with gap-fixing moves.
            # Try random moves to escape local minimum.
            # Pick 5 random primes and random residues.
            for _ in range(5):
                i = random.randint(0, N_PRIMES - 1)
                p = PRIME_LIST[i]
                current_residues[i] = random.randint(0, p - 1)
            
            current_score = _score(current_residues, max_check_for_score)
            
            if current_score > best_score:
                best_score = current_score
                best_residues = current_residues.copy()
            else:
                # Revert to best? Or keep current? 
                # Hill climbing usually reverts if no improvement.
                # But we just made random moves. 
                # If we don't improve, we are stuck.
                # Let's revert to the state before the random moves to try different random moves?
                # Or just continue.
                pass

    return [int(r) for r in best_residues]
