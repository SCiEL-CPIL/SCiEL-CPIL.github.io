"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n_primes = len(primes)
    
    # Helper to check if a sequence has non-decreasing gaps
    def is_valid(seq):
        if len(seq) < 3:
            return True
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        return all(gaps[i+1] >= gaps[i] for i in range(len(gaps)-1))
        
    # Helper to get gaps
    def get_gaps(seq):
        return [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        
    # 2. Greedy Construction with Local Search
    # Strategy: 
    # Start with a small valid sequence.
    # Try to insert any remaining prime into any position to maximize length.
    # If stuck, try restarting with different random seeds or perturbations.
    
    best_sequence = []
    
    # Initial seeds: Try starting with different small primes
    # We want to find the longest possible sequence.
    # A simple greedy: Start with 2. Try to add the next prime that fits.
    # But "fits" is tricky because adding a prime changes the gap constraint for subsequent primes.
    
    # Better approach: 
    # 1. Generate a valid sequence greedily.
    #    Start with p0.
    #    Next prime p1 must be > p0.
    #    Next prime p2 must be > p1 and p2-p1 >= p1-p0.
    #    This is a standard greedy that might get stuck in local optima.
    
    # Let's try multiple random restarts with a "smart" greedy insertion.
    
    # We will maintain a set of used primes for each attempt.
    
    max_len_found = 0
    
    # Attempt 1: Deterministic Greedy
    # Start with the smallest prime.
    def greedy_build(start_prime_idx, primes_list):
        seq = [primes_list[start_prime_idx]]
        used = set(seq)
        last_gap = 0
        for p in primes_list[start_prime_idx+1:]:
            if p in used:
                continue
            new_gap = p - seq[-1]
            if new_gap >= last_gap:
                seq.append(p)
                used.add(p)
                last_gap = new_gap
        return seq

    # Try starting from different primes
    for start_idx in range(0, min(20, n_primes)):
        seq = greedy_build(start_idx, primes)
        if len(seq) > len(best_sequence):
            best_sequence = seq
            max_len_found = len(seq)

    # Attempt 2: Local Search / Insertion Heuristic
    # Try to improve the best sequence found so far by inserting unused primes.
    # We can try to insert a prime 'p' at position 'k' in the sequence 'seq'.
    # This is valid if:
    # 1. seq[k-1] < p < seq[k] (if 0 < k < len(seq))
    # 2. If k == 0: p < seq[0] and (seq[0]-p) <= (seq[1]-seq[0]) ... wait, gaps are forward.
    #    Gap before insertion: g_prev = seq[k] - seq[k-1] (if k>0)
    #    Gap after insertion: g_new1 = p - seq[k-1], g_new2 = seq[k] - p
    #    Constraints:
    #    - If k>0: g_new1 >= gap_before(k-1)
    #    - g_new2 >= g_new1
    #    - If k < len(seq)-1: gap_after(k) >= g_new2
    
    def can_insert(seq, p, k):
        # Check if p can be inserted at index k
        # k is the index where p will be placed, shifting existing elements >= k to the right.
        # So p replaces position k.
        # Left neighbor: seq[k-1] if k > 0
        # Right neighbor: seq[k] if k < len(seq)
        
        if k < 0 or k > len(seq):
            return False
            
        # Check bounds relative to neighbors
        if k > 0 and p <= seq[k-1]:
            return False
        if k < len(seq) and p >= seq[k]:
            return False
            
        # Check gap constraints
        # Let L be index of left neighbor in original seq, R be index of right neighbor.
        # If k==0, no left neighbor.
        # If k==len(seq), no right neighbor.
        
        # Gap before the inserted element (gap ending at left neighbor)
        # If k > 0, the gap ending at seq[k-1] is seq[k-1] - seq[k-2] (if k-1 > 0)
        # Actually, let's just reconstruct the gaps locally.
        
        new_seq = seq[:k] + [p] + seq[k:]
        
        # Check validity of the whole sequence is O(N). 
        # Since N is small (max ~100?), this is fine.
        # But we can be smarter.
        if not is_valid(new_seq):
            return False
        return True

    # Iterative improvement
    # We will run a loop trying to insert any unused prime into the best_sequence.
    # To avoid infinite loops or getting stuck, we can use a simple hill climbing.
    
    current_seq = best_sequence[:]
    used_set = set(current_seq)
    unused_primes = [p for p in primes if p not in used_set]
    
    improved = True
    while improved:
        improved = False
        # Try inserting each unused prime at each possible position
        # To make it faster, we can sort unused primes or just iterate.
        # Since n_primes is 1229, and seq length is likely < 50, 
        # checking 1229 * 50 = 60k insertions per pass is fast.
        
        for p in unused_primes:
            if time.time() > deadline:
                return current_seq
            
            # Try all positions
            for k in range(len(current_seq) + 1):
                if can_insert(current_seq, p, k):
                    # Perform insertion
                    current_seq = current_seq[:k] + [p] + current_seq[k:]
                    used_set.add(p)
                    unused_primes.remove(p)
                    improved = True
                    break # Restart loop as positions changed
            if improved:
                break
        
        # If no improvement in a full pass, we are stuck.
        # We could try a "swap" or "remove and reinsert" strategy, but for now,
        # let's try a random restart if we are stuck and haven't hit the time limit.
        
    if len(current_seq) > len(best_sequence):
        best_sequence = current_seq

    # Attempt 3: Randomized Restart with Greedy
    # The deterministic greedy might be suboptimal.
    # Let's try a few random restarts where we pick a random start and random valid next steps.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Pick a random start
        start_idx = random.randint(0, n_primes - 1)
        seq = [primes[start_idx]]
        used = set(seq)
        last_gap = 0
        
        # Greedily add primes
        # We can bias towards smaller gaps to allow more primes?
        # Or just pick the smallest valid next prime.
        
        for p in primes[start_idx+1:]:
            if p in used:
                continue
            new_gap = p - seq[-1]
            if new_gap >= last_gap:
                # Heuristic: Prefer smaller gaps?
                # If we pick the smallest valid gap, we might allow more future primes.
                # Let's just take the first one (smallest p) for now.
                seq.append(p)
                used.add(p)
                last_gap = new_gap
                
        if len(seq) > len(best_sequence):
            best_sequence = seq
            
    return best_sequence
# EVOLVE-BLOCK-END
