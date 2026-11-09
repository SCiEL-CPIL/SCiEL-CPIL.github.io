"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by at least 2 congruences."""
    n = len(residues)
    coverage = np.zeros(max_check + 1, dtype=np.int8)
    for i, (r, p) in enumerate(zip(residues, primes)):
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            xs = np.arange(start, max_check + 1, p)
            coverage[xs] += 1
    
    L = 0
    for x in range(1, max_check + 1):
        if coverage[x] < 2:
            L = x - 1
            break
    else:
        L = max_check
    return L

def _find_first_gap(residues, primes, start_from=1, max_check=100000):
    """Find the first integer >= start_from that is covered by < 2 congruences."""
    # We can optimize this by checking only up to a reasonable limit first, 
    # but for accuracy in the CSP loop, we need the exact first gap.
    # Since we are extending the prefix, we can start checking from start_from.
    # However, changing a residue can create a gap before start_from? 
    # No, we only change residues to fix the current gap. 
    # But changing a residue might break coverage for x < start_from?
    # If we are careful, we only accept changes that don't reduce coverage below 2 for x < start_from.
    # Actually, the CSP approach usually fixes the gap at x, potentially affecting others.
    # Let's just recompute the first gap from 1 to be safe, or from start_from if we know prefix is safe.
    # Given the cost, let's compute coverage up to max_check and find the first gap >= start_from.
    # But we must ensure 1..start_from-1 are still covered.
    # A safer approach for the inner loop:
    # 1. Identify gap at x.
    # 2. Try to fix x by changing residues that cover x or could cover x.
    # 3. Verify that 1..x are all covered.
    
    coverage = np.zeros(max_check + 1, dtype=np.int8)
    for i, (r, p) in enumerate(zip(residues, primes)):
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            xs = np.arange(start, max_check + 1, p)
            coverage[xs] += 1
            
    for x in range(start_from, max_check + 1):
        if coverage[x] < 2:
            return x
    return max_check

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Initial solution: random or heuristic
    # A good heuristic is to pick residues that are "central" or random.
    # Let's start with a random initialization and then repair.
    random.seed(42)
    best_residues = [random.randint(0, p - 1) for p in primes]
    
    # Initial scoring to see where we stand
    current_L = _score(best_residues, primes)
    best_L = current_L
    best_residues_copy = list(best_residues)
    
    # CSP / Greedy Repair Strategy
    # We iteratively find the first gap and try to fix it.
    # To fix a gap at x, we need to increase the coverage of x by 1 or 2.
    # We can do this by changing a residue r_i to x % p_i.
    # Which i to change?
    # Heuristic: Choose i such that changing r_i to x % p_i has the least negative impact elsewhere,
    # or is most likely to help.
    # A simple effective heuristic: 
    # 1. Find all primes p_i such that x % p_i != r_i (i.e., we can make it cover x).
    # 2. Among these, pick the one where the old residue r_i is "least useful" or "most redundant".
    #    Or simply pick the one with the largest prime p_i? (Larger primes cover fewer numbers, so changing them is "cheaper"?)
    #    Actually, changing a residue for a small prime affects many numbers. Changing a residue for a large prime affects few.
    #    So, prefer changing residues for large primes to minimize disruption to the existing coverage prefix.
    
    max_iterations = 5000
    gap_start = 1
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Find the first gap starting from gap_start
        # We know 1..gap_start-1 are covered (invariant)
        gap_x = _find_first_gap(best_residues, primes, start_from=gap_start)
        
        if gap_x >= 100000:
            break
            
        if gap_x <= current_L:
             # This shouldn't happen if we maintain the invariant, but if it does, reset
             gap_start = 1
             continue
             
        # We need to fix gap_x
        # Identify candidates: indices i where best_residues[i] != gap_x % primes[i]
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            target_r = gap_x % p
            if best_residues[i] != target_r:
                candidates.append(i)
                
        if not candidates:
             # All residues already point to gap_x? Impossible if gap_x is not covered.
             # Wait, if all residues point to gap_x, then coverage is 168.
             # So this case implies a bug or misunderstanding.
             # Actually, if coverage[gap_x] < 2, it means at most 1 residue points to it.
             # So there are at least 167 candidates.
             break

        # Sort candidates by prime size descending (larger primes first)
        # Rationale: Changing a large prime's residue affects fewer numbers in the prefix,
        # making it less likely to break the coverage of 1..gap_start-1.
        candidates.sort(key=lambda i: primes[i], reverse=True)
        
        found_fix = False
        # Try top K candidates
        K = 20
        for i in candidates[:K]:
            old_r = best_residues[i]
            new_r = gap_x % primes[i]
            
            # Tentatively apply
            best_residues[i] = new_r
            
            # Check if 1..gap_x are now all covered
            # We only need to check if the prefix 1..gap_x is valid.
            # We can do a quick check of coverage for 1..gap_x.
            # But _score is expensive. Let's write a faster check for prefix validity.
            # Actually, we can just recompute the first gap from 1.
            # If the new first gap is > gap_x, we have extended the prefix.
            
            new_gap = _find_first_gap(best_residues, primes, start_from=1)
            
            if new_gap > gap_x:
                # Success! We extended the prefix to at least new_gap
                # Update state
                current_L = new_gap - 1
                gap_start = new_gap
                found_fix = True
                if current_L > best_L:
                    best_L = current_L
                    best_residues_copy = list(best_residues)
                break
            else:
                # Failure, revert
                best_residues[i] = old_r
                
        if not found_fix:
            # Local search stuck? 
            # Try a random restart or a more aggressive search?
            # For now, let's try a few random changes to escape local minima.
            # Or just break if we can't fix the immediate gap with top K.
            # Let's try a few random candidates from the rest.
            random.shuffle(candidates[K:])
            for i in candidates[K:K+10]:
                old_r = best_residues[i]
                new_r = gap_x % primes[i]
                best_residues[i] = new_r
                new_gap = _find_first_gap(best_residues, primes, start_from=1)
                if new_gap > gap_x:
                    current_L = new_gap - 1
                    gap_start = new_gap
                    found_fix = True
                    if current_L > best_L:
                        best_L = current_L
                        best_residues_copy = list(best_residues)
                    break
                else:
                    best_residues[i] = old_r
            
            if not found_fix:
                # If we can't fix the gap, we might be stuck in a dead end for this prefix length.
                # In a full CSP, we would backtrack. Here, we might just stop or randomize.
                # Given the time budget, let's try a random restart of the whole process if we're stuck early.
                # Or just break.
                break

    # Final polish with local search on the best found so far
    # The CSP approach gives a strong starting point.
    # We can run a few steps of local search to see if we can push further.
    if time.time() < deadline - 10:
        residues = list(best_residues_copy)
        current_score = best_L
        
        # Simple local search: try changing residues that are "active" near the gap
        # But the gap is already handled by CSP. 
        # Maybe just try to improve by swapping residues of primes that don't affect the prefix much?
        # Actually, the CSP method is quite strong. 
        # Let's just return the best found.
        
        pass

    return best_residues_copy
# EVOLVE-BLOCK-END
