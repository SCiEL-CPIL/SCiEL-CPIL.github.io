"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
import time
from math import isqrt

def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Target: cover 1..L for large L
    # Use bitset representation: bit i set means integer (i+1) is covered
    # We'll work with integers 1..L_max where L_max = 100000
    
    L_target = 100000
    
    # Precompute for each prime p, the bitset of positions covered by residue r
    # For prime p, residue r covers positions n where n ≡ r (mod p), i.e., n = r, r+p, r+2p, ...
    # We need to track coverage of integers 1..L_target
    
    # Strategy: Greedy construction + local search
    # State: for each prime, either not used, or used with a specific residue
    # Budget: sum of 1/p for used primes <= 1
    
    # Represent budget as Fraction
    budget = Fraction(1, 1)
    
    # We'll use a list: selected[i] = residue or -1 if prime i not selected
    selected = [-1] * n_primes
    used_budget = Fraction(0, 1)
    
    # Coverage bitset: use Python int as bitset
    # bit (n-1) set means integer n is covered
    covered = 0
    
    # Precompute coverage masks for each prime and residue
    # mask[p_idx][r] = bitset of positions covered by prime p_idx with residue r
    # This is expensive for large L_target, so let's be smart
    
    # For local search, we need to quickly evaluate the effect of changing a prime's residue
    # Let's work with a smaller initial target and expand
    
    # Actually, let's use a different approach:
    # 1. Greedily select primes and residues to maximize coverage prefix
    # 2. Then try to extend the prefix by adjusting residues
    
    # Greedy phase:
    # For each step, pick the (prime, residue) that maximizes new coverage in the prefix
    # while maintaining budget
    
    # To make this efficient, let's track the uncovered prefix
    # and for each candidate (p, r), count how many new integers it covers in the prefix
    
    # Let's use numpy for speed
    import numpy as np
    
    # Work with a reasonable L for the bitset operations
    # We'll track coverage up to L_work = 2000 initially, then verify
    
    L_work = 2000
    
    # coverage array: 1 if covered, 0 if not
    cov = np.zeros(L_work, dtype=np.uint8)
    
    # For each prime, precompute the pattern of coverage for each residue
    # pattern[p_idx][r] = boolean array of length L_work
    # This is too much memory. Instead, compute on the fly.
    
    # Better: for each prime p, the coverage pattern for residue r is:
    # positions n where (n-1) % p == (r-1) % p, i.e., n ≡ r (mod p)
    # In 0-indexed: position i (integer i+1) is covered if (i+1) % p == r % p
    # i.e., i % p == (r-1) % p
    
    # Let's use a simpler approach:
    # Greedily add primes, choosing the residue that maximizes coverage of the current uncovered prefix
    
    # Track the first uncovered position
    def first_uncovered():
        # Find first index where cov is 0
        idx = np.nonzero(cov == 0)[0]
        if len(idx) == 0:
            return L_work
        return idx[0] + 1  # 1-indexed
    
    # Greedy construction
    # Available primes: all primes not yet selected
    available = list(range(n_primes))
    
    # Sort primes by size - smaller primes give more coverage per unit budget
    # But we want to maximize prefix coverage, so let's be greedy
    
    # For efficiency, let's limit the number of iterations
    max_greedy_steps = 200
    step = 0
    while available and step < max_greedy_steps:
        if time.time() > deadline - 30:
            break
        
        best_gain = -1
        best_p_idx = -1
        best_r = -1
        best_budget = Fraction(0,1)
        
        fu = first_uncovered()
        if fu >= L_work:
            break
        
        # For each available prime, find the best residue
        for pi in available:
            p = primes[pi]
            if used_budget + Fraction(1, p) > budget:
                continue
            
            # Try all residues and find the one with max gain
            # Gain = number of newly covered positions in [1, fu]
            # Position i (0-indexed) is covered if (i+1) % p == r
            # i.e., i % p == (r-1) % p
            
            # For a given residue r, the covered positions are:
            # i where i % p == (r-1) % p, i.e., i = (r-1) % p, (r-1)%p + p, ...
            # up to L_work-1
            
            # We want to maximize coverage of positions [0, fu-1] that are currently uncovered
            # Let's compute gain for each residue
            
            # To speed up, note that the pattern repeats every p
            # In the range [0, fu-1], there are about fu/p full cycles
            # Each cycle has exactly 1 position covered for a given residue
            
            # The gain depends on which positions are uncovered
            # Let's just compute it directly for the best few residues
            
            # Actually, let's compute gain for all residues efficiently
            # For residue r, covered positions in [0, fu-1]:
            # start = (r - 1) % p
            # positions: start, start+p, start+2p, ... < fu
            
            # Gain = count of these positions where cov[pos] == 0
            
            # Let's vectorize this
            # For each residue r, the positions are (r-1)%p + k*p for k=0,1,...
            
            # This is still O(L_work/p) per residue, which is fine for small p
            
            # Let's try a subset of residues for large p
            if p > 100:
                # Try a few residues
                candidates = [0, 1, fu % p, (fu-1) % p]
                candidates = list(set(candidates))
            else:
                candidates = list(range(p))
            
            for r in candidates:
                start = (r - 1) % p
                if start >= fu:
                    continue
                # Positions: start, start+p, start+2p, ... < fu
                positions = list(range(start, fu, p))
                if not positions:
                    continue
                gain = np.sum(cov[positions] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_p_idx = pi
                    best_r = r
                    best_budget = Fraction(1, p)
        
        if best_p_idx == -1:
            break
        
        # Apply the selection
        selected[best_p_idx] = best_r
        used_budget += best_budget
        available.remove(best_p_idx)
        
        # Update coverage
        p = primes[best_p_idx]
        r = best_r
        start = (r - 1) % p
        for pos in range(start, L_work, p):
            cov[pos] = 1
        
        step += 1
    
    # Now we have a greedy solution. Let's try to improve with local search.
    # Local search: try changing residues of selected primes, or swapping primes
    
    def compute_coverage(selected_arr):
        """Compute coverage bitset from selected array."""
        cov_new = np.zeros(L_work, dtype=np.uint8)
        for pi, r in enumerate(selected_arr):
            if r < 0:
                continue
            p = primes[pi]
            start = (r - 1) % p
            for pos in range(start, L_work, p):
                cov_new[pos] = 1
        return cov_new
    
    def prefix_length(cov_arr):
        """Find the length of the covered prefix."""
        # Find first 0
        zeros = np.nonzero(cov_arr == 0)[0]
        if len(zeros) == 0:
            return L_work
        return zeros[0]  # 0-indexed, so prefix length is zeros[0]
    
    current_prefix = prefix_length(cov)
    
    # Local search phase
    import random
    random.seed(42)
    
    best_prefix = current_prefix
    best_selected = selected[:]
    
    # Try to improve by:
    # 1. Changing residue of a selected prime
    # 2. Swapping a selected prime with an unselected one
    # 3. Adding a new prime if budget allows
    # 4. Removing a prime to free budget
    
    local_search_iter = 0
    max_local_iter = 50000
    
    while local_search_iter < max_local_iter:
        if time.time() > deadline - 10:
            break
        
        local_search_iter += 1
        
        # Randomly pick an operation
        op = random.randint(0, 3)
        
        if op == 0:  # Change residue of a selected prime
            sel_indices = [i for i in range(n_primes) if selected[i] >= 0]
            if not sel_indices:
                continue
            pi = random.choice(sel_indices)
            p = primes[pi]
            old_r = selected[pi]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Evaluate
            selected[pi] = new_r
            cov_new = compute_coverage(selected)
            new_prefix = prefix_length(cov_new)
            
            if new_prefix > best_prefix:
                best_prefix = new_prefix
                best_selected = selected[:]
                cov = cov_new
                current_prefix = new_prefix
            else:
                selected[pi] = old_r
        
        elif op == 1:  # Swap: remove one selected, add one unselected
            sel_indices = [i for i in range(n_primes) if selected[i] >= 0]
            unsel_indices = [i for i in range(n_primes) if selected[i] < 0]
            if not sel_indices or not unsel_indices:
                continue
            pi_remove = random.choice(sel_indices)
            pi_add = random.choice(unsel_indices)
            
            p_remove = primes[pi_remove]
            p_add = primes[pi_add]
            
            new_budget = used_budget - Fraction(1, p_remove) + Fraction(1, p_add)
            if new_budget > budget:
                continue
            
            # Try
            old_r_remove = selected[pi_remove]
            selected[pi_remove] = -1
            selected[pi_add] = random.randint(0, p_add - 1)
            used_budget = new_budget
            
            cov_new = compute_coverage(selected)
            new_prefix = prefix_length(cov_new)
            
            if new_prefix > best_prefix:
                best_prefix = new_prefix
                best_selected = selected[:]
                cov = cov_new
                current_prefix = new_prefix
            else:
                selected[pi_remove] = old_r_remove
                selected[pi_add] = -1
                used_budget = used_budget - Fraction(1, p_add) + Fraction(1, p_remove)
        
        elif op == 2:  # Add a new prime
            unsel_indices = [i for i in range(n_primes) if selected[i] < 0]
            if not unsel_indices:
                continue
            pi_add = random.choice(unsel_indices)
            p_add = primes[pi_add]
            
            if used_budget + Fraction(1, p_add) > budget:
                continue
            
            selected[pi_add] = random.randint(0, p_add - 1)
            used_budget += Fraction(1, p_add)
            
            cov_new = compute_coverage(selected)
            new_prefix = prefix_length(cov_new)
            
            if new_prefix > best_prefix:
                best_prefix = new_prefix
                best_selected = selected[:]
                cov = cov_new
                current_prefix = new_prefix
            else:
                selected[pi_add] = -1
                used_budget -= Fraction(1, p_add)
        
        elif op == 3:  # Remove a prime
            sel_indices = [i for i in range(n_primes) if selected[i] >= 0]
            if not sel_indices:
                continue
            pi_remove = random.choice(sel_indices)
            p_remove = primes[pi_remove]
            
            old_r = selected[pi_remove]
            selected[pi_remove] = -1
            used_budget -= Fraction(1, p_remove)
            
            cov_new = compute_coverage(selected)
            new_prefix = prefix_length(cov_new)
            
            if new_prefix > best_prefix:
                best_prefix = new_prefix
                best_selected = selected[:]
                cov = cov_new
                current_prefix = new_prefix
            else:
                selected[pi_remove] = old_r
                used_budget += Fraction(1, p_remove)
    
    # Use the best solution found
    final_selected = best_selected
    
    # Verify and compute the actual prefix length
    # We need to check beyond L_work if possible
    # First, compute coverage up to a larger range to verify
    
    # Build the list of (p, r) pairs
    result = []
    total_budget = Fraction(0, 1)
    for pi, r in enumerate(final_selected):
        if r >= 0:
            p = primes[pi]
            result.append([p, r])
            total_budget += Fraction(1, p)
    
    # Verify budget
    if total_budget > budget:
        # This shouldn't happen, but just in case
        pass
    
    # Now compute the actual covered prefix length
    # We need to check all integers 1..L for coverage
    # Use a set of (p, r) for fast lookup
    
    # For efficiency, let's compute coverage up to min(L_work, 100000)
    # and find the first uncovered integer
    
    # Use the bitset approach with Python integers
    # For each (p, r), mark positions n where n % p == r % p
    
    # Actually, let's just check sequentially
    # For each integer n from 1 upward, check if it's covered by any (p, r)
    
    covered_set = set()
    for p, r in result:
        # n ≡ r (mod p) means n = r + k*p for k = 0, 1, 2, ...
        # n >= 1
        start = r if r >= 1 else p  # smallest positive n with n ≡ r (mod p)
        # Actually if r == 0, n = p, 2p, ...; if r > 0, n = r, r+p, ...
        if r == 0:
            start = p
        else:
            start = r
        
        for n in range(start, 100001, p):
            covered_set.add(n)
    
    # Find the largest L such that 1..L are all in covered_set
    L = 0
    for n in range(1, 100001):
        if n in covered_set:
            L = n
        else:
            break
    
    return result

# EVOLVE-BLOCK-END
