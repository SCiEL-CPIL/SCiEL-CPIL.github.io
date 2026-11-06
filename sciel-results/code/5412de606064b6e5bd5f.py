"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _sieve(1000)
    N = 100000
    
    # Precompute the indices for each (p, r) to allow fast delta updates
    # indices[p][r] is a numpy array of indices in 1..N such that i % p == r
    indices = {}
    for p in primes:
        indices[p] = {}
        for r in range(p):
            start = r if r > 0 else p
            if start <= N:
                indices[p][r] = np.arange(start, N + 1, p)
            else:
                indices[p][r] = np.array([], dtype=int)
                
    # Helper to calculate L given a coverage array
    def get_L(covered):
        # covered is a numpy array of uint8, length N+1
        # We want the largest L such that covered[1..L] are all 1
        # We can find the first 0 in covered[1:]
        # If no 0, L = N
        # Use argmax on (covered[1:] == 0)
        # If all 1s, (covered[1:] == 0) is all False, argmax returns 0.
        # But we need to distinguish "all 1s" from "first gap at 1".
        # If covered[1] is 0, gap is at 1.
        
        # Efficient way:
        # Find indices where covered is 0.
        # We only care about the first one >= 1.
        
        # Using numpy:
        # zeros = np.flatnonzero(covered[1:] == 0)
        # if len(zeros) == 0: return N
        # else: return zeros[0]
        
        # flatnonzero is fast.
        zeros = np.flatnonzero(covered[1:] == 0)
        if len(zeros) == 0:
            return N
        return zeros[0]

    def select_primes_greedy(min_p=2):
        sel = []
        s = Fraction(0, 1)
        for p in primes:
            if p < min_p:
                continue
            cost = Fraction(1, p)
            if s + cost <= Fraction(1, 1):
                sel.append(p)
                s += cost
        return sel, s

    # We will try a few different prime sets
    # Set 1: Greedy from 2
    # Set 2: Greedy from 5 (skip 2, 3 to maybe save budget for more primes? No, 2 and 3 are very efficient for coverage per cost)
    # Actually, 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 ...
    # Sum of 1/p for primes up to 29 is approx 1.0?
    # 1/2=0.5, 1/3=0.333 (0.833), 1/5=0.2 (1.033 > 1).
    # So we can't have 2,3,5.
    # We can have 2,3 (0.833). Remaining budget 0.166.
    # Next prime 5 (0.2) doesn't fit.
    # Next prime 7 (0.142) fits. 0.833 + 0.142 = 0.975.
    # Next prime 11 (0.09) doesn't fit.
    # So set {2, 3, 7} is one option.
    
    # What if we skip 3?
    # 1/2 = 0.5.
    # 1/5 = 0.2 -> 0.7.
    # 1/7 = 0.142 -> 0.842.
    # 1/11 = 0.09 -> 0.932.
    # 1/13 = 0.076 -> 1.008 (No).
    # So {2, 5, 7, 11} is another option.
    
    # Let's generate a few candidate sets based on greedy with different starting points or exclusions.
    # However, the residue assignment is more critical.
    
    # Strategy:
    # 1. Generate a base set of primes using greedy, but allow some flexibility.
    # 2. For each prime, determine the best residue greedily.
    # 3. Run a fast local search with incremental updates.

    # Let's define a few prime sets to try
    prime_sets = []
    
    # Set A: Greedy 2,3,7...
    sel, _ = select_primes_greedy(2)
    prime_sets.append(sel)
    
    # Set B: Greedy 2,5,7,11... (skip 3)
    sel, _ = select_primes_greedy(2)
    # Manually construct a set skipping 3
    sel_b = []
    s = Fraction(0,1)
    for p in primes:
        if p == 3: continue
        if s + Fraction(1,p) <= 1:
            sel_b.append(p)
            s += Fraction(1,p)
    if sel_b: prime_sets.append(sel_b)
    
    # Set C: Greedy 5,7,11... (skip 2,3)
    sel, _ = select_primes_greedy(5)
    if sel: prime_sets.append(sel)
    
    # Set D: Greedy 2,3,5... (Wait, 2+3+5 > 1, so greedy stops at 3 if 5 doesn't fit)
    # Greedy 2: 1/2. Next 3: 1/2+1/3=5/6. Next 5: 5/6+1/5=31/30 > 1.
    # So greedy(2) gives [2,3].
    
    # Let's also try a set that includes 2 but skips 3 to fit more primes?
    # 2 (0.5), 5 (0.2), 7 (0.14), 11 (0.09), 13 (0.07) -> 0.5+0.2+0.14+0.09+0.07 = 1.0.
    # This set has more primes, which might help cover gaps better.
    
    best_L = 0
    best_pairs = []
    
    for ps in prime_sets:
        if not ps: continue
        
        # Greedy Initialization of Residues
        # Start with empty coverage
        covered = np.zeros(N + 1, dtype=np.uint8)
        current_L = 0
        current_pairs = {}
        
        # For each prime, find the residue that maximizes L
        # This is O(|ps| * p * N/p) = O(|ps| * N) which is okay for initialization.
        # But we can do it faster:
        # For a prime p, we want to place it to cover the first gap.
        
        # Let's just do a simple greedy:
        # Add primes one by one. For each new prime, try all residues and pick the one that gives max L.
        
        covered = np.zeros(N + 1, dtype=np.uint8)
        L = 0
        assigned_primes = []
        
        # Sort primes? Smaller primes have more residues but are "denser".
        # It's usually better to place small primes first to establish a base, or large primes first to fill gaps?
        # Small primes cover large chunks. Let's try adding in order of size.
        
        for p in ps:
            best_r = 0
            best_L_for_p = 0
            
            # Try all residues for this prime
            # We can optimize by only checking residues that might affect the current gap
            # But N=10^5 is small enough that even O(p * N/p) = O(N) per prime is fine.
            # Total init cost: sum(N) for each prime = |ps| * N.
            # |ps| is small (<= 10). 10 * 10^5 = 10^6 ops. Very fast.
            
            for r in range(p):
                # Add this residue to a temporary copy? No, that's expensive.
                # We can calculate the new L by simulating the addition.
                # But we need to know if it extends the current L.
                
                # Let's just do a full update on a copy for simplicity in init, it's fast enough.
                temp_covered = covered.copy()
                idx = indices[p][r]
                if len(idx) > 0:
                    temp_covered[idx] = 1
                new_L = get_L(temp_covered)
                
                if new_L > best_L_for_p:
                    best_L_for_p = new_L
                    best_r = r
            
            # Commit the best residue
            covered[indices[p][best_r]] = 1
            L = best_L_for_p
            assigned_primes.append((p, best_r))
            
        # Now we have a good starting point.
        if L > best_L:
            best_L = L
            best_pairs = assigned_primes.copy()
            
        # Local Search with Incremental Updates
        # We maintain `covered` and `L`.
        # Move: change residue of a prime p from r_old to r_new.
        # Update: covered[indices[p][r_old]] -= 1 (but we use uint8, so we need to be careful if multiple primes cover same index)
        # Wait, the constraint is "every integer lies in SOME class".
        # If we use a count array, we can handle overlaps.
        # But `covered` as bool is not enough for removal.
        # We MUST use a count array `cover_count` for incremental updates.
        
        # Let's re-initialize with a count array for the local search phase.
        cover_count = np.zeros(N + 1, dtype=np.uint8)
        # Re-apply the assigned primes to build cover_count
        for p, r in assigned_primes:
            idx = indices[p][r]
            if len(idx) > 0:
                cover_count[idx] += 1
        
        # Function to get L from cover_count
        def get_L_count(cc):
            # L is the largest such that cc[1..L] > 0
            # Find first index i >= 1 where cc[i] == 0
            zeros = np.flatnonzero(cc[1:] == 0)
            if len(zeros) == 0:
                return N
            return zeros[0]
            
        current_L = get_L_count(cover_count)
        
        # Local Search Loop
        # We will do a random walk that accepts moves if L increases, or with some probability if L stays same/decreases slightly.
        # Given the time budget, we can do many iterations.
        
        T = 1.0
        cooling = 0.9999
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5:
            # Pick a random prime index
            idx = random.randint(0, len(assigned_primes) - 1)
            p, old_r = assigned_primes[idx]
            
            # Pick a new random residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Apply change
            idx_old = indices[p][old_r]
            idx_new = indices[p][new_r]
            
            # Update counts
            if len(idx_old) > 0:
                cover_count[idx_old] -= 1
            if len(idx_new) > 0:
                cover_count[idx_new] += 1
                
            # Calculate new L
            new_L = get_L_count(cover_count)
            
            # Acceptance criteria
            if new_L > current_L:
                # Accept
                assigned_primes[idx] = (p, new_r)
                current_L = new_L
                no_improve = 0
                if current_L > best_L:
                    best_L = current_L
                    best_pairs = assigned_primes.copy()
            else:
                # Reject or accept with probability
                # If new_L == current_L, accept to wander
                # If new_L < current_L, accept with exp((new_L - current_L)/T)
                
                delta = new_L - current_L
                if delta == 0:
                    # Accept to explore
                    assigned_primes[idx] = (p, new_r)
                    # current_L unchanged
                elif delta < 0:
                    # Probabilistic acceptance
                    if random.random() < np.exp(delta / T):
                        assigned_primes[idx] = (p, new_r)
                        current_L = new_L
                    else:
                        # Revert
                        if len(idx_new) > 0:
                            cover_count[idx_new] -= 1
                        if len(idx_old) > 0:
                            cover_count[idx_old] += 1
                        # current_L unchanged (reverted to old state)
                
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                    
            T *= cooling
            if T < 0.01:
                T = 0.01
                # Optional: restart with a different prime set or randomization if stuck
                # For now, just keep going with low T (greedy)
                
    return best_pairs

# EVOLVE-BLOCK-END
