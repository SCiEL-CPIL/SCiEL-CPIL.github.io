"""Covering 1..L by prime residue classes with reciprocal budget 1 (primes up to 1000)."""
import time
import random
import numpy as np
from fractions import Fraction

MAXL = 100000
DEADLINE = 135.0  # safety margin before the 150s kill


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = 0
    return [i for i in range(2, n + 1) if sieve[i]]


def construct():
    t0 = time.time()
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}
    
    # We will use 0-based indexing for numpy arrays, so position x corresponds to index x-1.
    # cnt[x-1] > 0 means x is covered.
    
    def get_L(cnt):
        # cnt is a numpy array of shape (MAXL,)
        # We want the largest L such that cnt[0..L-1] are all > 0.
        # np.argmax(cnt == 0) gives the first zero. If no zero, it returns 0.
        # We need to handle the case where all are covered up to MAXL.
        zeros = np.where(cnt == 0)[0]
        if len(zeros) == 0:
            return MAXL
        first_zero_idx = zeros[0]
        # Index is 0-based, so if first zero is at index k, the covered length is k.
        # Because indices 0..k-1 are covered, which correspond to 1..k.
        return first_zero_idx

    def add_class(cnt, p, r):
        # r is in [0, p-1]. We cover numbers x such that x % p == r.
        # In 0-based array index i = x-1.
        # x = r, r+p, r+2p, ...
        # If r == 0, x = p, 2p, ... (since x>=1). Index i = p-1, 2p-1, ...
        # If r > 0, x = r, r+p, ... Index i = r-1, r+p-1, ...
        
        if r == 0:
            start_idx = p - 1
        else:
            start_idx = r - 1
            
        if start_idx < MAXL:
            cnt[start_idx::p] += 1

    def remove_class(cnt, p, r):
        if r == 0:
            start_idx = p - 1
        else:
            start_idx = r - 1
            
        if start_idx < MAXL:
            cnt[start_idx::p] -= 1

    def calculate_L(cnt):
        return get_L(cnt)

    # State: list of (p, r)
    # Budget: Fraction
    # cnt: numpy array
    
    def greedy_fill(state_dict, budget, cnt, L):
        """
        Greedily add primes to cover the next gap.
        state_dict: {p: r}
        """
        if time.time() > DEADLINE - 20:
            return state_dict, budget, L
            
        while time.time() < DEADLINE - 20:
            # Find the first gap
            if L >= MAXL:
                break
            gap = L + 1
            
            # Find a prime to cover this gap
            # Candidates: primes not in state, or primes in state with wrong residue
            # We prefer small primes (cheap, dense)
            
            best_p = None
            best_r = None
            best_score = -1
            
            # Try all primes <= 1000
            # Sorting by prime value ensures we check small ones first
            for p in primes:
                if p > gap:
                    # If p > gap, we can only set r = gap % p = gap.
                    # But wait, if p > gap, gap % p = gap.
                    # The class covers gap, 2p+gap, etc.
                    # This is a valid option.
                    pass
                
                # Check if we can add or change
                if p in state_dict:
                    current_r = state_dict[p]
                    if current_r == gap % p:
                        # Already covers it? Shouldn't happen if L is correct gap.
                        continue
                    
                    # Changing residue is free in budget, but might uncover previous parts.
                    # For greedy, let's only consider adding new primes or swapping if we get stuck.
                    # Actually, changing residue of an existing prime is powerful.
                    # But let's start by just adding new primes.
                    continue
                
                # Check budget
                if budget < recip[p]:
                    continue
                
                # We can add p with r = gap % p
                r = gap % p
                
                # Heuristic: prefer smaller p
                # Score: smaller p is better.
                score = -p
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
                    
            if best_p is None:
                # No new prime can be added. Try swapping residues?
                # Or break.
                break
            
            # Commit the addition
            state_dict[best_p] = best_r
            budget -= recip[best_p]
            add_class(cnt, best_p, best_r)
            
            # Update L
            new_L = calculate_L(cnt)
            if new_L <= L:
                # This shouldn't happen if we covered the gap, unless we uncovered something before L?
                # Adding a class only increases counts, so it cannot uncover anything.
                # So new_L should be >= L.
                pass
            L = new_L
            
    def local_search(state_dict, budget, cnt, L):
        """
        Improve L by swapping residues or replacing primes.
        """
        if time.time() > DEADLINE - 10:
            return state_dict, budget, L
            
        rng = random.Random(42)
        no_improve_count = 0
        max_iterations = 1000
        iter_count = 0
        
        while iter_count < max_iterations and time.time() < DEADLINE - 10:
            iter_count += 1
            if L >= MAXL:
                break
                
            gap = L + 1
            improved = False
            
            # Strategy 1: Change residue of an existing prime to cover the gap
            # Pick a random prime from state
            if state_dict:
                p = rng.choice(list(state_dict.keys()))
                old_r = state_dict[p]
                new_r = gap % p
                
                if old_r != new_r:
                    # Try the swap
                    remove_class(cnt, p, old_r)
                    add_class(cnt, p, new_r)
                    state_dict[p] = new_r
                    
                    new_L = calculate_L(cnt)
                    if new_L > L:
                        L = new_L
                        improved = True
                        no_improve_count = 0
                    else:
                        # Revert
                        remove_class(cnt, p, new_r)
                        add_class(cnt, p, old_r)
                        state_dict[p] = old_r
                        no_improve_count += 1
                        
            # Strategy 2: Replace a large prime with a smaller unused one
            if not improved and time.time() < DEADLINE - 10:
                # Find a large prime in state
                large_primes_in_state = [p for p in state_dict.keys() if p > 100]
                unused_primes = [p for p in primes if p not in state_dict and budget >= recip[p]]
                
                if large_primes_in_state and unused_primes:
                    # Pick a random large prime to remove
                    p_remove = rng.choice(large_primes_in_state)
                    # Pick a random small unused prime to add
                    p_add = rng.choice(unused_primes)
                    
                    # Only proceed if budget allows
                    if budget + recip[p_remove] >= recip[p_add]:
                        old_r_remove = state_dict[p_remove]
                        
                        # Remove p_remove
                        remove_class(cnt, p_remove, old_r_remove)
                        del state_dict[p_remove]
                        budget += recip[p_remove]
                        
                        # Add p_add with residue to cover gap
                        r_add = gap % p_add
                        add_class(cnt, p_add, r_add)
                        state_dict[p_add] = r_add
                        budget -= recip[p_add]
                        
                        new_L = calculate_L(cnt)
                        if new_L > L:
                            L = new_L
                            improved = True
                            no_improve_count = 0
                        else:
                            # Revert
                            remove_class(cnt, p_add, r_add)
                            del state_dict[p_add]
                            budget += recip[p_add]
                            
                            add_class(cnt, p_remove, old_r_remove)
                            state_dict[p_remove] = old_r_remove
                            budget -= recip[p_remove]
                            no_improve_count += 1
                            
            if no_improve_count > 50:
                break
                
        return state_dict, budget, L

    # Main execution
    best_state = None
    best_budget = Fraction(1, 1)
    best_L = 0
    
    # Run multiple greedy passes with different randomizations?
    # Greedy is deterministic here (always picks smallest p). 
    # To diversify, we can shuffle the order of primes we consider, 
    # but the heuristic "smallest p" is strong. 
    # Let's try a few variations.
    
    for seed in range(5):
        if time.time() > DEADLINE - 30:
            break
            
        state_dict = {}
        budget = Fraction(1, 1)
        cnt = np.zeros(MAXL, dtype=np.int32)
        L = 0
        
        # Greedy phase
        state_dict, budget, L = greedy_fill(state_dict, budget, cnt, L)
        
        # Local search phase
        state_dict, budget, L = local_search(state_dict, budget, cnt, L)
        
        if L > best_L:
            best_L = L
            best_state = dict(state_dict)
            best_budget = budget
            
    if best_state is None:
        # Fallback: just return something valid
        # Use small primes
        result = []
        budget = Fraction(1, 1)
        for p in primes:
            if budget >= recip[p]:
                result.append([p, 0])
                budget -= recip[p]
        return result
        
    result = [[p, r] for p, r in best_state.items()]
    
    # Final verification
    total = Fraction(0, 1)
    for p, _ in result:
        total += Fraction(1, p)
    if total > Fraction(1, 1):
        # Fallback
        result = []
        budget = Fraction(1, 1)
        for p in primes:
            if budget >= recip[p]:
                result.append([p, 0])
                budget -= recip[p]
                
    return result
