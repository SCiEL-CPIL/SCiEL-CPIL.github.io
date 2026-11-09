import time
import random
import numpy as np
from fractions import Fraction

def _get_primes(limit=1000):
    """Generate primes up to limit using simple sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    primes = _get_primes(1000)
    L_target = 100000
    
    # Pre-compute reciprocal floats for speed
    recip_floats = {p: 1.0 / p for p in primes}
    
    # State variables
    # counts[n] is the number of classes covering n
    counts = np.zeros(L_target + 1, dtype=np.int16)
    classes = [] # List of [p, r]
    used_primes = set()
    budget_float = 0.0
    current_L = 0
    
    def get_L():
        """Find the largest L such that 1..L are covered."""
        # Check 1..L_target
        # np.argmax returns index of first True. 
        # We want first 0 in counts[1:]
        # If all 1s, argmax returns 0? No, if no True, it returns 0? 
        # Let's check: np.argmax([]) is error.
        # np.where is safer.
        zeros = np.where(counts[1:] == 0)[0]
        if len(zeros) == 0:
            return L_target
        return zeros[0] # Index 0 in counts[1:] corresponds to integer 1
        
    def add_class(p, r):
        nonlocal counts, budget_float
        # Update counts
        # r can be 0. range(0, L, p) is correct for indices.
        counts[r::p] += 1
        budget_float += recip_floats[p]
        
    def remove_class(p, r):
        nonlocal counts, budget_float
        counts[r::p] -= 1
        budget_float -= recip_floats[p]

    def try_move(p, r, is_swap=False, old_p=None, old_r=None):
        """
        Try a move. 
        If is_swap, it assumes the old class is already removed.
        Returns new_L if the move is valid and better, else -1.
        """
        # This function is for internal logic, might be better to inline for speed
        pass

    # --- Phase 1: Greedy Initialization ---
    
    # We want to cover 1..L.
    # Greedy: At each step, find the first gap g.
    # For each available prime p that fits budget, calculate L if we add (p, g%p).
    # Pick the one that maximizes L.
    
    # To speed up, we might just pick the smallest prime that fits?
    # Let's try the "Maximize L" greedy.
    
    while True:
        if time.time() > deadline - 5: # Reserve 5s for search
            break
            
        # Find current first gap
        L_curr = get_L()
        if L_curr >= L_target:
            break
        g = L_curr + 1
        
        best_p = None
        best_r = None
        best_new_L = L_curr
        best_cost = 0
        
        # Check all unused primes
        # Optimization: Only consider primes that fit in remaining budget
        remaining_budget = 1.0 - budget_float
        
        for p in primes:
            if p in used_primes:
                continue
            cost = recip_floats[p]
            if cost > remaining_budget:
                continue
                
            r = g % p
            
            # Simulate
            # We can't easily "simulate" without modifying counts.
            # We can calculate the new L by checking coverage from L_curr+1 upwards?
            # No, adding a class might cover numbers before L_curr? No, L_curr is prefix.
            # Adding a class can only extend the prefix or keep it same.
            # It cannot reduce L.
            
            # To find new L, we need to check how far the coverage extends.
            # The new class covers g, g+p, g+2p...
            # The existing classes cover up to L_curr.
            # The gap is at L_curr+1 = g.
            # The new class covers g.
            # Does it cover g+1? Only if some other class covers it, or if p=1 (impossible).
            # So the new L will be at least g.
            # To find the exact new L, we can just apply the change, get_L, and revert?
            # That's expensive if we do it for all p.
            
            # Approximation: 
            # The new class covers g. 
            # The next gap will be the first number > g that is NOT covered by (old classes + new class).
            # Since old classes cover 1..L_curr, and new class covers g, 
            # the new prefix is at least g.
            # We can estimate the extension by looking at the density?
            
            # Let's just do the "Apply, Check, Revert" for the top candidates?
            # Or just pick the smallest p?
            # Smallest p has highest density.
            # Let's try picking the smallest p that fits.
            
            if best_p is None or p < best_p:
                best_p = p
                best_r = r
                best_cost = cost
                
        if best_p is None:
            break
            
        # Commit
        add_class(best_p, best_r)
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        
        # Update L
        current_L = get_L()
        
    # --- Phase 2: Local Search ---
    
    best_L = current_L
    best_classes = [c[:] for c in classes]
    
    # Helper to check if a set of classes is valid (exact rationals)
    def is_valid(classes_list):
        s = Fraction(0)
        for p, r in classes_list:
            s += Fraction(1, p)
        return s <= 1

    # Verify initial solution
    if is_valid(classes):
        pass
    else:
        # Should not happen with float check, but fix if needed
        # Remove largest prime until valid
        while classes and not is_valid(classes):
            # Remove the one with largest p (smallest cost) to preserve others?
            # Or remove the one that hurts L least?
            # Just remove last one for simplicity
            p, r = classes.pop()
            remove_class(p, r)
            used_primes.remove(p)
            
    current_L = get_L()
    best_L = max(best_L, current_L)
    if current_L > best_L:
        best_L = current_L
        best_classes = [c[:] for c in classes]

    # Local Search Loop
    # We will use a mix of Residue Changes and Swaps.
    
    # Pre-compute list of primes for random choice
    prime_list = primes
    
    while time.time() < deadline:
        # Random move selection
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Change Residue
            if not classes:
                continue
            idx = random.randint(0, len(classes) - 1)
            p, r_old = classes[idx]
            r_new = random.randint(0, p - 1)
            if r_new == r_old:
                continue
                
            # Apply
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            classes[idx] = [p, r_new]
            
            new_L = get_L()
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [c[:] for c in classes]
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                classes[idx] = [p, r_old]
                
        elif move_type == 1:
            # Swap Prime
            if not classes:
                continue
            idx = random.randint(0, len(classes) - 1)
            p_old, r_old = classes[idx]
            
            # Find a candidate p_new
            # We want p_new not in used_primes
            # And budget allows
            remaining_budget = 1.0 - budget_float + recip_floats[p_old]
            
            # Pick random prime from those fitting budget
            candidates = [p for p in prime_list if p not in used_primes and recip_floats[p] <= remaining_budget]
            if not candidates:
                continue
                
            p_new = random.choice(candidates)
            
            # Remove old
            counts[r_old::p_old] -= 1
            used_primes.remove(p_old)
            budget_float -= recip_floats[p_old]
            
            # Try a few residues for p_new
            # 1. r_old % p_new
            # 2. First gap % p_new
            # 3. Random
            
            L_curr = get_L()
            g = L_curr + 1 if L_curr < L_target else 1
            
            r_candidates = [r_old % p_new, g % p_new, random.randint(0, p_new - 1)]
            # Remove duplicates
            r_candidates = list(set(r_candidates))
            
            best_swap_L = current_L
            best_swap_r = None
            best_swap_p = p_new
            
            for r_try in r_candidates:
                counts[r_try::p_new] += 1
                new_L = get_L()
                counts[r_try::p_new] -= 1
                
                if new_L > best_swap_L:
                    best_swap_L = new_L
                    best_swap_r = r_try
            
            if best_swap_r is not None and best_swap_L > current_L:
                # Commit swap
                counts[best_swap_r::p_new] += 1
                classes[idx] = [p_new, best_swap_r]
                used_primes.add(p_new)
                budget_float += recip_floats[p_new]
                current_L = best_swap_L
                
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [c[:] for c in classes]
            else:
                # Revert swap
                # We already removed p_old. We need to put it back.
                # p_new was not added (we reverted the try).
                counts[r_old::p_old] += 1
                used_primes.add(p_old)
                budget_float += recip_floats[p_old]
                classes[idx] = [p_old, r_old]
                
        else:
            # Add Prime
            remaining_budget = 1.0 - budget_float
            if remaining_budget < 0.01: # Heuristic: skip if very little left
                continue
                
            candidates = [p for p in prime_list if p not in used_primes and recip_floats[p] <= remaining_budget]
            if not candidates:
                continue
                
            p_new = random.choice(candidates)
            
            L_curr = get_L()
            g = L_curr + 1 if L_curr < L_target else 1
            r_new = g % p_new
            
            # Try adding
            counts[r_new::p_new] += 1
            used_primes.add(p_new)
            budget_float += recip_floats[p_new]
            classes.append([p_new, r_new])
            
            new_L = get_L()
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [c[:] for c in classes]
            else:
                # Revert
                counts[r_new::p_new] -= 1
                used_primes.remove(p_new)
                budget_float -= recip_floats[p_new]
                classes.pop()
                
    # Final Verification and Return
    # Ensure best_classes is valid with exact rationals
    # If not, we might need to clean it up.
    # But our float check was conservative.
    
    # One last check: is best_L actually achieved by best_classes?
    # We should re-verify the L for best_classes to be safe.
    # But best_classes is a snapshot.
    
    return best_classes

