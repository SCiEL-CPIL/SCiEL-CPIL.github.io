import time
import numpy as np
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    
    LIMIT = 100000
    # We are interested in L up to LIMIT.
    # Using numpy boolean array for coverage.
    # Index 0 is unused, we care about 1..LIMIT.
    
    # Beam Search Parameters
    BEAM_WIDTH = 50
    MAX_PRIMES_TO_TRY = 30  # Per state, how many primes to consider
    
    # State: (covered_array, used_primes_set, remaining_budget_fraction, current_L)
    # To save memory and time, we might store covered as a tuple of bytes or just numpy array.
    # Since beam width is small, storing numpy arrays is fine.
    
    # Initial state
    # covered: False everywhere
    init_covered = np.zeros(LIMIT + 1, dtype=bool)
    init_state = {
        'covered': init_covered,
        'used': set(),
        'budget': Fraction(1, 1),
        'L': 0
    }
    
    beam = [init_state]
    best_L = 0
    best_state = None
    
    # Precompute prime costs as Fractions for exact comparison
    prime_costs = {p: Fraction(1, p) for p in primes}
    
    while time.time() < deadline:
        next_beam = []
        
        for state in beam:
            if time.time() > deadline:
                break
            
            L = state['L']
            if L >= LIMIT:
                # Already covered everything, can't get better
                next_beam.append(state)
                continue
            
            covered = state['covered']
            used = state['used']
            budget = state['budget']
            
            # Find first uncovered number
            # We know 1..L are covered. So first uncovered is L+1.
            # But wait, if we are in a beam search, do we guarantee 1..L are covered?
            # Yes, if we define L as the largest k such that 1..k are covered.
            # So we just look at n = L + 1.
            
            n = L + 1
            if n > LIMIT:
                continue
            
            # Generate candidates
            candidates = []
            
            # We want to pick primes that cover n and extend L as far as possible.
            # To limit computation, we only check a subset of primes.
            # Heuristic: Smaller primes cover more numbers, so they are more likely to extend L further.
            # But larger primes might be needed if smaller ones are used.
            # Let's iterate through primes, but skip those already used.
            # To be fast, we can just take the first MAX_PRIMES_TO_TRY available primes?
            # Or sort by potential?
            # Let's just iterate all primes, but break if we have enough candidates?
            # Actually, checking all 168 primes is not that slow if the simulation is fast.
            # But simulation involves numpy operations.
            
            # Let's try all available primes, but limit the number of candidates we keep per state
            # to the top few based on a quick score?
            # No, just generate all valid (p, r) pairs and simulate.
            
            for p in primes:
                if p in used:
                    continue
                cost = prime_costs[p]
                if budget < cost:
                    continue
                
                r = n % p
                
                # Simulate adding (p, r)
                # new_covered = covered.copy()
                # new_covered[r::p] = True
                # But we only need to find the new L.
                # We know 1..L are covered.
                # We need to check L+1, L+2, ...
                
                # Optimization:
                # Check if n is covered by (p, r). Yes, by definition.
                # Check n+1, n+2...
                
                # Using numpy:
                # Create a temporary view or copy?
                # Copying 100k bools is fast (~100KB).
                # Let's do it.
                
                new_covered = covered.copy()
                # Mark r, r+p, r+2p... as True
                # Note: indices in numpy are 0-based.
                # If r==0, indices are 0, p, 2p...
                # If r>0, indices are r, r+p...
                # We care about 1..LIMIT.
                # new_covered[r::p] = True works for 0-based.
                # If r=0, it marks 0, p, 2p...
                # If r>0, it marks r, r+p...
                
                new_covered[r::p] = True
                
                # Find new L
                # We know 1..L are covered in new_covered (since they were in covered).
                # We need to find the first False in new_covered starting from L+1.
                
                # Slice from L+1 to end
                # If all True, new_L = LIMIT
                # Else, new_L = index of first False - 1
                
                # np.argmax returns the first True. We want first False.
                # ~new_covered gives True where it was False.
                # But we only care from L+1 onwards.
                
                # Optimization: Only check up to a certain limit?
                # No, we want max L.
                
                # To be faster, we can check in chunks?
                # Or just use np.argmax on the slice.
                
                # Slice: new_covered[L+1:]
                # If this slice is all True, new_L = LIMIT.
                # Else, find first False.
                
                # Using np.where(~new_covered[L+1:], True, False)
                # This creates a temporary array.
                
                # Alternative:
                # Loop in Python? No, too slow.
                
                # Let's use numpy.
                remaining = ~new_covered[L+1:]
                if not np.any(remaining):
                    new_L = LIMIT
                else:
                    first_gap_offset = np.argmax(remaining)
                    new_L = L + 1 + first_gap_offset - 1
                    new_L = L + first_gap_offset
                    
                # Score: new_L
                # We also might want to prefer states with more remaining budget?
                # For now, maximize new_L.
                
                candidates.append((new_L, p, r, new_covered, budget - cost))
                
            # Sort candidates by new_L descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top K candidates from this state to add to next_beam
            # This prevents beam from exploding
            top_candidates = candidates[:MAX_PRIMES_TO_TRY]
            
            for new_L, p, r, new_covered, new_budget in top_candidates:
                if new_L > best_L:
                    best_L = new_L
                    best_state = {
                        'covered': new_covered,
                        'used': used | {p},
                        'budget': new_budget,
                        'L': new_L
                    }
                
                next_beam.append({
                    'covered': new_covered,
                    'used': used | {p},
                    'budget': new_budget,
                    'L': new_L
                })
        
        if not next_beam:
            break
            
        # Sort next_beam by L descending and keep top BEAM_WIDTH
        next_beam.sort(key=lambda x: x['L'], reverse=True)
        beam = next_beam[:BEAM_WIDTH]
        
        # If best_L is very high, we might want to stop early?
        # But we want to maximize.
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    if best_state is None:
        return []
        
    # Extract classes from best_state
    # We didn't store the list of classes in the state, only the set of used primes.
    # We need the residues too.
    # This is a problem. We need to reconstruct the classes.
    
    # We should store the list of classes in the state.
    # Let's restart the logic with that fix.
    
    pass

# Revised Implementation with Class Storage

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    LIMIT = 100000
    
    BEAM_WIDTH = 50
    MAX_PRIMES_TO_TRY = 30
    
    prime_costs = {p: Fraction(1, p) for p in primes}
    
    init_covered = np.zeros(LIMIT + 1, dtype=bool)
    init_state = {
        'covered': init_covered,
        'used': set(),
        'budget': Fraction(1, 1),
        'L': 0,
        'classes': []
    }
    
    beam = [init_state]
    best_L = 0
    best_classes = []
    
    while time.time() < deadline:
        next_beam = []
        
        for state in beam:
            if time.time() > deadline:
                break
            
            L = state['L']
            if L >= LIMIT:
                next_beam.append(state)
                continue
            
            covered = state['covered']
            used = state['used']
            budget = state['budget']
            classes = state['classes']
            
            n = L + 1
            if n > LIMIT:
                continue
            
            candidates = []
            
            for p in primes:
                if p in used:
                    continue
                cost = prime_costs[p]
                if budget < cost:
                    continue
                
                r = n % p
                
                new_covered = covered.copy()
                new_covered[r::p] = True
                
                remaining = ~new_covered[L+1:]
                if not np.any(remaining):
                    new_L = LIMIT
                else:
                    first_gap_offset = np.argmax(remaining)
                    new_L = L + first_gap_offset
                    
                candidates.append((new_L, p, r, new_covered, budget - cost))
                
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = candidates[:MAX_PRIMES_TO_TRY]
            
            for new_L, p, r, new_covered, new_budget in top_candidates:
                if new_L > best_L:
                    best_L = new_L
                    best_classes = classes + [[p, r]]
                
                next_beam.append({
                    'covered': new_covered,
                    'used': used | {p},
                    'budget': new_budget,
                    'L': new_L,
                    'classes': classes + [[p, r]]
                })
        
        if not next_beam:
            break
            
        next_beam.sort(key=lambda x: x['L'], reverse=True)
        beam = next_beam[:BEAM_WIDTH]
        
        if time.time() > deadline - 5:
            break
            
    return best_classes

# Final Check:
# The above code stores `classes` as a list of lists.
# The return type is `classes [[p, r], ...]`.
# This matches the interface.
# One potential issue: `new_covered = covered.copy()` is expensive if done many times.
# With BEAM_WIDTH=50 and MAX_PRIMES=30, that's 1500 copies per iteration.
# Each copy is 100KB. 1500 * 100KB = 150MB per iteration.
# If we do 100 iterations, that's 15GB of memory traffic. Might be slow.
# Optimization: Use a "dirty" flag or just accept it if time allows.
# Given 120s, and numpy is fast, it might be okay.
# However, we can reduce MAX_PRIMES_TO_TRY if needed.
# Also, we can reduce BEAM_WIDTH.

# Let's refine the code to be robust.

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    LIMIT = 100000
    
    BEAM_WIDTH = 30
    MAX_PRIMES_TO_TRY = 20
    
    prime_costs = {p: Fraction(1, p) for p in primes}
    
    init_covered = np.zeros(LIMIT + 1, dtype=bool)
    init_state = {
        'covered': init_covered,
        'used': set(),
        'budget': Fraction(1, 1),
        'L': 0,
        'classes': []
    }
    
    beam = [init_state]
    best_L = 0
    best_classes = []
    
    while time.time() < deadline:
        next_beam = []
        
        for state in beam:
            if time.time() > deadline:
                break
            
            L = state['L']
            if L >= LIMIT:
                next_beam.append(state)
                continue
            
            covered = state['covered']
            used = state['used']
            budget = state['budget']
            classes = state['classes']
            
            n = L + 1
            if n > LIMIT:
                continue
            
            candidates = []
            
            for p in primes:
                if p in used:
                    continue
                cost = prime_costs[p]
                if budget < cost:
                    continue
                
                r = n % p
                
                new_covered = covered.copy()
                new_covered[r::p] = True
                
                # Find new L
                # Check from L+1
                # If L+1 > LIMIT, handled above
                remaining = ~new_covered[L+1:]
                if not np.any(remaining):
                    new_L = LIMIT
                else:
                    first_gap_offset = np.argmax(remaining)
                    new_L = L + first_gap_offset
                    
                candidates.append((new_L, p, r, new_covered, budget - cost))
                
            if not candidates:
                continue
                
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = candidates[:MAX_PRIMES_TO_TRY]
            
            for new_L, p, r, new_covered, new_budget in top_candidates:
                if new_L > best_L:
                    best_L = new_L
                    best_classes = classes + [[p, r]]
                
                next_beam.append({
                    'covered': new_covered,
                    'used': used | {p},
                    'budget': new_budget,
                    'L': new_L,
                    'classes': classes + [[p, r]]
                })
        
        if not next_beam:
            break
            
        next_beam.sort(key=lambda x: x['L'], reverse=True)
        beam = next_beam[:BEAM_WIDTH]
        
        if time.time() > deadline - 5:
            break
            
    return best_classes
