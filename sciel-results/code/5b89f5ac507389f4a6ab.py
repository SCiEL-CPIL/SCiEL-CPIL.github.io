import time
import random
from fractions import Fraction
import math

START_TIME = time.time()
DEADLINE = START_TIME + 140  # Safety margin before 150s kill

def _sieve(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = _sieve(1000)
INV_PRIMES = {p: Fraction(1, p) for p in PRIMES}

def calc_L(classes, max_check=100000):
    """Calculate the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    
    covered = bytearray(max_check + 1)
    
    # Mark covered numbers
    for p, r in classes:
        start = r
        if start == 0:
            start = p
        elif start < 1:
            start = r + p  # Actually r>=0, if r==0 handled. If r>0, start=r is >=1.
            # Wait, if r=0, start=p. If r>0, start=r.
            # But what if r=0 and p=1? p is prime so p>=2.
            # If r=0, numbers are 0, p, 2p... In 1..L, we want p, 2p...
            # So start = p.
            # If r>0, numbers are r, r+p... In 1..L, if r>=1, start=r.
            # My code: if start==0: start=p. elif start<1: ...
            # r is in 0..p-1. So r is 0,1,...,p-1.
            # If r=0, start=p.
            # If r>=1, start=r.
            # So the elif is never triggered for valid r.
            pass
        
        for n in range(start, max_check + 1, p):
            covered[n] = 1
            
    # Find first gap
    for n in range(1, max_check + 1):
        if not covered[n]:
            return n - 1
            
    return max_check

def greedy_with_backtracking(L_target, budget=1, max_depth=6, time_limit=None):
    """
    Greedy construction with a limited backtracking window.
    Instead of committing to the single best choice, we keep the top K candidates
    and backtrack if we get stuck.
    """
    if time_limit is None:
        time_limit = DEADLINE - 2
    
    classes = []
    used_primes = set()
    current_sum = Fraction(0, 1)
    uncovered = set(range(1, L_target + 1))
    
    # We will use a stack of states for backtracking.
    # Each state: (uncovered, used_primes, current_sum, classes_copy, next_candidates)
    # To save memory, we might just track the last few decisions.
    
    # Let's implement a simpler "beam search" or "limited backtracking".
    # We'll maintain a list of states.
    # State: (uncovered_set, used_primes_set, current_sum, classes_list)
    
    # For efficiency, we'll just do a DFS with a depth limit on the "last few" moves?
    # No, let's do a simple recursive backtracking with a global timeout and pruning.
    # But L_target can be large.
    
    # Alternative:
    # Run greedy. If it fails to cover L_target (i.e., uncovered is not empty and we run out of budget/primes),
    # we can try to "undo" the last move and try the second best option for that move.
    
    # Let's implement a function that returns the best solution found within a time limit.
    
    best_classes = []
    best_L = 0
    
    # We'll simulate the greedy process but keep a history of choices.
    # history: list of (p, r, uncovered_before, count_before)
    
    history = []
    
    def backtrack(state_idx, uncovered, used_primes, current_sum, classes):
        """
        Try to complete the solution from state_idx.
        state_idx is the index in history where we are currently making a choice.
        """
        nonlocal best_classes, best_L
        
        if time.time() > time_limit:
            return False
            
        if not uncovered:
            # We covered everything up to L_target!
            # Update best
            L_achieved = L_target
            # We could potentially extend, but for now L_target is a good baseline.
            if L_achieved > best_L:
                best_L = L_achieved
                best_classes = classes[:]
            return True
            
        if current_sum >= 1:
            # Out of budget
            return False
            
        # Get candidates for this state
        n_min = min(uncovered)
        
        candidates = []
        for p in PRIMES:
            if p in used_primes:
                continue
            if current_sum + INV_PRIMES[p] > 1:
                continue
            
            r = n_min % p
            count = sum(1 for n in uncovered if n % p == r)
            score = count * p  # Efficiency score
            
            # Pruning: if count is 0, skip (though n_min ensures count >= 1)
            candidates.append((score, p, r, count))
            
        if not candidates:
            return False
            
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Limit branching factor for speed
        max_branches = 3
        
        for score, p, r, count in candidates[:max_branches]:
            # Make move
            new_uncovered = set(n for n in uncovered if n % p != r)
            new_used = used_primes | {p}
            new_sum = current_sum + INV_PRIMES[p]
            new_classes = classes + [[p, r]]
            
            # Recurse
            if backtrack(state_idx + 1, new_uncovered, new_used, new_sum, new_classes):
                return True
                
        return False
        
    # Start backtracking from the initial state
    # But pure backtracking from scratch is expensive.
    # Let's use the greedy result as a starting point?
    # Or just run the backtracking with a small depth limit effectively by pruning?
    
    # Actually, the function above explores the search tree.
    # To make it feasible, we rely on the "score" sorting and time limit.
    
    # Let's try it.
    backtrack(0, uncovered, used_primes, current_sum, classes)
    
    return best_classes

def construct():
    """
    Combines greedy efficiency scoring with limited backtracking to escape local minima.
    """
    best_classes_overall = None
    best_L_overall = 0
    
    # Try a few different L targets
    # The seed achieves 999. We want to beat it.
    # Let's try 1000, 1200, 1500.
    
    targets = [1000, 1200, 1500, 2000]
    
    for L_target in targets:
        if time.time() > DEADLINE - 10:
            break
            
        # Run the backtracking solver
        # It returns the best classes found for this target
        classes = greedy_with_backtracking(L_target)
        
        if classes:
            # Verify sum
            s = sum(Fraction(1, p) for p, r in classes)
            if s <= 1:
                # Calculate actual L
                # We check up to L_target + 500 to see if we overshot
                check_limit = max(L_target + 500, 1000)
                L_achieved = calc_L(classes, check_limit)
                
                if L_achieved > best_L_overall:
                    best_L_overall = L_achieved
                    best_classes_overall = classes
                    
    # Fallback if backtracking didn't find anything good
    if best_classes_overall is None or best_L_overall < 999:
        # Use a simple greedy as fallback
        def simple_greedy(L_target):
            classes = []
            used_primes = set()
            current_sum = Fraction(0, 1)
            uncovered = set(range(1, L_target + 1))
            
            while uncovered and current_sum < 1:
                best_p = None
                best_r = None
                best_score = -1
                n_min = min(uncovered)
                
                for p in PRIMES:
                    if p in used_primes:
                        continue
                    if current_sum + INV_PRIMES[p] > 1:
                        continue
                    
                    r = n_min % p
                    count = sum(1 for n in uncovered if n % p == r)
                    score = count * p
                    
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_r = r
                        
                if best_p is None:
                    break
                    
                classes.append([best_p, best_r])
                used_primes.add(best_p)
                current_sum += INV_PRIMES[best_p]
                
                new_uncovered = set(n for n in uncovered if n % best_p != best_r)
                uncovered = new_uncovered
                
                if time.time() > DEADLINE - 5:
                    break
                    
            return classes
            
        classes = simple_greedy(1000)
        L_achieved = calc_L(classes, 1500)
        
        if L_achieved > best_L_overall:
            best_L_overall = L_achieved
            best_classes_overall = classes
            
    # Final safety: if still nothing, return a trivial valid set
    if best_classes_overall is None:
        # Just return a single class to be valid
        best_classes_overall = [[2, 1]]
        
    return best_classes_overall

if __name__ == "__main__":
    # Self-test
    classes = construct()
    L = calc_L(classes, 10000)
    s = sum(Fraction(1, p) for p, r in classes)
    print(f"Found L={L}, Sum={float(s)}, Classes={len(classes)}")
