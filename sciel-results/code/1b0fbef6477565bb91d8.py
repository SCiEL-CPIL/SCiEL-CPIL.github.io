"""Structural change: Fixed prime set selection + Intensive Residue Local Search."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _coverage_length_fast(classes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by the given classes.
    Uses a bytearray for speed.
    """
    if not classes:
        return 0
    
    # We only need to check up to max_check
    cov = bytearray(max_check + 1)
    
    for p, r in classes:
        if r == 0:
            start = p
        else:
            start = r
            
        if start > max_check:
            continue
            
        # Mark coverage
        # Using slice assignment is faster in Python for bytearrays
        # But we need to handle the step correctly.
        # cov[start:max_check+1:p] = b'\x01' * len(cov[start:max_check+1:p])
        
        # Manual loop is often faster in pure Python for small steps, but slice is better for large.
        # Let's use a simple loop as it's robust and p is small.
        for n in range(start, max_check + 1, p):
            cov[n] = 1
            
    # Find first gap
    L = 0
    # We can break early if we find a 0
    for n in range(1, max_check + 1):
        if cov[n]:
            L = n
        else:
            break
            
    return L

def _total_cost(classes):
    """Compute sum of 1/p for all classes using exact rationals."""
    total = Fraction(0)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def construct():
    start_time = time.time()
    # Strict deadline to avoid timeout (120s limit, use 100s for safety)
    deadline = start_time + 100.0
    
    primes = _sieve(1000)
    
    # Structural Change:
    # Instead of searching over prime sets AND residues, we fix a few promising prime sets
    # and perform intensive local search on residues for each.
    # This reduces the search space dimensionality significantly.
    
    # Candidate prime sets:
    # 1. Smallest primes that fit in budget 1.
    # 2. A mix of small and medium primes.
    
    candidate_sets = []
    
    # Set 1: Greedy smallest primes
    cost = Fraction(0)
    set1 = []
    for p in primes:
        if cost + Fraction(1, p) <= Fraction(1):
            set1.append(p)
            cost += Fraction(1, p)
        else:
            break
    candidate_sets.append(set1)
    
    # Set 2: Skip 2? (2 covers half, very expensive 1/2). Maybe start from 3?
    # If we skip 2, we have 1/2 budget left.
    # Let's try a set that includes 3,5,7,11,13...
    cost = Fraction(0)
    set2 = []
    for p in primes:
        if p == 2:
            continue
        if cost + Fraction(1, p) <= Fraction(1):
            set2.append(p)
            cost += Fraction(1, p)
        else:
            break
    candidate_sets.append(set2)
    
    # Set 3: The seed solution primes (>= 13) if they fit?
    # Sum of 1/p for p>=13 up to 1000 is ~ 2.2 - (1/2+1/3+1/5+1/7+1/11) = 2.2 - 1.1 = 1.1.
    # So we can't take ALL. We need to trim.
    cost = Fraction(0)
    set3 = []
    for p in primes:
        if p < 13:
            continue
        if cost + Fraction(1, p) <= Fraction(1):
            set3.append(p)
            cost += Fraction(1, p)
        else:
            break
    candidate_sets.append(set3)
    
    # Set 4: A balanced set. Pick primes such that 1/p sum is close to 1 but diverse.
    # We'll just use the smallest primes that fit, but maybe skip 2 to save budget for more primes?
    # 1/2 = 0.5. Remaining 0.5.
    # 1/3=0.33, 1/5=0.2. Sum=0.53 > 0.5. So if we take 2, we can take 3 but not 5?
    # 2+3 = 0.83. Remaining 0.17. 1/7=0.14. 2+3+7 = 0.97. Remaining 0.03. 1/29=0.03.
    # So {2,3,7,29} is a valid set.
    
    best_L = 0
    best_classes = []
    
    # For each candidate set, optimize residues
    for p_set in candidate_sets:
        if not p_set:
            continue
            
        # Initialize residues to 0 (or random)
        current_classes = [[p, 0] for p in p_set]
        
        # Local Search on Residues
        # We iterate through primes and try all residues to maximize coverage.
        # We do this for several passes until no improvement.
        
        current_L = _coverage_length_fast(current_classes, 5000)
        
        improved = True
        passes = 0
        while improved and time.time() < deadline and passes < 10:
            improved = False
            passes += 1
            for idx, p in enumerate(p_set):
                old_r = current_classes[idx][1]
                best_r = old_r
                best_local_L = current_L
                
                # Try all residues
                for r in range(p):
                    if r == old_r:
                        continue
                    current_classes[idx][1] = r
                    # Check coverage up to 5000 for speed during search
                    L_new = _coverage_length_fast(current_classes, 5000)
                    if L_new > best_local_L:
                        best_local_L = L_new
                        best_r = r
                
                current_classes[idx][1] = best_r
                
                if best_local_L > current_L:
                    current_L = best_local_L
                    improved = True
        
        # If this set looks promising, verify with a larger range (100,000)
        if current_L > best_L or (current_L == best_L and current_L > 0):
            # Final check on the best candidate found so far
            final_L = _coverage_length_fast(current_classes, 100000)
            if final_L > best_L:
                best_L = final_L
                best_classes = current_classes[:]
                
    # If no good set was found, fallback to empty or a trivial set
    if not best_classes:
        # Return a valid trivial solution
        # Just [2, 0] is valid (cost 1/2 <= 1)
        best_classes = [[2, 0]]
        
    # Final validation
    final_cost = _total_cost(best_classes)
    if final_cost > Fraction(1):
        # This should not happen if we constructed sets carefully, but just in case
        best_classes.sort(key=lambda x: -1.0 / x[0])
        while final_cost > Fraction(1) and best_classes:
            p, r = best_classes.pop()
            final_cost -= Fraction(1, p)
            
    # Ensure distinct primes (they are by construction, but just in case)
    seen = set()
    filtered = []
    for p, r in best_classes:
        if p not in seen:
            seen.add(p)
            filtered.append([p, r])
    best_classes = filtered
    
    return best_classes

# EVOLVE-BLOCK-END
