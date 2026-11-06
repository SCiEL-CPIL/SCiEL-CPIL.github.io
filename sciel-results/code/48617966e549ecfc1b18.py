"""Structural change: Coverage-aware greedy prime selection + multi-pass residue local search."""
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

def _coverage_length(classes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by the given classes."""
    if not classes:
        return 0
    
    cov = bytearray(max_check + 1)
    
    for p, r in classes:
        start = r if r > 0 else p
        if start > max_check:
            continue
        for n in range(start, max_check + 1, p):
            cov[n] = 1
            
    L = 0
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

def _coverage_gain(classes, p, r, max_check=2000):
    """Compute how many new numbers in 1..max_check are covered by adding [p, r]."""
    if not classes:
        # Count how many numbers 1..max_check are in class r mod p
        start = r if r > 0 else p
        if start > max_check:
            return 0
        return (max_check - start) // p + 1
    
    # Build coverage set for current classes
    cov = bytearray(max_check + 1)
    for cp, cr in classes:
        cstart = cr if cr > 0 else cp
        if cstart <= max_check:
            for n in range(cstart, max_check + 1, cp):
                cov[n] = 1
    
    # Count new coverage from [p, r]
    start = r if r > 0 else p
    if start > max_check:
        return 0
    gain = 0
    for n in range(start, max_check + 1, p):
        if not cov[n]:
            gain += 1
    return gain

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    primes = _sieve(1000)
    
    best_L = 0
    best_classes = []
    
    # Strategy: Greedy prime selection based on coverage gain
    # At each step, pick the prime+residue that maximizes new coverage per unit cost
    # This is a structural change from fixed prime sets
    
    current_classes = []
    current_cost = Fraction(0)
    
    # Precompute primes list
    available_primes = list(primes)
    
    # Greedy selection: repeatedly add the prime/residue with best gain/cost ratio
    # We'll do multiple rounds of local search after each addition
    max_primes_to_add = 30  # Cap to avoid too many classes
    
    for step in range(max_primes_to_add):
        if time.time() > deadline - 30:
            break
        
        best_gain_cost = 0
        best_p = None
        best_r = None
        
        # Try all available primes and their best residue
        for p in available_primes:
            if current_cost + Fraction(1, p) > Fraction(1):
                continue
            
            # Find best residue for this prime
            best_r_for_p = 0
            best_gain_for_p = -1
            
            for r in range(p):
                gain = _coverage_gain(current_classes, p, r, 2000)
                if gain > best_gain_for_p:
                    best_gain_for_p = gain
                    best_r_for_p = r
            
            if best_gain_for_p <= 0:
                continue
            
            gain_cost = best_gain_for_p / p  # Approximate: gain per unit of p (cost is 1/p)
            # Actually cost is 1/p, so gain per cost = gain / (1/p) = gain * p
            # But we want to maximize gain subject to cost constraint
            # Let's just maximize gain directly, then check cost
            if best_gain_for_p > best_gain_cost:
                best_gain_cost = best_gain_for_p
                best_p = p
                best_r = best_r_for_p
        
        if best_p is None:
            break
        
        # Add the best prime/residue
        current_classes.append([best_p, best_r])
        current_cost += Fraction(1, best_p)
        available_primes.remove(best_p)
        
        # Quick local search to improve residues
        improved = True
        passes = 0
        while improved and time.time() < deadline - 20 and passes < 3:
            improved = False
            passes += 1
            for idx in range(len(current_classes)):
                p = current_classes[idx][0]
                old_r = current_classes[idx][1]
                best_r_local = old_r
                best_L_local = _coverage_length(current_classes, 3000)
                
                for r in range(p):
                    if r == old_r:
                        continue
                    current_classes[idx][1] = r
                    L_new = _coverage_length(current_classes, 3000)
                    if L_new > best_L_local:
                        best_L_local = L_new
                        best_r_local = r
                
                current_classes[idx][1] = best_r_local
                if best_L_local > _coverage_length(current_classes, 3000):
                    improved = True
        
        # Track best
        L_check = _coverage_length(current_classes, 5000)
        if L_check > best_L:
            best_L = L_check
            best_classes = [c[:] for c in current_classes]
    
    # Final intensive local search on the best solution
    if best_classes:
        current_classes = [c[:] for c in best_classes]
        current_L = _coverage_length(current_classes, 10000)
        
        improved = True
        passes = 0
        while improved and time.time() < deadline - 10 and passes < 15:
            improved = False
            passes += 1
            for idx in range(len(current_classes)):
                p = current_classes[idx][0]
                old_r = current_classes[idx][1]
                best_r_local = old_r
                best_L_local = current_L
                
                for r in range(p):
                    if r == old_r:
                        continue
                    current_classes[idx][1] = r
                    L_new = _coverage_length(current_classes, 10000)
                    if L_new > best_L_local:
                        best_L_local = L_new
                        best_r_local = r
                
                current_classes[idx][1] = best_r_local
                if best_L_local > current_L:
                    current_L = best_L_local
                    improved = True
        
        # Final verification
        final_L = _coverage_length(current_classes, 100000)
        if final_L > best_L:
            best_L = final_L
            best_classes = [c[:] for c in current_classes]
    
    # Fallback
    if not best_classes:
        best_classes = [[2, 0]]
    
    # Final validation
    final_cost = _total_cost(best_classes)
    if final_cost > Fraction(1):
        best_classes.sort(key=lambda x: -1.0 / x[0])
        while final_cost > Fraction(1) and best_classes:
            p, r = best_classes.pop()
            final_cost -= Fraction(1, p)
    
    # Ensure distinct primes
    seen = set()
    filtered = []
    for p, r in best_classes:
        if p not in seen:
            seen.add(p)
            filtered.append([p, r])
    best_classes = filtered
    
    return best_classes

# EVOLVE-BLOCK-END
