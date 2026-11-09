"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _covered_prefix(coverage_set):
    """Find largest L such that 1..L all in coverage_set."""
    L = 0
    for n in range(1, 1001):
        if n in coverage_set:
            L = n
        else:
            break
    return L

def _coverage_from_classes(classes, L_max=1000):
    """Compute set of covered integers in 1..L_max."""
    covered = set()
    for p, r in classes:
        # Integers n in 1..L_max with n ≡ r (mod p)
        # First such n >= 1: if r == 0, first is p; else first is r (if r>=1) or p+r
        if r == 0:
            start = p
        else:
            start = r if r >= 1 else p + r
        if start > L_max:
            continue
        for n in range(start, L_max + 1, p):
            covered.add(n)
    return covered

def _recip_sum(classes):
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def _greedy_init(primes):
    """Greedy: pick primes that maximize new coverage per cost."""
    classes = []
    covered = set()
    budget = Fraction(1, 1)
    remaining = list(primes)
    
    while remaining:
        best_score = -1
        best_idx = -1
        best_p = -1
        best_r = -1
        for i, p in enumerate(remaining):
            cost = Fraction(1, p)
            if cost > budget:
                continue
            # Try all residues, find best coverage gain
            best_gain = -1
            best_res = -1
            for r in range(p):
                # Count new coverage
                gain = 0
                if r == 0:
                    start = p
                else:
                    start = r if r >= 1 else p + r
                if start > 1000:
                    continue
                for n in range(start, 1001, p):
                    if n not in covered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_res = r
            if best_gain > best_score:
                best_score = best_gain
                best_idx = i
                best_p = p
                best_r = best_res
        if best_score <= 0:
            break
        classes.append([best_p, best_r])
        budget -= Fraction(1, best_p)
        remaining.pop(best_idx)
        # Update covered
        p, r = best_p, best_r
        if r == 0:
            start = p
        else:
            start = r if r >= 1 else p + r
        for n in range(start, 1001, p):
            covered.add(n)
    return classes

def construct():
    deadline = time.time() + 130  # safety margin
    primes = _primes_upto(1000)
    random.seed(42)
    
    # Start with greedy initialization
    classes = _greedy_init(primes)
    covered = _coverage_from_classes(classes)
    best_L = _covered_prefix(covered)
    best_classes = [c[:] for c in classes]
    best_recip = _recip_sum(classes)
    
    if best_L >= 999:
        return best_classes
    
    # Local search: try changing residues, adding/removing primes
    T = 1.0  # temperature
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        # Choose a random move type
        move_type = random.random()
        
        if move_type < 0.6 and classes:
            # Change residue of a random class
            idx = random.randint(0, len(classes) - 1)
            p, r = classes[idx]
            new_r = random.randint(0, p - 1)
            if new_r == r:
                continue
            old_r = r
            classes[idx][1] = new_r
            
            new_covered = _coverage_from_classes(classes)
            new_L = _covered_prefix(new_covered)
            
            delta = new_L - best_L
            if delta >= 0 or random.random() < (1.0 / (1.0 + (delta if delta < 0 else 0))):
                if delta >= 0:
                    best_L = new_L
                    best_classes = [c[:] for c in classes]
                    best_recip = _recip_sum(classes)
                    covered = new_covered
                    if best_L >= 999:
                        return best_classes
                else:
                    covered = new_covered
            else:
                classes[idx][1] = old_r
        
        elif move_type < 0.8 and classes:
            # Swap a prime out and try adding another
            idx = random.randint(0, len(classes) - 1)
            p_out, r_out = classes[idx][:]
            classes.pop(idx)
            
            # Find a prime not currently in use
            used_primes = set(c[0] for c in classes)
            available = [p for p in primes if p not in used_primes]
            if available:
                p_in = random.choice(available)
                cost_in = Fraction(1, p_in)
                current_recip = _recip_sum(classes)
                if current_recip + cost_in <= Fraction(1, 1):
                    # Find best residue for new prime
                    new_covered_temp = _coverage_from_classes(classes)
                    best_gain = -1
                    best_r = 0
                    for r in range(p_in):
                        if r == 0:
                            start = p_in
                        else:
                            start = r if r >= 1 else p_in + r
                        if start > 1000:
                            continue
                        gain = 0
                        for n in range(start, 1001, p_in):
                            if n not in new_covered_temp:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    classes.append([p_in, best_r])
                    
                    new_covered = _coverage_from_classes(classes)
                    new_L = _covered_prefix(new_covered)
                    delta = new_L - best_L
                    if delta >= 0 or random.random() < (1.0 / (1.0 + (abs(delta) if delta < 0 else 0))):
                        if delta >= 0:
                            best_L = new_L
                            best_classes = [c[:] for c in classes]
                            best_recip = _recip_sum(classes)
                            covered = new_covered
                            if best_L >= 999:
                                return best_classes
                        else:
                            covered = new_covered
                    else:
                        classes.pop()
                        classes.append([p_out, r_out])
                else:
                    classes.append([p_out, r_out])
            else:
                classes.append([p_out, r_out])
        
        elif move_type < 0.9 and classes:
            # Remove a class (if we have budget slack or want to restructure)
            idx = random.randint(0, len(classes) - 1)
            removed = classes.pop(idx)
            
            new_covered = _coverage_from_classes(classes)
            new_L = _covered_prefix(new_covered)
            delta = new_L - best_L
            if delta >= 0 or random.random() < (1.0 / (1.0 + (abs(delta) if delta < 0 else 0))):
                if delta >= 0:
                    best_L = new_L
                    best_classes = [c[:] for c in classes]
                    best_recip = _recip_sum(classes)
                    covered = new_covered
                    if best_L >= 999:
                        return best_classes
                else:
                    covered = new_covered
            else:
                classes.append(removed)
        
        else:
            # Add a new prime class if budget allows
            used_primes = set(c[0] for c in classes)
            available = [p for p in primes if p not in used_primes]
            if available:
                p_in = random.choice(available)
                cost_in = Fraction(1, p_in)
                current_recip = _recip_sum(classes)
                if current_recip + cost_in <= Fraction(1, 1):
                    # Find best residue
                    new_covered_temp = _coverage_from_classes(classes)
                    best_gain = -1
                    best_r = 0
                    for r in range(p_in):
                        if r == 0:
                            start = p_in
                        else:
                            start = r if r >= 1 else p_in + r
                        if start > 1000:
                            continue
                        gain = 0
                        for n in range(start, 1001, p_in):
                            if n not in new_covered_temp:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    classes.append([p_in, best_r])
                    
                    new_covered = _coverage_from_classes(classes)
                    new_L = _covered_prefix(new_covered)
                    delta = new_L - best_L
                    if delta >= 0 or random.random() < (1.0 / (1.0 + (abs(delta) if delta < 0 else 0))):
                        if delta >= 0:
                            best_L = new_L
                            best_classes = [c[:] for c in classes]
                            best_recip = _recip_sum(classes)
                            covered = new_covered
                            if best_L >= 999:
                                return best_classes
                        else:
                            covered = new_covered
                    else:
                        classes.pop()
        
        T *= cooling
    
    return best_classes
# EVOLVE-BLOCK-END
