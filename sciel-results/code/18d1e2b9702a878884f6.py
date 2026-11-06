import math
import random
import time

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _construct_with_core(core_prime, time_limit):
    """
    Construct classes using a core prime factor strategy, then extend greedily.
    """
    # 1. Generate core classes
    core_mods = [m for m in range(2, MAX_MOD + 1) if m % core_prime == 0]
    if len(core_mods) < core_prime:
        # Can't even fill the core
        core_mods = core_mods[:len(core_mods)]
    
    core_classes = []
    for i, m in enumerate(core_mods):
        r = i % core_prime
        core_classes.append((r, m))
        
    # 2. Extend greedily
    current_classes = core_classes[:]
    used_mods = set(m for r, m in current_classes)
    
    # Remaining moduli
    remaining_mods = [m for m in range(2, MAX_MOD + 1) if m not in used_mods]
    # Shuffle for variety? Or sort by something? 
    # Larger moduli have more residue options, might be easier to fit?
    # Or smaller moduli are more constrained?
    # Let's try sorting by modulus descending, then ascending, and pick best.
    
    best_extension = 0
    best_final_classes = current_classes
    
    # Try a few orders
    for order in ['desc', 'asc', 'random']:
        trial_classes = current_classes[:]
        trial_used = set(m for r, m in trial_classes)
        trial_remaining = [m for m in range(2, MAX_MOD + 1) if m not in trial_used]
        
        if order == 'desc':
            trial_remaining.sort(reverse=True)
        elif order == 'asc':
            trial_remaining.sort()
        else:
            random.shuffle(trial_remaining)
            
        for m in trial_remaining:
            if time.time() > time_limit:
                break
            # Try to find a residue
            # Residues can be 0..m-1
            # We can optimize by checking only residues that don't conflict
            # But m is small (<=128), so trying all is fine.
            found = False
            # Randomize residue order to avoid bias
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                valid = True
                for r2, m2 in trial_classes:
                    if not _disjoint(r, m, r2, m2):
                        valid = False
                        break
                if valid:
                    trial_classes.append((r, m))
                    trial_used.add(m)
                    found = True
                    break
                    
        if len(trial_classes) > len(best_final_classes):
            best_final_classes = trial_classes
            
    return best_final_classes

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    best_classes = []
    
    # Try different core primes
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    # Only primes where core_prime <= 128 // 2 ? No, just try reasonable ones.
    # If core_prime is large, number of mods is small.
    # Max useful prime is around 128/2 = 64?
    # Actually, if P=127, only 1 multiple.
    # Let's try primes up to 40.
    
    for p in primes:
        if p > 40:
            break
        if time.time() > deadline:
            break
            
        classes = _construct_with_core(p, deadline)
        if len(classes) > len(best_classes):
            best_classes = classes
            
    # Fallback: if best_classes is empty or small, try the original backtracking on evens?
    # Or just return best_classes.
    # The original program found 12.
    # If _construct_with_core finds less, we might want to try the old method.
    # But the old method was slow.
    # Let's add a quick greedy on all moduli as a final fallback.
    
    if len(best_classes) < 10:
        # Quick greedy on all moduli
        greedy_mods = list(range(2, MAX_MOD + 1))
        random.shuffle(greedy_mods)
        classes = []
        for m in greedy_mods:
            if time.time() > deadline:
                break
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                valid = True
                for r2, m2 in classes:
                    if not _disjoint(r, m, r2, m2):
                        valid = False
                        break
                if valid:
                    classes.append((r, m))
                    break
        if len(classes) > len(best_classes):
            best_classes = classes

    if not best_classes:
        best_classes = [(0, 2)]
        
    return [[r, m] for r, m in best_classes]
