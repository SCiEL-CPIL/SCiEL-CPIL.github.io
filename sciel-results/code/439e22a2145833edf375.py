import time
import math
import random

BASE = 51975

def _all_divisors(n):
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    divs.discard(1)
    return sorted(divs)

_DIVISORS = _all_divisors(BASE)

# Precompute masks using Python integers (arbitrary precision)
_MASKS = {}
for m in _DIVISORS:
    masks_m = [0] * m
    for r in range(m):
        mask = 0
        for i in range(r, BASE, m):
            mask |= (1 << i)
        masks_m[r] = mask
    _MASKS[m] = masks_m

def _popcount(x):
    return x.bit_count()

def _uncovered_count(covered_mask):
    """Count residues not covered by covered_mask."""
    full_mask = (1 << BASE) - 1
    return _popcount(~covered_mask & full_mask)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Phase 1: Greedy initialization with multiple restarts
    best_result = None
    best_score = BASE  # Start with all uncovered
    
    # Try several different greedy strategies
    for restart in range(5):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            # Standard greedy: pick max gain
            covered = 0
            classes = {}
            available_moduli = set(_DIVISORS)
            
            while time.time() < deadline - 40:
                best_m = None
                best_r = None
                best_gain = 0
                
                for m in available_moduli:
                    masks_m = _MASKS[m]
                    min_overlap = BASE
                    best_r_m = 0
                    for r in range(m):
                        overlap = _popcount(masks_m[r] & covered)
                        if overlap < min_overlap:
                            min_overlap = overlap
                            best_r_m = r
                    
                    gain = _popcount(masks_m[best_r_m] & ~covered)
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = best_r_m
                
                if best_m is None or best_gain == 0:
                    break
                    
                classes[best_m] = best_r
                covered |= _MASKS[best_m][best_r]
                available_moduli.remove(best_m)
                
                if not available_moduli:
                    break
            
            current_classes = classes
            current_covered = covered
        else:
            # Randomized greedy: add some randomness to break ties
            covered = 0
            classes = {}
            available_moduli = list(_DIVISORS)
            random.shuffle(available_moduli)
            
            while time.time() < deadline - 40:
                best_m = None
                best_r = None
                best_gain = 0
                
                for m in available_moduli:
                    masks_m = _MASKS[m]
                    min_overlap = BASE
                    best_r_m = 0
                    for r in range(m):
                        overlap = _popcount(masks_m[r] & covered)
                        if overlap < min_overlap:
                            min_overlap = overlap
                            best_r_m = r
                    
                    gain = _popcount(masks_m[best_r_m] & ~covered)
                    # Add small random perturbation to break ties
                    gain += random.random() * 0.1
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = best_r_m
                
                if best_m is None or best_gain <= 0.1:  # Threshold to account for perturbation
                    break
                    
                classes[best_m] = best_r
                covered |= _MASKS[best_m][best_r]
                available_moduli.remove(best_m)
                
                if not available_moduli:
                    break
            
            current_classes = classes
            current_covered = covered
        
        # Phase 2: Local search - optimize residues
        if time.time() < deadline - 20:
            improved = True
            max_iter = 10
            iteration = 0
            while improved and iteration < max_iter and time.time() < deadline - 20:
                improved = False
                iteration += 1
                
                # Compute current coverage count
                current_uncovered = _uncovered_count(current_covered)
                
                # For each modulus in current classes, try to find better residue
                for m in list(current_classes.keys()):
                    if time.time() > deadline - 20:
                        break
                    
                    # Compute mask of all other classes
                    others_mask = 0
                    for m2, r2 in current_classes.items():
                        if m2 != m:
                            others_mask |= _MASKS[m2][r2]
                    
                    masks_m = _MASKS[m]
                    current_r = current_classes[m]
                    current_gain = _popcount(masks_m[current_r] & ~others_mask)
                    
                    best_r = current_r
                    best_gain = current_gain
                    
                    for r in range(m):
                        if r == current_r:
                            continue
                        gain = _popcount(masks_m[r] & ~others_mask)
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    
                    if best_r != current_r:
                        # Verify improvement
                        new_covered = others_mask | masks_m[best_r]
                        new_uncovered = _uncovered_count(new_covered)
                        if new_uncovered < current_uncovered:
                            current_classes[m] = best_r
                            current_covered = new_covered
                            current_uncovered = new_uncovered
                            improved = True
                
                # Try adding a new class if available
                if time.time() < deadline - 20 and len(current_classes) < len(_DIVISORS):
                    best_m_add = None
                    best_r_add = None
                    best_gain_add = 0
                    
                    for m in _DIVISORS:
                        if m in current_classes:
                            continue
                        
                        masks_m = _MASKS[m]
                        best_r_m = 0
                        best_gain_m = 0
                        for r in range(m):
                            gain = _popcount(masks_m[r] & ~current_covered)
                            if gain > best_gain_m:
                                best_gain_m = gain
                                best_r_m = r
                        
                        if best_gain_m > best_gain_add:
                            best_gain_add = best_gain_m
                            best_m_add = m
                            best_r_add = best_r_m
                    
                    if best_m_add is not None and best_gain_add > 0:
                        current_classes[best_m_add] = best_r_add
                        current_covered |= _MASKS[best_m_add][best_r_add]
                        improved = True
                
                # Try removing a class that doesn't help
                if time.time() < deadline - 20:
                    for m in list(current_classes.keys()):
                        if time.time() > deadline - 20:
                            break
                        
                        others_mask = 0
                        for m2, r2 in current_classes.items():
                            if m2 != m:
                                others_mask |= _MASKS[m2][r2]
                        
                        without_m_uncovered = _uncovered_count(others_mask)
                        with_m_uncovered = _uncovered_count(current_covered)
                        
                        if without_m_uncovered <= with_m_uncovered:
                            # Removing m doesn't hurt (or helps)
                            del current_classes[m]
                            current_covered = others_mask
                            improved = True
        
        # Phase 3: Simulated annealing polish
        if time.time() < deadline - 5:
            sa_start = time.time()
            sa_deadline = deadline - 5
            temp = 5.0
            cooling = 0.95
            current_uncovered = _uncovered_count(current_covered)
            best_sa_uncovered = current_uncovered
            best_sa_classes = current_classes.copy()
            best_sa_covered = current_covered
            
            while time.time() < sa_deadline:
                # Pick a random move
                move_type = random.random()
                
                if move_type < 0.6 and current_classes:
                    # Change residue of existing class
                    m = random.choice(list(current_classes.keys()))
                    old_r = current_classes[m]
                    new_r = random.randint(0, m - 1)
                    if new_r == old_r:
                        continue
                    
                    others_mask = 0
                    for m2, r2 in current_classes.items():
                        if m2 != m:
                            others_mask |= _MASKS[m2][r2]
                    
                    old_covered = others_mask | _MASKS[m][old_r]
                    new_covered = others_mask | _MASKS[m][new_r]
                    delta = _uncovered_count(new_covered) - current_uncovered
                    
                    if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                        current_classes[m] = new_r
                        current_covered = new_covered
                        current_uncovered = _uncovered_count(new_covered)
                        
                        if current_uncovered < best_sa_uncovered:
                            best_sa_uncovered = current_uncovered
                            best_sa_classes = current_classes.copy()
                            best_sa_covered = current_covered
                elif move_type < 0.8 and len(current_classes) < len(_DIVISORS):
                    # Add a class
                    available = [m for m in _DIVISORS if m not in current_classes]
                    if available:
                        m = random.choice(available)
                        r = random.randint(0, m - 1)
                        new_covered = current_covered | _MASKS[m][r]
                        delta = _uncovered_count(new_covered) - current_uncovered
                        
                        if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                            current_classes[m] = r
                            current_covered = new_covered
                            current_uncovered = _uncovered_count(new_covered)
                            
                            if current_uncovered < best_sa_uncovered:
                                best_sa_uncovered = current_uncovered
                                best_sa_classes = current_classes.copy()
                                best_sa_covered = current_covered
                elif move_type < 0.9 and current_classes:
                    # Remove a class
                    m = random.choice(list(current_classes.keys()))
                    del current_classes[m]
                    
                    new_covered = 0
                    for m2, r2 in current_classes.items():
                        new_covered |= _MASKS[m2][r2]
                    delta = _uncovered_count(new_covered) - current_uncovered
                    
                    if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                        current_covered = new_covered
                        current_uncovered = _uncovered_count(new_covered)
                        
                        if current_uncovered < best_sa_uncovered:
                            best_sa_uncovered = current_uncovered
                            best_sa_classes = current_classes.copy()
                            best_sa_covered = current_covered
                
                temp *= cooling
            
            current_classes = best_sa_classes
            current_covered = best_sa_covered
        
        # Update best
        final_uncovered = _uncovered_count(current_covered)
        if final_uncovered < best_score:
            best_score = final_uncovered
            best_result = current_classes
    
    # Convert to required format
    if best_result is None:
        return []
    
    result = []
    for m, r in best_result.items():
        result.append([m, r])
        
    return result
