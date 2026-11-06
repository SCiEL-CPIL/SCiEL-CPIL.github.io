"""Search-based program: randomized greedy + local search."""
import time
import random
import numpy as np

_BASE = 51975
_DIVS = [d for d in range(3, _BASE + 1, 2) if _BASE % d == 0]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    rng = random.Random(42)
    
    best_score = _BASE
    best_classes = []
    
    while time.time() < deadline:
        # 1. Randomized Greedy Construction
        divs = _DIVS[:]
        rng.shuffle(divs)
        
        covered = np.zeros(_BASE, dtype=bool)
        classes = []
        used_moduli = set()
        
        for m in divs:
            # Find best residue r
            best_gain = 0
            best_r = -1
            for r in range(m):
                # Gain is number of uncovered residues in this slice
                total = len(covered[r::m])
                covered_count = np.count_nonzero(covered[r::m])
                gain = total - covered_count
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                classes.append([m, best_r])
                used_moduli.add(m)
                covered[best_r::m] = True
        
        # 2. Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # A. Try changing residue for used moduli
            for i in range(len(classes)):
                m, r = classes[i]
                best_gain = 0
                best_r = r
                for r_prime in range(m):
                    total = len(covered[r_prime::m])
                    covered_count = np.count_nonzero(covered[r_prime::m])
                    gain = total - covered_count
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r_prime
                
                if best_r != r:
                    # Update class
                    classes[i][1] = best_r
                    # Recalculate covered from scratch
                    covered = np.zeros(_BASE, dtype=bool)
                    for m2, r2 in classes:
                        covered[r2::m2] = True
                    improved = True
                    break # Restart loop to re-evaluate with new state
            
            # B. Try adding new moduli
            if not improved:
                for m in _DIVS:
                    if m in used_moduli:
                        continue
                    best_gain = 0
                    best_r = -1
                    for r in range(m):
                        total = len(covered[r::m])
                        covered_count = np.count_nonzero(covered[r::m])
                        gain = total - covered_count
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    
                    if best_gain > 0:
                        classes.append([m, best_r])
                        used_moduli.add(m)
                        covered[best_r::m] = True
                        improved = True
                        break # Restart loop
        
        # Calculate final score
        score = np.count_nonzero(~covered)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
            
    return best_classes
# EVOLVE-BLOCK-END
