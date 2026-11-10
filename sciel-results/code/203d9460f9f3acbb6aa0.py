"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    best = []
    best_len = 0
    rng = random.Random(42)
    
    def can_add_fast(selected, x):
        n = len(selected)
        # Check 1: does x divide sum of any 3 in selected?
        if n >= 3:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if (selected[i] + selected[j] + selected[k]) % x == 0:
                            return False
        # Check 2: for each a in selected, does a divide x + two others?
        for i in range(n):
            a = selected[i]
            for j in range(n):
                if j == i: continue
                for k in range(j+1, n):
                    if k == i: continue
                    if (x + selected[j] + selected[k]) % a == 0:
                        return False
        return True
    
    for trial in range(30):
        if time.time() - start_time > 50:
            break
        
        candidates = list(range(40, 193))
        rng.shuffle(candidates)
        
        selected = []
        for x in candidates:
            if time.time() - start_time > 55:
                break
            if can_add_fast(selected, x):
                selected.append(x)
        
        # Try to add more from remaining
        for x in range(192, 0, -1):
            if time.time() - start_time > 55:
                break
            if x not in selected:
                if can_add_fast(selected, x):
                    selected.append(x)
        
        if len(selected) > best_len:
            best_len = len(selected)
            best = selected[:]
        
        if best_len >= 48:
            break
    
    return best
# EVOLVE-BLOCK-END
