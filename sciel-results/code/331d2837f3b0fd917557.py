"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill
    
    base = 51975
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    used_moduli = set()
    
    while True:
        if time.time() > deadline - 5:
            break
        
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in divisors:
            if m in used_moduli:
                continue
            if time.time() > deadline - 5:
                break
            
            num_positions = base // m
            flat = covered[:num_positions * m].reshape(num_positions, m)
            gains = flat.sum(axis=0)
            
            max_gain = int(np.max(gains))
            if max_gain > best_gain:
                best_gain = max_gain
                best_m = m
                best_r = int(np.argmax(gains))
        
        if best_m is None or best_gain == 0:
            break
        
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        
        positions = np.arange(best_r, base, best_m)
        covered[positions] = 1
    
    return classes
# EVOLVE-BLOCK-END
