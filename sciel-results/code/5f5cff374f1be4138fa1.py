import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    base = 51975
    # Get all divisors > 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    def score_of(classes):
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return base - covered.sum()
    
    best_score = base
    best_classes = []
    
    # Multiple randomized greedy attempts
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline - 30:
            break
        
        random.seed(attempt * 12345 + 7)
        order = divisors.copy()
        random.shuffle(order)
        
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        used_moduli = set()
        
        for m in order:
            if time.time() > deadline - 30:
                break
            # Find best residue for this modulus
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = np.sum(covered[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
            used_moduli.add(m)
        
        sc = score_of(classes)
        if sc < best_score:
            best_score = sc
            best_classes = classes[:]
        
        # Local search
        if time.time() > deadline - 10:
            break
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try changing residues
            for i, (m, r) in enumerate(classes):
                if time.time() > deadline - 10:
                    break
                best_r = r
                best_sc = sc
                # Try a few random residues
                for _ in range(min(20, m)):
                    tr = random.randint(0, m - 1)
                    if tr == r:
                        continue
                    classes[i][1] = tr
                    new_sc = score_of(classes)
                    if new_sc < best_sc:
                        best_sc = new_sc
                        best_r = tr
                    classes[i][1] = r
                if best_r != r:
                    classes[i][1] = best_r
                    sc = best_sc
                    improved = True
                    if sc < best_score:
                        best_score = sc
                        best_classes = classes[:]
            
            # Try adding unused moduli
            for m in divisors:
                if m in used_moduli:
                    continue
                if time.time() > deadline - 10:
                    break
                best_r = 0
                best_gain = -1
                for _ in range(min(10, m)):
                    r = random.randint(0, m - 1)
                    gain = np.sum(covered[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                if best_gain > 0:
                    classes.append([m, best_r])
                    covered[best_r::m] = 1
                    used_moduli.add(m)
                    sc = score_of(classes)
                    improved = True
                    if sc < best_score:
                        best_score = sc
                        best_classes = classes[:]
        
        if time.time() > deadline - 5:
            break
    
    return best_classes if best_classes else [[3, 0]]
