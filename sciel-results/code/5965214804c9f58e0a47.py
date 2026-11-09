import time
import random
import math
import numpy as np

def construct():
    N = 51975
    # Find all divisors > 1
    divisors = []
    for i in range(3, N + 1, 2):
        if N % i == 0:
            divisors.append(i)
    divisors.sort()
    
    start_time = time.time()
    deadline = start_time + 145 # Safety margin
    
    def score_classes(classes):
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(np.sum(1 - covered))

    def fast_greedy(moduli_list, seed=None):
        if seed is not None:
            random.seed(seed)
        mods = list(moduli_list)
        if seed is not None:
            random.shuffle(mods)
        
        covered_np = np.zeros(N, dtype=np.uint8)
        classes = []
        
        for m in mods:
            U = 1 - covered_np
            # Find best r
            gains = np.zeros(m)
            for r in range(m):
                gains[r] = np.sum(U[r::m])
            best_r = int(np.argmax(gains))
            classes.append([m, best_r])
            covered_np[best_r::m] = 1
            
        return classes

    def local_search(classes):
        counts_np = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            counts_np[r::m] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            U = (counts_np == 0).astype(np.uint8)
            
            for k in range(len(classes)):
                m, r_old = classes[k]
                
                idxs_old = np.arange(r_old, N, m)
                loss = np.sum(counts_np[idxs_old] == 1)
                
                gains = np.zeros(m)
                for r in range(m):
                    gains[r] = np.sum(U[r::m])
                
                best_r_new = int(np.argmax(gains))
                max_gain = gains[best_r_new]
                
                if max_gain > loss:
                    counts_np[idxs_old] -= 1
                    idxs_new = np.arange(best_r_new, N, m)
                    counts_np[idxs_new] += 1
                    classes[k][1] = best_r_new
                    improved = True
                    U = (counts_np == 0).astype(np.uint8)
        
        return classes

    # Initial best
    best_classes = fast_greedy(divisors, seed=42)
    best_score = score_classes(best_classes)
    
    # Iterated Local Search
    curr_classes = best_classes
    curr_score = best_score
    
    while time.time() < deadline:
        # Perturb
        new_classes = [c[:] for c in curr_classes]
        num_changes = random.randint(1, 2)
        for _ in range(num_changes):
            idx = random.randint(0, len(new_classes) - 1)
            m, r = new_classes[idx]
            new_r = random.randint(0, m - 1)
            new_classes[idx][1] = new_r
        
        # Local Search
        improved_classes = local_search(new_classes)
        new_score = score_classes(improved_classes)
        
        if new_score < best_score:
            best_score = new_score
            best_classes = improved_classes
            curr_classes = improved_classes
        else:
            # Accept to keep exploring from this new local optimum
            curr_classes = improved_classes

    return best_classes
