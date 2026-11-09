import numpy as np
import time
import random

def construct():
    base = 51975
    
    # Get all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
            
    # Precompute indices for each modulus and residue
    # indices_cache[m][r] = np.array of indices covered by (m, r)
    indices_cache = {}
    for m in divisors:
        indices_cache[m] = []
        for r in range(m):
            indices_cache[m].append(np.arange(r, base, m))

    def greedy_construct():
        covered = np.zeros(base, dtype=np.int8)
        remaining = list(divisors)
        classes = {}
        
        while remaining:
            best_gain = -1
            best_m = None
            best_r = None
            
            for m in remaining:
                # We need gain for each r
                # gain[r] = sum(covered[indices_cache[m][r]] == 0)
                # Vectorize?
                # covered is 1D.
                # We can use the reshape trick which is faster than summing arrays from cache if cache is not used for greedy?
                # Actually, accessing cache arrays and summing is also fast.
                # Let's stick to reshape for greedy as it's a single operation per m.
                c_2d = covered.reshape(base // m, m)
                gains = (1 - c_2d).sum(axis=0)
                
                max_gain_idx = np.argmax(gains)
                if gains[max_gain_idx] > best_gain:
                    best_gain = gains[max_gain_idx]
                    best_m = m
                    best_r = max_gain_idx
            
            if best_m is None:
                break
                
            classes[best_m] = best_r
            covered[best_r::best_m] = 1
            remaining.remove(best_m)
            
        return classes

    def local_search(items):
        # items: list of [m, r]
        count = np.zeros(base, dtype=np.int32)
        for m, r in items:
            count[indices_cache[m][r]] += 1
        
        uncovered_count = int(np.sum(count == 0))
        best_score = uncovered_count
        best_items = [list(item) for item in items]
        
        improved = True
        max_passes = 100
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            # 1-moves
            for i in range(len(items)):
                m, r_old = items[i]
                best_r_new = r_old
                best_uncovered = uncovered_count
                
                idx_old = indices_cache[m][r_old]
                lost = int(np.sum(count[idx_old] == 1))
                
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    idx_new = indices_cache[m][r_new]
                    newly = int(np.sum(count[idx_new] == 0))
                    
                    new_uncovered = uncovered_count + lost - newly
                    
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r_new = r_new
                
                if best_r_new != r_old:
                    idx_old = indices_cache[m][r_old]
                    idx_new = indices_cache[m][best_r_new]
                    count[idx_old] -= 1
                    count[idx_new] += 1
                    items[i][1] = best_r_new
                    uncovered_count = best_uncovered
                    improved = True
                    if uncovered_count < best_score:
                        best_score = uncovered_count
                        best_items = [list(item) for item in items]
                        
        return best_items, best_score

    # Run Greedy
    classes_dict = greedy_construct()
    items = [[m, r] for m, r in classes_dict.items()]
    
    # Run Local Search
    items, score = local_search(items)
    
    # Add Random Restarts to improve robustness
    best_items = items
    best_score = score
    
    start_time = time.time()
    while time.time() - start_time < 90: # Leave 30s for safety
        # Perturb current best
        perturbed = [list(item) for item in best_items]
        # Randomly change 2-3 residues
        for _ in range(3):
            idx = random.randint(0, len(perturbed) - 1)
            m = perturbed[idx][0]
            perturbed[idx][1] = random.randint(0, m - 1)
        
        # Local search from perturbed state
        new_items, new_score = local_search(perturbed)
        
        if new_score < best_score:
            best_score = new_score
            best_items = new_items

    return best_items
