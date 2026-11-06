import time
import math
import random
import numpy as np

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit to be safe
    
    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    
    divisors.sort()
    
    # Precompute indices for each (m, r) to speed up updates
    # indices[m][r] = numpy array of residues covered by class (m, r)
    indices = {}
    for m in divisors:
        indices[m] = []
        for r in range(m):
            indices[m].append(np.arange(r, base, m))

    best_score = base
    best_classes = []
    
    def greedy_and_local_search(random_seed):
        random.seed(random_seed)
        # Shuffle divisors for randomized greedy
        shuffled_divs = divisors.copy()
        random.shuffle(shuffled_divs)
        
        classes = []
        coverage = np.zeros(base, dtype=np.int8)
        score = base
        used_moduli = set()
        
        # Greedy Phase
        remaining_moduli = list(shuffled_divs)
        
        while remaining_moduli:
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in remaining_moduli:
                for r in range(m):
                    idxs = indices[m][r]
                    gain = np.count_nonzero(coverage[idxs] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
            
            if best_gain == 0:
                break
                
            classes.append([best_m, best_r])
            used_moduli.add(best_m)
            remaining_moduli.remove(best_m)
            
            # Update coverage
            idxs = indices[best_m][best_r]
            mask = (coverage[idxs] == 0)
            score -= np.count_nonzero(mask)
            coverage[idxs] += 1
            
            if time.time() > deadline - 20.0:
                break
        
        # Local Search Phase
        local_deadline = min(deadline - 5.0, start_time + 148.0)
        
        def apply_change(m, r, delta_sign):
            nonlocal score
            idxs = indices[m][r]
            if delta_sign == 1:
                mask = (coverage[idxs] == 0)
                score -= np.count_nonzero(mask)
                coverage[idxs] += 1
            else:
                mask = (coverage[idxs] == 1)
                score += np.count_nonzero(mask)
                coverage[idxs] -= 1

        improved = True
        while improved and time.time() < local_deadline:
            improved = False
            # Perform a batch of random moves
            moves_per_batch = 2000
            for _ in range(moves_per_batch):
                if time.time() > local_deadline:
                    break
                
                move_type = random.choice(['residue', 'swap', 'add', 'remove'])
                
                if move_type == 'residue' and classes:
                    i = random.randint(0, len(classes) - 1)
                    m, r = classes[i]
                    r_new = random.randint(0, m - 1)
                    if r_new == r: continue
                    
                    apply_change(m, r, -1)
                    apply_change(m, r_new, 1)
                    
                    if score < best_score:
                        best_score = score
                        classes[i] = [m, r_new]
                        improved = True
                    else:
                        apply_change(m, r_new, -1)
                        apply_change(m, r, 1)
                        
                elif move_type == 'swap' and classes:
                    i = random.randint(0, len(classes) - 1)
                    m_out, r_out = classes[i]
                    unused = [m for m in divisors if m not in used_moduli]
                    if not unused: continue
                    m_in = random.choice(unused)
                    r_in = random.randint(0, m_in - 1)
                    
                    apply_change(m_out, r_out, -1)
                    apply_change(m_in, r_in, 1)
                    
                    if score < best_score:
                        best_score = score
                        classes[i] = [m_in, r_in]
                        used_moduli.remove(m_out)
                        used_moduli.add(m_in)
                        improved = True
                    else:
                        apply_change(m_in, r_in, -1)
                        apply_change(m_out, r_out, 1)

                elif move_type == 'add':
                    unused = [m for m in divisors if m not in used_moduli]
                    if not unused: continue
                    m_in = random.choice(unused)
                    r_in = random.randint(0, m_in - 1)
                    
                    apply_change(m_in, r_in, 1)
                    
                    if score < best_score:
                        best_score = score
                        classes.append([m_in, r_in])
                        used_moduli.add(m_in)
                        improved = True
                    else:
                        apply_change(m_in, r_in, -1)

                elif move_type == 'remove' and classes:
                    i = random.randint(0, len(classes) - 1)
                    m_out, r_out = classes[i]
                    
                    apply_change(m_out, r_out, -1)
                    
                    if score < best_score:
                        best_score = score
                        classes.pop(i)
                        used_moduli.remove(m_out)
                        improved = True
                    else:
                        apply_change(m_out, r_out, 1)

        return classes, score

    # Run multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        classes, score = greedy_and_local_search(i)
        if score < best_score:
            best_score = score
            best_classes = classes

    return best_classes
