import numpy as np
import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 130.0  # Safety margin: evaluator kills at 150s

    base = 51975
    divs = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    def time_left():
        return DEADLINE - time.time()

    def greedy_build(order):
        cover_count = np.zeros(base, dtype=np.int32)
        classes = []
        for m in order:
            # Find residue r that covers most currently uncovered
            zeros = np.where(cover_count == 0)[0]
            if len(zeros) == 0:
                classes.append([m, 0])
                continue
            
            # Count uncovered per residue class mod m
            A = np.bincount(zeros % m, minlength=m)
            best_r = int(np.argmax(A))
            classes.append([m, best_r])
            
            # Update cover_count
            cover_count[best_r::m] += 1
            
        # Calculate score
        score = base - int(np.sum(cover_count > 0))
        return classes, score, cover_count

    def local_search(classes, cover_count, cur_score):
        moduli = [c[0] for c in classes]
        residues = [c[1] for c in classes]
        n = len(moduli)
        
        # Precompute position arrays for each modulus
        pos_arrays = {}
        for i in range(n):
            m = moduli[i]
            pos_arrays[m] = [np.arange(r, base, m) for r in range(m)]
        
        improved = True
        while improved and time_left() > 3:
            improved = False
            # Randomize iteration order to avoid getting stuck
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                if time_left() < 1.5:
                    break
                m = moduli[i]
                r = residues[i]
                
                # Compute A: count of uncovered (cover_count==0) per residue mod m
                zeros = np.where(cover_count == 0)[0]
                if len(zeros) == 0:
                    break  # Full coverage achieved
                A = np.bincount(zeros % m, minlength=m)
                
                # Compute L_fixed: count of positions in current class P_r that are covered ONLY by this class
                old_pos = pos_arrays[m][r]
                L_fixed = int(np.sum(cover_count[old_pos] == 1))
                
                # Find best new residue r' != r
                A_temp = A.copy()
                A_temp[r] = -1
                best_r_prime = int(np.argmax(A_temp))
                best_gain = A_temp[best_r_prime] - L_fixed
                
                if best_gain > 0:
                    # Update
                    new_pos = pos_arrays[m][best_r_prime]
                    cover_count[old_pos] -= 1
                    cover_count[new_pos] += 1
                    residues[i] = best_r_prime
                    cur_score -= best_gain
                    improved = True
        
        return [[moduli[i], residues[i]] for i in range(n)], cur_score

    best_score = base
    best_classes = None
    best_cover_count = None

    # Try multiple orderings with different strategies
    n_divs = len(divs)
    orderings = []
    
    # Sorted ascending (small moduli first)
    orderings.append(sorted(divs))
    # Sorted descending (large moduli first)
    orderings.append(sorted(divs, reverse=True))
    
    # Multiple random orderings
    num_random = 8
    for _ in range(num_random):
        orderings.append(sorted(divs, key=lambda x: random.random()))
    
    # Hybrid: mix of sorted and random
    for _ in range(3):
        order = list(divs)
        random.shuffle(order)
        # Sort first half ascending, second half descending
        half = len(order) // 2
        order[:half] = sorted(order[:half])
        order[half:] = sorted(order[half:], reverse=True)
        orderings.append(order)
    
    for order in orderings:
        if time_left() < 8:
            break
        
        classes, score, cover_count = greedy_build(order)
        if score < best_score:
            best_score = score
            best_classes = classes
            best_cover_count = cover_count.copy()
        
        # Local search
        if time_left() > 8:
            classes, score = local_search(classes, cover_count, score)
            if score < best_score:
                best_score = score
                best_classes = classes
                best_cover_count = cover_count.copy()
        
        # Additional local search passes if time permits
        if time_left() > 15 and score < best_score:
            classes, score = local_search(classes, cover_count, score)
            if score < best_score:
                best_score = score
                best_classes = classes
                best_cover_count = cover_count.copy()

    return best_classes if best_classes is not None else [[3, 0]]
