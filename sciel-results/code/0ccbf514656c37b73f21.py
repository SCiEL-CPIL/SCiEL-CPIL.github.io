import numpy as np
import time
import random
from math import gcd

BASE = 51975

def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted([d for d in divs if d > 1])

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    divs = _divisors_of(BASE)
    all_moduli = divs
    n_moduli = len(all_moduli)
    
    def greedy_init(order='desc'):
        covered = np.zeros(BASE, dtype=np.uint8)
        classes = []
        
        if order == 'desc':
            mods_to_use = sorted(all_moduli, reverse=True)
        elif order == 'asc':
            mods_to_use = sorted(all_moduli)
        else:
            mods_to_use = all_moduli[:]
            random.shuffle(mods_to_use)
            
        for m in mods_to_use:
            unc_mask = (covered == 0)
            indices_unc = np.where(unc_mask)[0]
            if len(indices_unc) == 0:
                classes.append([m, 0])
                continue
                
            gains = np.bincount(indices_unc % m, minlength=m)
            best_r = int(np.argmax(gains))
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes

    def compute_ref_count(classes):
        ref_count = np.zeros(BASE, dtype=np.int32)
        for m, r in classes:
            ref_count[r::m] += 1
        return ref_count

    def local_search_sa(classes, time_limit):
        ref_count = compute_ref_count(classes)
        uncovered_mask = (ref_count == 0)
        uncovered = int(np.sum(uncovered_mask))
        best_classes = [c[:] for c in classes]
        best_uncovered = uncovered
        
        n_classes = len(classes)
        indices = list(range(n_classes))
        
        # Simulated annealing parameters
        temp = 10.0
        cooling = 0.999
        min_temp = 0.01
        
        improved = True
        while time.time() < time_limit:
            if temp < min_temp:
                temp = min_temp
            
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() >= time_limit:
                    break
                
                m, r_old = classes[idx]
                pos_old = np.arange(r_old, BASE, m)
                loss = int(np.sum(ref_count[pos_old] == 1))
                
                unc_indices = np.where(uncovered_mask)[0]
                if len(unc_indices) == 0:
                    return best_classes, 0
                
                gains = np.bincount(unc_indices % m, minlength=m)
                best_r = int(np.argmax(gains))
                best_gain = int(gains[best_r])
                
                if best_r == r_old:
                    continue
                
                delta = loss - best_gain  # positive means worsening
                new_uncovered = uncovered + delta
                
                # Acceptance criterion
                if delta <= 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / temp)
                
                if accept:
                    pos_new = np.arange(best_r, BASE, m)
                    ref_count[pos_old] -= 1
                    ref_count[pos_new] += 1
                    uncovered_mask = (ref_count == 0)
                    uncovered = new_uncovered
                    classes[idx][1] = best_r
                    
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_classes = [c[:] for c in classes]
                        if uncovered == 0:
                            return best_classes, 0
                    
                    temp = max(temp * cooling, min_temp)
            
            # Occasionally try 2-swap moves
            if random.random() < 0.3 and n_classes >= 2:
                i, j = random.sample(range(n_classes), 2)
                mi, ri = classes[i]
                mj, rj = classes[j]
                
                # Swap residues
                classes[i][1] = rj
                classes[j][1] = ri
                
                new_ref = compute_ref_count(classes)
                new_unc = int(np.sum(new_ref == 0))
                
                delta = new_unc - uncovered
                if delta <= 0:
                    ref_count = new_ref
                    uncovered_mask = (new_ref == 0)
                    uncovered = new_unc
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_classes = [c[:] for c in classes]
                        if uncovered == 0:
                            return best_classes, 0
                else:
                    # Revert
                    classes[i][1] = ri
                    classes[j][1] = rj
                    accept = random.random() < np.exp(-delta / temp)
                    if accept:
                        ref_count = new_ref
                        uncovered_mask = (new_ref == 0)
                        uncovered = new_unc
                        if uncovered < best_uncovered:
                            best_uncovered = uncovered
                            best_classes = [c[:] for c in classes]
                
                temp = max(temp * cooling, min_temp)
        
        return best_classes, best_uncovered

    best_classes_global = None
    best_score_global = BASE + 1
    
    # Multi-start with different strategies
    strategies = ['desc', 'asc', 'rand', 'rand', 'rand', 'rand']
    
    for strat in strategies:
        if time.time() >= deadline - 5:
            break
            
        time_left = deadline - time.time()
        if time_left < 3:
            continue
            
        classes = greedy_init(order=strat)
        
        search_limit = time.time() + (time_left * 0.95)
        
        res_classes, res_score = local_search_sa(classes, search_limit)
        
        if res_score < best_score_global:
            best_score_global = res_score
            best_classes_global = res_classes
            if best_score_global == 0:
                break
                
    if best_classes_global is None:
        best_classes_global = greedy_init(order='desc')
        
    return best_classes_global
