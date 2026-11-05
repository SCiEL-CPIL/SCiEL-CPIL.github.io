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

def _uncovered_count(classes, base=BASE):
    """Compute number of uncovered residues."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(covered == 0))

def _compute_covered(classes):
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return covered

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    divs = _divisors_of(BASE)
    divs = [d for d in divs if d > 1]
    
    def greedy_init(moduli_set, order='desc'):
        covered = np.zeros(BASE, dtype=np.uint8)
        classes = []
        if order == 'desc':
            sorted_mods = sorted(moduli_set, reverse=True)
        elif order == 'asc':
            sorted_mods = sorted(moduli_set)
        else:
            sorted_mods = list(moduli_set)
            random.shuffle(sorted_mods)
        for m in sorted_mods:
            best_r = 0
            best_count = -1
            for r in range(m):
                cnt = np.sum(covered[r::m] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes
    
    # Try multiple initializations
    best_classes = None
    best_score = BASE + 1
    
    for order in ['desc', 'asc', 'rand', 'rand', 'rand']:
        if time.time() >= deadline:
            break
        classes = greedy_init(divs, order=order)
        score = _uncovered_count(classes)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    if best_classes is None:
        best_classes = greedy_init(divs, order='desc')
        best_score = _uncovered_count(best_classes)
    
    # Local search with incremental coverage tracking
    current_classes = best_classes[:]
    current_covered = _compute_covered(current_classes)
    current_uncovered = int(np.sum(current_covered == 0))
    
    # For each class, precompute which residues it covers
    # We'll use a smarter local search: for each modulus, try all residues
    # but use incremental update
    
    improved = True
    max_rounds = 50
    round_num = 0
    while improved and time.time() < deadline and round_num < max_rounds:
        improved = False
        round_num += 1
        current_uncovered = int(np.sum(current_covered == 0))
        
        # Shuffle order for diversification
        indices = list(range(len(current_classes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() >= deadline:
                break
            m, r = current_classes[idx]
            
            # Find best residue for this modulus
            best_r = r
            best_uncovered = current_uncovered
            
            # Try all residues for small moduli, sample for large
            if m <= 300:
                candidate_rs = range(m)
            else:
                # Sample + current
                n_samples = min(m, 200)
                candidate_rs = set(random.sample(range(m), n_samples))
                candidate_rs.add(r)
            
            for r_new in candidate_rs:
                if r_new == r:
                    continue
                
                # Incremental update: 
                # Remove old coverage, add new coverage, count uncovered
                # But we need to be careful: a residue might be covered by multiple classes
                
                # Quick filter: count how many currently-uncovered residues would be covered
                newly_covered = np.sum(current_covered[r_new::m] == 0)
                if newly_covered == 0:
                    continue
                
                # Full check for promising candidates
                old_r = current_classes[idx][1]
                current_classes[idx][1] = r_new
                new_covered = _compute_covered(current_classes)
                new_uncovered = int(np.sum(new_covered == 0))
                current_classes[idx][1] = old_r
                
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    best_r = r_new
            
            if best_r != r:
                current_classes[idx][1] = best_r
                current_covered = _compute_covered(current_classes)
                current_uncovered = best_uncovered
                improved = True
                if current_uncovered < best_score:
                    best_score = current_uncovered
                    best_classes = current_classes[:]
    
    return best_classes
