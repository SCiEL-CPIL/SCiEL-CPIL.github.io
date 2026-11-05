"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

BASE = 51975

def _get_divisors():
    divs = []
    for d in range(2, BASE + 1):
        if BASE % d == 0:
            divs.append(d)
    return divs

DIVS = _get_divisors()

def _build_mask(m, r):
    """Boolean array: True at positions congruent to r mod m."""
    mask = np.zeros(BASE, dtype=bool)
    mask[r::m] = True
    return mask

def _score(covered):
    return int(np.sum(~covered))

def _greedy_init(order=None):
    if order is None:
        order = list(DIVS)
        random.shuffle(order)
    covered = np.zeros(BASE, dtype=bool)
    classes = []
    for m in order:
        best_r = 0
        best_cnt = -1
        for r in range(m):
            cnt = int(np.sum(~covered[r::m]))
            if cnt > best_cnt:
                best_cnt = cnt
                best_r = r
        if best_cnt > 0:
            classes.append([m, best_r])
            covered[best_r::m] = True
    return classes, covered

def _local_search(classes, covered, deadline):
    best_score = _score(covered)
    best_classes = [c[:] for c in classes]
    best_covered = covered.copy()
    no_improve = 0
    start = time.time()
    
    while time.time() < deadline and no_improve < 3000:
        if not classes:
            break
        idx = random.randint(0, len(classes) - 1)
        m, r = classes[idx]
        
        if random.random() < 0.5:
            # Change residue
            new_r = random.randint(0, m - 1)
            if new_r == r:
                continue
            tmp = best_covered.copy()
            tmp[r::m] = False
            tmp[new_r::m] = True
            ns = _score(tmp)
            if ns < best_score:
                best_score = ns
                best_classes[idx] = [m, new_r]
                best_covered = tmp
                classes[idx] = [m, new_r]
                covered = tmp
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Swap modulus
            used = set(c[0] for c in classes)
            unused = [d for d in DIVS if d not in used]
            if not unused:
                continue
            new_m = random.choice(unused)
            tmp = best_covered.copy()
            tmp[r::m] = False
            # Find best residue for new_m
            best_r2 = 0
            best_cnt = -1
            for r2 in range(new_m):
                cnt = int(np.sum(~tmp[r2::new_m]))
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_r2 = r2
            tmp[best_r2::new_m] = True
            ns = _score(tmp)
            if ns < best_score:
                best_score = ns
                best_classes[idx] = [new_m, best_r2]
                best_covered = tmp
                classes[idx] = [new_m, best_r2]
                covered = tmp
                no_improve = 0
            else:
                no_improve += 1
    
    return best_classes, best_score

def construct():
    deadline = time.time() + 135
    best_classes = None
    best_score = BASE + 1
    
    # Try several different greedy orderings
    attempts = 0
    while time.time() < deadline - 15:
        attempts += 1
        classes, covered = _greedy_init()
        score = _score(covered)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        classes, score = _local_search(classes, covered, deadline - 5)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        if best_score <= 1:
            break
    
    if best_classes is None:
        best_classes, _ = _greedy_init()
    
    return best_classes
# EVOLVE-BLOCK-END
