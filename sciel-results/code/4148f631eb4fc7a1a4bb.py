import numpy as np
import time
import random
import math

BASE = 51975

def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted([d for d in divs if d > 1])

def _greedy_init(mods_sorted):
    """Fast greedy initialization: pick residue that maximizes uncovered coverage."""
    ref_count = np.zeros(BASE, dtype=np.int32)
    classes = []
    
    for m in mods_sorted:
        best_r = 0
        best_gain = -1
        uncovered_mask = (ref_count == 0)
        if m <= 1000:
            positions = np.where(uncovered_mask)[0]
            if len(positions) > 0:
                residues = positions % m
                gains, _ = np.histogram(residues, bins=m, range=(0, m))
                best_r = int(np.argmax(gains))
                best_gain = int(gains[best_r])
            else:
                best_gain = 0
        else:
            for r in range(m):
                gain = np.sum(uncovered_mask[r::m])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
        classes.append([m, best_r])
        ref_count[best_r::m] += 1
        
    return classes, ref_count

def _simulated_annealing(classes, ref_count, deadline, T0=5.0, T_min=0.01, cooling=0.9998):
    """Simulated annealing with fast move evaluation."""
    current_uncovered = int(np.sum(ref_count == 0))
    best_uncovered = current_uncovered
    best_classes = [c[:] for c in classes]
    
    n_classes = len(classes)
    indices = list(range(n_classes))
    T = T0
    
    while time.time() < deadline and T > T_min:
        idx = random.choice(indices)
        m, r_old = classes[idx]
        r_new = random.randint(0, m - 1)
        if r_new == r_old:
            continue
        
        old_slice = ref_count[r_old::m]
        new_slice = ref_count[r_new::m]
        
        loss = np.count_nonzero(old_slice == 1)
        gain = np.count_nonzero(new_slice == 0)
        delta = loss - gain
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            ref_count[r_old::m] -= 1
            ref_count[r_new::m] += 1
            current_uncovered += delta
            classes[idx][1] = r_new
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
        
        T *= cooling
    
    return best_uncovered, best_classes

def _polish(classes, ref_count, deadline):
    """Final polish: try all residues for each modulus, keep best."""
    current_uncovered = int(np.sum(ref_count == 0))
    best_uncovered = current_uncovered
    best_classes = [c[:] for c in classes]
    
    n_classes = len(classes)
    
    for idx in range(n_classes):
        if time.time() >= deadline:
            break
        m, r_old = classes[idx]
        best_r = r_old
        best_delta = 0
        
        for r_new in range(m):
            if r_new == r_old:
                continue
            old_slice = ref_count[r_old::m]
            new_slice = ref_count[r_new::m]
            loss = np.count_nonzero(old_slice == 1)
            gain = np.count_nonzero(new_slice == 0)
            delta = loss - gain
            if delta < best_delta:
                best_delta = delta
                best_r = r_new
        
        if best_r != r_old:
            ref_count[r_old::m] -= 1
            ref_count[best_r::m] += 1
            current_uncovered += best_delta
            classes[idx][1] = best_r
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
    
    return best_uncovered, best_classes

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    divs = _divisors_of(BASE)
    mods_sorted = sorted(divs, reverse=True)
    
    best_uncovered_global = float('inf')
    best_classes_global = None
    
    polish_time = 35
    sa_deadline = deadline - polish_time
    
    num_restarts = 4
    time_per_restart = (sa_deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() >= sa_deadline:
            break
        
        classes, ref_count = _greedy_init(mods_sorted)
        
        restart_deadline = start_time + (restart + 1) * time_per_restart
        uncovered, classes_sa = _simulated_annealing(
            classes, ref_count, restart_deadline,
            T0=5.0, T_min=0.01, cooling=0.9998
        )
        
        uncovered_polished, classes_polished = _polish(
            classes_sa, ref_count, deadline
        )
        
        if uncovered_polished < best_uncovered_global:
            best_uncovered_global = uncovered_polished
            best_classes_global = [c[:] for c in classes_polished]
    
    return best_classes_global
