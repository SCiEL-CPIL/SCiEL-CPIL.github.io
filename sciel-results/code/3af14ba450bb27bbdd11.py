import random
import time
import numpy as np
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples) # 560
    
    # Map triple tuple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    triple_to_kconfigs = [[] for _ in range(num_triples)]
    kconfig_masks = [] # List of ints (bitmasks)
    
    def partitions_into_pairs(points):
        a = points[0]
        for i in range(1, 6):
            b = points[i]
            remaining = [p for j, p in enumerate(points) if j not in (0, i)]
            c, d, e, f = remaining
            yield (a, b), (c, d), (e, f)
            yield (a, b), (c, e), (d, f)
            yield (a, b), (c, f), (d, e)
            
    for six_points in combinations(range(n), 6):
        for pairs in partitions_into_pairs(six_points):
            (a, a_prime), (b, b_prime), (c, c_prime) = pairs
            transversals = set()
            for x in (a, a_prime):
                for y in (b, b_prime):
                    for z in (c, c_prime):
                        transversals.add(tuple(sorted((x, y, z))))
            
            mask = 0
            for t in transversals:
                idx = triple_to_idx[t]
                mask |= (1 << idx)
                triple_to_kconfigs[idx].append(len(kconfig_masks))
            
            kconfig_masks.append(mask)
            
    num_kconfigs = len(kconfig_masks)
    
    # Greedy Initial Solution
    def greedy_solve(order):
        present = set()
        kconfig_counts = np.zeros(num_kconfigs, dtype=np.int32)
        
        for t_idx in order:
            cfg_indices = triple_to_kconfigs[t_idx]
            if not cfg_indices:
                present.add(t_idx)
                continue
            counts = kconfig_counts[cfg_indices]
            if not np.any(counts == 7):
                present.add(t_idx)
                kconfig_counts[cfg_indices] += 1
                
        return present, kconfig_counts

    # Local Improve: Add any possible triple
    def local_improve(present, kconfig_counts):
        improved = True
        while improved:
            improved = False
            for t_idx in range(num_triples):
                if t_idx in present:
                    continue
                cfg_indices = triple_to_kconfigs[t_idx]
                if not cfg_indices:
                    present.add(t_idx)
                    improved = True
                    continue
                counts = kconfig_counts[cfg_indices]
                if not np.any(counts == 7):
                    present.add(t_idx)
                    kconfig_counts[cfg_indices] += 1
                    improved = True
        return present, kconfig_counts

    best_count = 0
    best_triples = None
    
    random.seed(42)
    
    # Run multiple greedy starts
    for _ in range(20):
        order = list(range(num_triples))
        random.shuffle(order)
        present, kconfig_counts = greedy_solve(order)
        present, kconfig_counts = local_improve(present, kconfig_counts)
        
        if len(present) > best_count:
            best_count = len(present)
            best_triples = [list(all_triples[t]) for t in present]
            
    # Simulated Annealing
    start_time = time.time()
    time_limit = 115.0
    
    # Start SA from the best greedy solution
    # We need to re-derive the state for the best solution
    # Since we only stored the list, we reconstruct
    # Better: Keep the best state from the loop above
    # Let's just pick the last one or re-run best order?
    # Re-running is safer.
    
    # Find best order
    best_order = None
    best_len = -1
    for _ in range(5):
        order = list(range(num_triples))
        random.shuffle(order)
        p, c = greedy_solve(order)
        p, c = local_improve(p, c)
        if len(p) > best_len:
            best_len = len(p)
            best_order = order
            
    current_present, current_counts = greedy_solve(best_order)
    current_present, current_counts = local_improve(current_present, current_counts)
    
    if len(current_present) > best_count:
        best_count = len(current_present)
        best_triples = [list(all_triples[t]) for t in current_present]

    T = 1.0
    T_min = 0.01
    cooling = 0.99995
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate move: Add or Remove
        if random.random() < 0.6: # Bias towards adding
            # Pick random missing triple
            missing = [t for t in range(num_triples) if t not in current_present]
            if not missing:
                continue
            t_idx = random.choice(missing)
            cfg_indices = triple_to_kconfigs[t_idx]
            if cfg_indices:
                counts = current_counts[cfg_indices]
                if np.any(counts == 7):
                    continue # Invalid
            # Valid Add
            current_present.add(t_idx)
            current_counts[cfg_indices] += 1
            # Accept (Improvement)
        else:
            # Try Remove
            if not current_present:
                continue
            t_idx = random.choice(list(current_present))
            cfg_indices = triple_to_kconfigs[t_idx]
            current_present.remove(t_idx)
            current_counts[cfg_indices] -= 1
            # Accept with probability exp(-1/T)
            if random.random() >= np.exp(-1.0 / T):
                # Reject, put back
                current_present.add(t_idx)
                current_counts[cfg_indices] += 1
            else:
                # Accept removal
                pass
            
        if len(current_present) > best_count:
            best_count = len(current_present)
            best_triples = [list(all_triples[t]) for t in current_present]
            
        T *= cooling
        
    return best_triples
