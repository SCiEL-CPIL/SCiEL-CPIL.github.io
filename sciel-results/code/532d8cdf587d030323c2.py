import random
import time
import math
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute K(2,2,2) configurations
    triple_to_kconfigs = {t: [] for t in all_triples}
    kconfigs = []
    
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
            transversals_canon = set()
            for x in (a, a_prime):
                for y in (b, b_prime):
                    for z in (c, c_prime):
                        transversals_canon.add(tuple(sorted((x, y, z))))
            kconfigs.append(transversals_canon)
            for t in transversals_canon:
                triple_to_kconfigs[t].append(len(kconfigs) - 1)
    
    num_kconfigs = len(kconfigs)
    
    def greedy_solve(order):
        present = set()
        kconfig_counts = [0] * num_kconfigs
        for t in order:
            can_add = True
            for kidx in triple_to_kconfigs[t]:
                if kconfig_counts[kidx] == 7:
                    can_add = False
                    break
            if not can_add:
                continue
            present.add(t)
            for kidx in triple_to_kconfigs[t]:
                kconfig_counts[kidx] += 1
        return present, kconfig_counts

    # Initial best solution
    best_count = 0
    best_triples = None
    
    # Warm start with multiple random greedy attempts
    random.seed(42)
    for _ in range(5):
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        present, kconfig_counts = greedy_solve(shuffled)
        # Simple local improvement: add any possible
        improved = True
        while improved:
            improved = False
            for t in all_triples:
                if t in present:
                    continue
                can_add = True
                for kidx in triple_to_kconfigs[t]:
                    if kconfig_counts[kidx] == 7:
                        can_add = False
                        break
                if can_add:
                    present.add(t)
                    for kidx in triple_to_kconfigs[t]:
                        kconfig_counts[kidx] += 1
                    improved = True
        
        if len(present) > best_count:
            best_count = len(present)
            best_triples = [list(t) for t in present]
            best_present_set = set(present)
            best_counts = kconfig_counts[:]

    # Simulated Annealing
    start_time = time.time()
    time_limit = 115.0
    
    current_present = set(best_present_set)
    current_counts = best_counts[:]
    current_size = len(current_present)
    
    T = 1.0
    T_min = 0.05
    cooling = 0.9995
    
    # Precompute list of triples for fast random choice
    triples_list = all_triples
    
    while time.time() - start_time < time_limit:
        t = random.choice(triples_list)
        
        if t in current_present:
            # Remove
            current_present.remove(t)
            for kidx in triple_to_kconfigs[t]:
                current_counts[kidx] -= 1
            delta = -1
            new_size = current_size - 1
        else:
            # Try to add
            can_add = True
            for kidx in triple_to_kconfigs[t]:
                if current_counts[kidx] == 7:
                    can_add = False
                    break
            if can_add:
                current_present.add(t)
                for kidx in triple_to_kconfigs[t]:
                    current_counts[kidx] += 1
                delta = 1
                new_size = current_size + 1
            else:
                # Invalid move, skip
                continue
        
        # Acceptance
        if delta > 0:
            current_size = new_size
            if current_size > best_count:
                best_count = current_size
                best_triples = [list(x) for x in current_present]
        else:
            # delta == -1
            # Accept with probability exp(-1/T)
            if random.random() < math.exp(-1.0 / T):
                current_size = new_size
            else:
                # Reject: Undo removal
                current_present.add(t)
                for kidx in triple_to_kconfigs[t]:
                    current_counts[kidx] += 1
        
        T *= cooling
        if T < T_min:
            T = T_min

    return best_triples
