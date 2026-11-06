import random
import time
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute K(2,2,2) configurations
    # Each config is defined by 3 disjoint pairs, yielding 8 transversal triples
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
            kconfigs.append(tuple(sorted(transversals_canon)))
            for t in transversals_canon:
                triple_to_kconfigs[t].append(len(kconfigs) - 1)
    
    num_kconfigs = len(kconfigs)
    
    # Helper to check if a triple can be added to current state
    # present: set of triples
    # counts: list of counts for each kconfig
    def can_add(t, present, counts):
        if t in present:
            return False
        for kidx in triple_to_kconfigs[t]:
            if counts[kidx] == 7:
                return False
        return True

    def add_triple(t, present, counts):
        present.add(t)
        for kidx in triple_to_kconfigs[t]:
            counts[kidx] += 1

    def remove_triple(t, present, counts):
        present.remove(t)
        for kidx in triple_to_kconfigs[t]:
            counts[kidx] -= 1

    def greedy_init(seed_val):
        random.seed(seed_val)
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        present = set()
        counts = [0] * num_kconfigs
        for t in shuffled:
            if can_add(t, present, counts):
                add_triple(t, present, counts)
        return present, counts

    # 1. Find a good initial solution using multiple random greedy restarts
    best_present = set()
    best_counts = [0] * num_kconfigs
    best_size = 0
    
    # Try a few seeds to get a strong start
    for i in range(20):
        p, c = greedy_init(i)
        if len(p) > best_size:
            best_size = len(p)
            best_present = p
            best_counts = c

    current_present = best_present
    current_counts = best_counts[:]
    current_size = len(current_present)
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 110.0
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    # We need to be able to pick a random triple from present or absent quickly
    # present is a set. all_triples is a list.
    # To pick a random absent triple, we can't just sample from all_triples and check membership
    # if the set is dense. But N=560, so it's fine.
    
    while time.time() - start_time < time_limit and T > min_T:
        # Choose a move type: Add, Remove, Swap
        # Probability of swap should be high to maintain size while exploring
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random triple
            t = random.choice(all_triples)
            if t in current_present:
                continue
            if can_add(t, current_present, current_counts):
                # Always accept improvement
                add_triple(t, current_present, current_counts)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_present = set(current_present)
                    best_counts = current_counts[:]
            else:
                # Cannot add, do nothing or try swap? 
                # In SA, we usually propose a move and accept/reject based on energy.
                # Here, invalid moves are just skipped or rejected.
                # Let's treat "Add" as a proposal. If invalid, reject.
                pass
                
        elif move_type < 0.7:
            # Try to remove a random triple
            if not current_present:
                continue
            t = random.choice(list(current_present))
            # Remove is always valid, but reduces size.
            # Accept with probability exp(-delta_E / T) where delta_E = -1 (energy increases by 1? No, we maximize size).
            # Energy = -size. Delta E = - (new_size - old_size) = - (-1) = 1.
            # P(accept) = exp(-1/T).
            if random.expovariate(1/T) < 1: # This is wrong logic for acceptance.
                 pass
            # Correct SA acceptance:
            # delta_energy = -1 (size decreases by 1, so "cost" increases by 1 if we define cost = -size)
            # Actually, let's define Value = size.
            # Delta Value = -1.
            # Accept with prob exp(Delta Value / T) = exp(-1/T).
            if random.random() < (2.71828 ** (-1.0 / T)):
                remove_triple(t, current_present, current_counts)
                current_size -= 1
        else:
            # Swap: Remove one, Add another
            if not current_present:
                continue
            t_remove = random.choice(list(current_present))
            t_add = random.choice(all_triples)
            
            if t_add in current_present:
                continue
                
            # Check if we can swap
            # Temporarily remove t_remove
            remove_triple(t_remove, current_present, current_counts)
            
            if can_add(t_add, current_present, current_counts):
                add_triple(t_add, current_present, current_counts)
                # Size remains same. 
                # In SA, we usually accept neutral moves or slightly worse ones?
                # Here, size is same. We accept to explore.
                # But we should track if this leads to a better state later.
                # For now, accept swap to change configuration.
                # However, if we just swap, we might loop.
                # Let's accept swap with high probability to move around the landscape.
                pass
            else:
                # Revert
                add_triple(t_remove, current_present, current_counts)
                
        T *= cooling_rate

    return [list(t) for t in best_present]
