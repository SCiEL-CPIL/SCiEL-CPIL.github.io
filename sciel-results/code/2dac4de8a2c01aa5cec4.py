"""Max K(2,2,2)-free triple system on 16 points via precomputed config counts + local search."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    random.seed(42)

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all configurations (sets of 6 points partitioned into 3 pairs)
    # and which triples belong to which configurations.
    # A configuration is "satisfied" (violating) if all 8 transversal triples are present.
    # We maintain config_count[c] = number of transversals present for config c.
    # A triple can be added only if for all configs containing it, config_count[c] < 7.
    
    # Hardcoded 15 pairings of 6 indices into 3 pairs
    pairings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    # triple_configs[i] = list of config indices that contain triple i
    triple_configs = [[] for _ in range(num_triples)]
    config_count = []  # Will be filled per config with the 8 triple indices
    
    for vertices in itertools.combinations(range(N), 6):
        for p in pairings:
            a1, a2 = vertices[p[0][0]], vertices[p[0][1]]
            b1, b2 = vertices[p[1][0]], vertices[p[1][1]]
            c1, c2 = vertices[p[2][0]], vertices[p[2][1]]
            
            t_indices = []
            for a in (a1, a2):
                for b in (b1, b2):
                    for c in (c1, c2):
                        t = tuple(sorted((a, b, c)))
                        idx = triple_to_idx[t]
                        t_indices.append(idx)
            
            config_idx = len(config_count)
            config_count.append([0] * 8)  # counts for this config
            for idx in t_indices:
                triple_configs[idx].append(config_idx)
    
    num_configs = len(config_count)
    config_counts = [0] * num_configs  # current count of present triples per config
    in_set = [False] * num_triples
    
    def can_add(i):
        for c in triple_configs[i]:
            if config_counts[c] == 7:
                return False
        return True

    def do_add(i):
        in_set[i] = True
        for c in triple_configs[i]:
            config_counts[c] += 1

    def do_remove(i):
        in_set[i] = False
        for c in triple_configs[i]:
            config_counts[c] -= 1

    # --- Greedy Initialization ---
    indices = list(range(num_triples))
    random.shuffle(indices)
    for i in indices:
        if can_add(i):
            do_add(i)
    
    current_size = sum(in_set)
    best_size = current_size
    best_state = in_set[:]
    best_counts = config_counts[:]
    
    # --- Local Search ---
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any single missing triple
        # Use random order to escape local optima
        add_indices = [i for i in range(num_triples) if not in_set[i]]
        random.shuffle(add_indices)
        for i in add_indices:
            if can_add(i):
                do_add(i)
                current_size += 1
                improved = True
                break
        
        if improved:
            if current_size > best_size:
                best_size = current_size
                best_state = in_set[:]
                best_counts = config_counts[:]
            continue
        
        # Phase 2: Try 1-out, k-in swaps (remove one, try to add multiple)
        in_indices = [i for i in range(num_triples) if in_set[i]]
        if not in_indices:
            break
            
        # Randomly pick a triple to remove
        i_out = random.choice(in_indices)
        do_remove(i_out)
        current_size -= 1
        
        added_count = 0
        added_indices = []
        
        # Try to add as many as possible
        add_indices = [i for i in range(num_triples) if not in_set[i]]
        random.shuffle(add_indices)
        for i in add_indices:
            if can_add(i):
                do_add(i)
                added_count += 1
                added_indices.append(i)
                current_size += 1
        
        if added_count > 1:
            # Net gain
            improved = True
            if current_size > best_size:
                best_size = current_size
                best_state = in_set[:]
                best_counts = config_counts[:]
        else:
            # Revert
            for i in added_indices:
                do_remove(i)
                current_size -= 1
            do_add(i_out)
            current_size += 1
        
        if not improved:
            # Occasionally try a 2-out, 3-in swap or just continue
            # For now, just continue to try different random removals
            pass

    # Restore best state
    in_set = best_state
    config_counts = best_counts
    
    return [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
