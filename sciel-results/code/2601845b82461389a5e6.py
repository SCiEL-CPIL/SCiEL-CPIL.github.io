"""Triple systems on 16 points without K(2,2,2) - local search with precomputed conflicts."""
import time
import random
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is formed by 3 disjoint pairs of points: {a,b}, {c,d}, {e,f}
    # The 8 transversal triples are all ways to pick one from each pair
    # For a given 6-set {a,b,c,d,e,f}, there are 3 ways to partition into 3 pairs
    # We need to track: for each configuration (defined by the 6 points and the partition),
    # how many of the 8 transversals are present
    
    # Actually, let's think differently. A K(2,2,2) configuration is defined by 6 distinct points
    # partitioned into 3 pairs. The 8 transversals are the triples picking one from each pair.
    # If all 8 are present, we have a forbidden configuration.
    
    # Let's precompute all possible K(2,2,2) configurations
    # Each configuration is a set of 8 triples (the transversals)
    # Two triples conflict if they are part of some K(2,2,2) where all 8 transversals would be present
    
    # Better approach: precompute for each pair of triples, whether they can be part of a K(2,2,2)
    # Actually, the conflict is: adding triple T creates a K(2,2,2) if there exist 6 points
    # partitioned into 3 pairs such that T is one of the 8 transversals and the other 7 are already present.
    
    # Let's precompute all K(2,2,2) configurations as sets of 8 triples
    # There are C(16,6) * 3 ways to partition 6 points into 3 pairs = 8008 * 3 = 24024 configurations
    # For each, the 8 transversals are determined
    
    configs = []
    config_triples = []  # list of sets of 8 triple indices
    
    for six in combinations(range(n), 6):
        a, b, c, d, e, f = six
        # 3 ways to partition into 3 pairs
        partitions = [
            ((a,b),(c,d),(e,f)),
            ((a,c),(b,d),(e,f)),
            ((a,d),(b,c),(e,f)),
        ]
        for p1, p2, p3 in partitions:
            transversals = set()
            for x in p1:
                for y in p2:
                    for z in p3:
                        transversals.add(tuple(sorted((x,y,z))))
            # Convert to indices
            indices = frozenset(triple_index[t] for t in transversals)
            configs.append(indices)
            config_triples.append(indices)
    
    num_configs = len(configs)
    
    # For each triple, list of config indices that contain it
    triple_to_configs = [[] for _ in range(num_triples)]
    for ci, cfg in enumerate(config_triples):
        for ti in cfg:
            triple_to_configs[ti].append(ci)
    
    # For each config, track how many of its 8 triples are present
    # State: present[] boolean array, config_count[ci] = number of present triples in config ci
    # Adding triple ti: for each ci in triple_to_configs[ti], config_count[ci] += 1
    # If config_count[ci] becomes 8, we have a violation
    # Removing triple ti: for each ci in triple_to_configs[ti], config_count[ci] -= 1
    
    deadline = time.time() + 110  # safety margin
    
    # Greedy initialization
    random.seed(42)
    present = [False] * num_triples
    config_count = [0] * num_configs
    best_size = 0
    best_present = None
    
    def try_add(ti):
        nonlocal best_size, best_present
        if present[ti]:
            return False
        # Check if adding ti would create a violation
        for ci in triple_to_configs[ti]:
            if config_count[ci] == 7:
                return False
        present[ti] = True
        for ci in triple_to_configs[ti]:
            config_count[ci] += 1
        return True
    
    def remove(ti):
        if not present[ti]:
            return
        present[ti] = False
        for ci in triple_to_configs[ti]:
            config_count[ci] -= 1
    
    # Greedy: add triples in random order
    order = list(range(num_triples))
    random.shuffle(order)
    for ti in order:
        if try_add(ti):
            best_size += 1
    
    best_present = present[:]
    best_size = sum(present)
    
    # Local search: try swaps and additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any missing triple
        for ti in range(num_triples):
            if not present[ti]:
                if try_add(ti):
                    best_size += 1
                    improved = True
                    if time.time() > deadline:
                        break
        if time.time() > deadline:
            break
        
        # Try swaps: remove one, add two
        if not improved:
            for ti_remove in range(num_triples):
                if not present[ti_remove]:
                    continue
                # Temporarily remove
                remove(ti_remove)
                added = 0
                added_list = []
                for ti_add in range(num_triples):
                    if ti_add == ti_remove or present[ti_add]:
                        continue
                    if try_add(ti_add):
                        added += 1
                        added_list.append(ti_add)
                        if added >= 2:
                            break
                # Restore
                for ti_a in added_list:
                    remove(ti_a)
                present[ti_remove] = True
                for ci in triple_to_configs[ti_remove]:
                    config_count[ci] += 1
                
                if added >= 2:
                    # Do the swap for real
                    remove(ti_remove)
                    for ti_a in added_list:
                        try_add(ti_a)
                    best_size = sum(present)
                    improved = True
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        # Try removing one to allow adding more than one
        if not improved:
            for ti_remove in range(num_triples):
                if not present[ti_remove]:
                    continue
                remove(ti_remove)
                added = 0
                added_list = []
                for ti_add in range(num_triples):
                    if ti_add == ti_remove or present[ti_add]:
                        continue
                    if try_add(ti_add):
                        added += 1
                        added_list.append(ti_add)
                # Restore
                for ti_a in added_list:
                    remove(ti_a)
                present[ti_remove] = True
                for ci in triple_to_configs[ti_remove]:
                    config_count[ci] += 1
                
                if added > 1:
                    remove(ti_remove)
                    for ti_a in added_list:
                        try_add(ti_a)
                    best_size = sum(present)
                    improved = True
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Extract result
    result = [list(all_triples[ti]) for ti in range(num_triples) if present[ti]]
    return result
