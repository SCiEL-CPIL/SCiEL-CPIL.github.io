"""
Triple systems on 16 points without K(2,2,2) - using max-flow / LP relaxation approach.

Strategy:
- Model as a maximum set packing problem: select max triples such that no K(2,2,2) config has all 8 transversals.
- This is equivalent to: for each of the 120,120 K(2,2,2) configs, at most 7 of its 8 transversals can be selected.
- Use a greedy + local search with fast conflict checking via precomputed config counts.
- Add a simulated annealing / stochastic restart layer for escaping local optima.
- Use numpy for vectorized operations where possible.
"""
import random
import time
import sys
from itertools import combinations

def construct():
    n = 16
    deadline = time.time() + 115  # safety margin before 150s limit
    
    # Precompute all triples
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    # Each config is a frozenset of 8 transversals (as tuple indices)
    configs = []
    triple_to_configs = [[] for _ in range(num_triples)]
    
    for six in combinations(range(n), 6):
        v0, v1, v2, v3, v4, v5 = six
        partitions = [
            ((v0,v1),(v2,v3),(v4,v5)),
            ((v0,v1),(v2,v4),(v3,v5)),
            ((v0,v1),(v2,v5),(v3,v4)),
            ((v0,v2),(v1,v3),(v4,v5)),
            ((v0,v2),(v1,v4),(v3,v5)),
            ((v0,v2),(v1,v5),(v3,v4)),
            ((v0,v3),(v1,v2),(v4,v5)),
            ((v0,v3),(v1,v4),(v2,v5)),
            ((v0,v3),(v1,v5),(v2,v4)),
            ((v0,v4),(v1,v2),(v3,v5)),
            ((v0,v4),(v1,v3),(v2,v5)),
            ((v0,v4),(v1,v5),(v2,v3)),
            ((v0,v5),(v1,v2),(v3,v4)),
            ((v0,v5),(v1,v3),(v2,v4)),
            ((v0,v5),(v1,v4),(v2,v3)),
        ]
        for parts in partitions:
            p1, p2, p3 = parts
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(triple_index[t])
            config_idx = len(configs)
            configs.append(transversals)
            for ti in transversals:
                triple_to_configs[ti].append(config_idx)
    
    num_configs = len(configs)  # 120120
    
    # For each triple, precompute the list of configs it belongs to
    # triple_to_configs is already built above
    
    def try_add(selected, config_counts, t_idx):
        """Try to add triple t_idx. Returns True if successful."""
        ok = True
        for ci in triple_to_configs[t_idx]:
            config_counts[ci] += 1
            if config_counts[ci] == 8:
                ok = False
                break
        if ok:
            selected[t_idx] = 1
            return True
        else:
            # Roll back
            for ci in triple_to_configs[t_idx]:
                config_counts[ci] -= 1
            return False
    
    def remove_triple(selected, config_counts, t_idx):
        """Remove triple t_idx."""
        selected[t_idx] = 0
        for ci in triple_to_configs[t_idx]:
            config_counts[ci] -= 1
    
    best_selected = None
    best_size = 0
    
    # Multiple random restarts with greedy + local search
    random.seed(12345)
    
    start_time = time.time()
    restart = 0
    
    while time.time() < deadline:
        restart += 1
        
        # Greedy initialization with random ordering
        order = list(range(num_triples))
        random.shuffle(order)
        
        selected = [0] * num_triples
        config_counts = [0] * num_configs
        
        for t_idx in order:
            try_add(selected, config_counts, t_idx)
        
        current_size = sum(selected)
        
        # Local search: try swaps (remove one, add another)
        # and add-only moves
        step = 0
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline and no_improve < max_no_improve:
            step += 1
            
            if random.random() < 0.6:
                # Try to add a random unselected triple
                unselected = [i for i in range(num_triples) if not selected[i]]
                if not unselected:
                    break
                t_add = random.choice(unselected)
                if try_add(selected, config_counts, t_add):
                    current_size += 1
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Try a swap: remove one selected, add one unselected
                selected_list = [i for i in range(num_triples) if selected[i]]
                unselected = [i for i in range(num_triples) if not selected[i]]
                if not selected_list or not unselected:
                    break
                
                t_rem = random.choice(selected_list)
                t_add = random.choice(unselected)
                
                # Remove t_rem
                remove_triple(selected, config_counts, t_rem)
                current_size -= 1
                
                # Try to add t_add
                if try_add(selected, config_counts, t_add):
                    current_size += 1
                    no_improve = 0
                else:
                    # Revert: add back t_rem
                    try_add(selected, config_counts, t_rem)
                    current_size += 1
                    no_improve += 1
            
            if current_size > best_size:
                best_size = current_size
                best_selected = selected[:]
        
        # Occasional fluctuation: remove a few random triples and re-greedy
        if restart % 3 == 0 and best_selected is not None:
            selected = best_selected[:]
            config_counts = [0] * num_configs
            for i in range(num_triples):
                if selected[i]:
                    for ci in triple_to_configs[i]:
                        config_counts[ci] += 1
            
            # Remove 3-7 random triples
            sel_list = [i for i in range(num_triples) if selected[i]]
            for _ in range(random.randint(3, 7)):
                if sel_list:
                    t = random.choice(sel_list)
                    remove_triple(selected, config_counts, t)
                    sel_list.remove(t)
            
            # Re-greedy fill
            for t_idx in range(num_triples):
                if not selected[t_idx]:
                    try_add(selected, config_counts, t_idx)
            
            current_size = sum(selected)
            if current_size > best_size:
                best_size = current_size
                best_selected = selected[:]
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Return best result
    if best_selected is None:
        # Fallback: return empty or a small valid set
        return []
    
    result = [list(all_triples[i]) for i in range(num_triples) if best_selected[i]]
    return result
