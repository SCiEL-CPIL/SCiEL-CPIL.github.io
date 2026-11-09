import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 16
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute the conflict structures
    # A forbidden configuration is defined by 3 disjoint pairs.
    # We need to track for each configuration how many of its 8 transversals are present.
    
    # Generate all unique configurations (sets of 3 disjoint pairs)
    # There are C(16,6) * 15 = 8008 * 15 = 120120 such configurations.
    
    configs = []
    config_to_idx = {}
    
    # Helper to generate all 15 perfect matchings on 6 vertices
    def get_matchings(v6):
        matchings = []
        # v0 pairs with v1..v5
        for i in range(1, 6):
            rem = [v for j, v in enumerate(v6) if j != 0 and j != i]
            # rem has 4 vertices. Matchings on 4 vertices: 3
            for j in range(1, 4):
                rem2 = [v for k, v in enumerate(rem) if k != 0 and k != j]
                p1 = tuple(sorted((v6[0], v6[i])))
                p2 = tuple(sorted((rem[0], rem[j])))
                p3 = tuple(sorted((rem2[0], rem2[1])))
                pairs = sorted([p1, p2, p3])
                matchings.append(tuple(pairs))
        return matchings

    for v6 in itertools.combinations(range(n), 6):
        for m in get_matchings(v6):
            if m not in config_to_idx:
                config_to_idx[m] = len(configs)
                configs.append(m)
    
    num_configs = len(configs)
    
    # Map each triple to the list of config indices it belongs to
    # A triple belongs to a config if it is one of the 8 transversals of that config.
    triple_to_configs = [[] for _ in range(num_triples)]
    
    for cfg_idx, config in enumerate(configs):
        p1, p2, p3 = config
        a, b = p1
        c, d = p2
        e, f = p3
        
        transversals = [
            (a, c, e), (a, c, f), (a, d, e), (a, d, f),
            (b, c, e), (b, c, f), (b, d, e), (b, d, f)
        ]
        
        for t in transversals:
            t_sorted = tuple(sorted(t))
            if t_sorted in triple_to_idx:
                triple_to_configs[triple_to_idx[t_sorted]].append(cfg_idx)
    
    # State variables
    selected = [False] * num_triples
    config_counts = [0] * num_configs
    
    def is_valid_add(t_idx):
        """Check if adding triple t_idx would violate any constraint."""
        for cfg_idx in triple_to_configs[t_idx]:
            if config_counts[cfg_idx] == 7:
                return False
        return True

    def add_triple(t_idx):
        """Add triple t_idx to the system, updating state."""
        selected[t_idx] = True
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] += 1

    def remove_triple(t_idx):
        """Remove triple t_idx from the system, updating state."""
        selected[t_idx] = False
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] -= 1
            
    def try_add(t_idx):
        """Try to add a triple. Returns True if added."""
        if is_valid_add(t_idx):
            add_triple(t_idx)
            return True
        return False

    best_system = None
    best_count = 0
    random.seed(42)
    
    # Strategy: 
    # 1. Multiple randomized greedy starts.
    # 2. Local search with 1-swaps (remove one, try to add another) and 2-swaps.
    # 3. The key is that removing a triple might unblock many additions.
    
    max_runs = 100
    steps_per_run = 2000
    
    for run in range(max_runs):
        if time.time() > deadline - 5:
            break
            
        # Reset state
        for i in range(num_triples):
            if selected[i]:
                remove_triple(i)
        # Ensure counts are zero (safety)
        for i in range(num_configs):
            config_counts[i] = 0
            
        # 1. Randomized Greedy Initialization
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            try_add(t_idx)
            
        current_count = sum(selected)
        
        # 2. Local Search Phase
        # We maintain a list of "blocked" triples? No, just iterate.
        # To make it efficient, we can focus on swaps.
        
        # 1-Swap: Pick a selected triple to remove, then try to add a random unselected one.
        # If the add succeeds, net gain 0 (swap). But maybe subsequent adds become possible?
        # Actually, a swap is only useful if it allows *more* additions later.
        # A better local search move: 
        # - Try to add a random unselected triple. If it fails, it's blocked by some configs at count 7.
        # - Pick one of those blocking configs.
        # - Pick a random triple from that config that is currently selected.
        # - Remove it.
        # - Try to add the original unselected triple again.
        # - If it succeeds, we've swapped. 
        # - Then, do a "sweep" to see if we can add anything else.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Sweep 1: Try to add any unselected triple that is now valid
            # This is O(N) per sweep, but N=560. 
            # We can optimize by only checking triples that were recently blocked?
            # For now, full sweep is safe.
            for t_idx in range(num_triples):
                if not selected[t_idx]:
                    if try_add(t_idx):
                        current_count += 1
                        improved = True
                        # If we added, we might have blocked others, but we continue sweep.
                        # Actually, if we add, counts increase, so other adds might become invalid.
                        # But we are checking `try_add` which handles validity.
                        
            if not improved:
                # No simple additions possible. Try swaps.
                # Pick a random unselected triple that is blocked.
                unselected = [i for i in range(num_triples) if not selected[i]]
                if not unselected:
                    break
                    
                # Try a few random swaps
                swap_found = False
                for _ in range(50): # Limit swap attempts per sweep to avoid getting stuck
                    if time.time() > deadline - 5:
                        break
                    t_add = random.choice(unselected)
                    if is_valid_add(t_add):
                        # This shouldn't happen if sweep 1 was perfect, but maybe state changed?
                        # Or maybe sweep 1 missed it? No, sweep 1 checks all.
                        # So t_add must be blocked.
                        continue
                    
                    # Find a config that blocks t_add
                    blocking_configs = []
                    for cfg_idx in triple_to_configs[t_add]:
                        if config_counts[cfg_idx] == 7:
                            blocking_configs.append(cfg_idx)
                    
                    if not blocking_configs:
                        continue
                        
                    cfg_block = random.choice(blocking_configs)
                    
                    # Get the 8 transversals of this config
                    p1, p2, p3 = configs[cfg_block]
                    a, b = p1
                    c, d = p2
                    e, f = p3
                    transversals = [
                        (a, c, e), (a, c, f), (a, d, e), (a, d, f),
                        (b, c, e), (b, c, f), (b, d, e), (b, d, f)
                    ]
                    
                    # Find selected transversals in this config
                    selected_in_cfg = []
                    for t in transversals:
                        t_sorted = tuple(sorted(t))
                        t_idx_t = triple_to_idx[t_sorted]
                        if selected[t_idx_t]:
                            selected_in_cfg.append(t_idx_t)
                            
                    if not selected_in_cfg:
                        continue
                        
                    # Pick one to remove
                    t_remove = random.choice(selected_in_cfg)
                    
                    # Perform swap: remove t_remove, try add t_add
                    # We must be careful: removing t_remove might make t_add valid.
                    # But it might also make other things invalid? No, removing only decreases counts.
                    # So if t_add was blocked only by cfg_block (and others at 7), 
                    # removing t_remove (which is in cfg_block) reduces cfg_block count to 6.
                    # If t_add is in other configs at 7, it will still be blocked.
                    
                    # Let's check if removing t_remove makes t_add valid.
                    # We can simulate:
                    # Temporarily remove t_remove
                    remove_triple(t_remove)
                    
                    if is_valid_add(t_add):
                        # Accept swap
                        add_triple(t_add)
                        current_count += 0 # Net change 0 for the swap itself
                        # But maybe this unblocks others?
                        improved = True
                        swap_found = True
                        break
                    else:
                        # Revert
                        add_triple(t_remove)
                        # Don't break, try another swap? 
                        # If we don't break, we keep trying swaps in the same "while improved" loop iteration?
                        # No, the `for _ in range(50)` loop handles multiple attempts.
                        # If none work, we exit the for loop.
                        
                if swap_found:
                    # If we found a swap, restart the "while improved" loop to do another sweep
                    # because the state has changed.
                    continue
                else:
                    # No swaps found. We are stuck in a local optimum.
                    # Break out of the while loop for this run.
                    break
        
        # Update best
        if current_count > best_count:
            best_count = current_count
            best_system = [all_triples[i] for i in range(num_triples) if selected[i]]
            
        # Optional: If we have time, do a "random restart" from the current state?
        # Or just start a new run.
        
    if best_system is None:
        # Fallback: just return a small valid set
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_system
