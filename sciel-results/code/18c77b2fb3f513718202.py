import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = [] # List of lists of 8 triple indices

    points = range(N)
    for pts in itertools.combinations(points, 6):
        # Generate all partitions of these 6 points into 3 pairs
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            # Normalize pair order and pair internal order for unique ID
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                # Find the 8 transversal triples
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # For each triple, which configs does it belong to?
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            
    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        # Check if any config it belongs to already has 7 others
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def get_missing():
        return [i for i in range(num_triples) if not present[i]]

    def run_annealing(seed_idx):
        random.seed(seed_idx)
        present[:] = False
        config_counts[:] = 0
        
        # 1. Initial Random Greedy Fill
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                
        current_score = np.sum(present)
        best_score = current_score
        best_state = present.copy()
        
        # 2. Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999 # Adjusted for the number of steps within 110s
        
        # We don't know exact steps, so we just run until time limit or T < T_min
        # To ensure we finish, we check time in the loop
        
        while time.time() < deadline - 2:
            # 1. Climb: Try to add any missing triple (Greedy)
            # This is cheap and always improves
            # We can do a quick pass. If we add, reset no_improve logic? 
            # SA doesn't have "no_improve", it accepts based on T.
            # But adding is always delta > 0, so always accept.
            # We'll just do a random check of a few missing to see if any fit.
            
            # For efficiency, we'll mix:
            # 50% chance: Try a pure add (if any exist)
            # 50% chance: Try a 2-remove 1-add move
            
            if random.random() < 0.5:
                # Try to add
                # Pick a random missing triple
                # To be efficient, we can just pick a random index and check if missing
                # Or maintain a list of missing. Maintaining list is expensive to update.
                # Let's just pick random indices until we find a missing one that fits.
                added = False
                for _ in range(10): # Limit attempts per step
                    t_idx = random.randint(0, num_triples - 1)
                    if not present[t_idx]:
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_score += 1
                            added = True
                            break
                if added:
                    if current_score > best_score:
                        best_score = current_score
                        best_state = present.copy()
                    continue # Skip SA move, just climb
            
            # SA Move: Remove 2, Add 1
            # Pick 2 random present triples to remove
            # Find a missing triple that becomes valid
            # If it becomes valid, accept with probability exp(delta/T)
            
            # To make this efficient, we need to pick 2 removals that "help" a specific addition.
            # Or just pick 2 random removals and see if ANY missing triple becomes valid.
            # "Any missing triple" is 560 checks. Too slow.
            
            # Better: Pick a target missing triple t_add.
            # Identify blockers (configs with count 7).
            # Pick 2 present triples from these blockers to remove.
            # Check if t_add becomes valid.
            # If yes, delta = 1 - 2 = -1.
            # Accept with prob exp(-1/T).
            
            # Pick random missing triple
            t_add = random.randint(0, num_triples - 1)
            if present[t_add]:
                continue
            
            # Check if it's currently blocked
            blockers = []
            for cid in triple_to_configs[t_add]:
                if config_counts[cid] == 7:
                    blockers.append(cid)
            
            if not blockers:
                # It should have been added in the climb phase, or it's not actually missing?
                # If not present and not blocked, add it!
                add_triple(t_add)
                current_score += 1
                if current_score > best_score:
                    best_score = current_score
                    best_state = present.copy()
                continue
            
            # Pick 2 distinct present triples from the union of triples in blockers
            # We want to remove 2 that are in the blockers.
            # If we remove 2 from the same blocker, count drops to 5. t_add becomes valid (if no other blockers).
            # If we remove 1 from blocker A and 1 from blocker B, both drop to 6. t_add becomes valid.
            
            # Collect candidate removals
            candidates = set()
            for cid in blockers:
                for t in triples_in_config[cid]:
                    if present[t]:
                        candidates.add(t)
            
            if len(candidates) < 2:
                continue
                
            candidates_list = list(candidates)
            r1 = random.choice(candidates_list)
            r2 = random.choice(candidates_list)
            if r1 == r2:
                continue
            
            # Tentatively remove
            remove_triple(r1)
            remove_triple(r2)
            
            # Check if t_add is now valid
            if is_valid_addition(t_add):
                # Delta = +1 (add) - 2 (remove) = -1
                # Accept?
                if random.random() < np.exp(-1.0 / T):
                    add_triple(t_add)
                    current_score -= 1
                    if current_score > best_score:
                        best_score = current_score
                        best_state = present.copy()
                else:
                    # Reject: Restore r1, r2. Do not add t_add.
                    add_triple(r1)
                    add_triple(r2)
                    # current_score unchanged
            else:
                # t_add still blocked (e.g. other blockers not touched, or r1/r2 didn't reduce count enough)
                # Restore
                add_triple(r1)
                add_triple(r2)
            
            T *= cooling_rate
            if T < T_min:
                # Reheat? Or just stop?
                # Let's reheat to T=1.0 to explore more
                T = 1.0
                
        return best_score, best_state

    best_score = 0
    best_state = np.zeros(num_triples, dtype=bool)
    
    num_restarts = 0
    while time.time() < deadline - 5:
        seed = num_restarts
        score, state = run_annealing(seed)
        if score > best_score:
            best_score = score
            best_state = state
        num_restarts += 1
        
    # Convert indices back to triples
    result = [all_triples[i] for i in range(num_triples) if best_state[i]]
    
    return result
