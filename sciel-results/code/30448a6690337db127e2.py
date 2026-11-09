import time
import random
import itertools
import sys

# Precomputation
N_POINTS = 16
ALL_TRIPLES = list(itertools.combinations(range(N_POINTS), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES)

# Generate all K(2,2,2) configurations
CONFIG_TRIPLES = []
TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]

def precompute():
    global CONFIG_TRIPLES, TRIPLE_CONFIGS
    CONFIG_TRIPLES = []
    TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]
    
    six_subsets = list(itertools.combinations(range(N_POINTS), 6))
    
    for pts in six_subsets:
        p0 = pts[0]
        rest = pts[1:]
        for i in range(5):
            p1 = rest[i]
            remaining = [x for x in rest if x != p1]
            r0, r1, r2, r3 = remaining
            pairings = [
                [(p0, p1), (r0, r1), (r2, r3)],
                [(p0, p1), (r0, r2), (r1, r3)],
                [(p0, p1), (r0, r3), (r1, r2)]
            ]
            for pairs in pairings:
                a, a_p = pairs[0]
                b, b_p = pairs[1]
                c, c_p = pairs[2]
                
                transversals = [
                    (a, b, c),
                    (a, b, c_p),
                    (a, b_p, c),
                    (a, b_p, c_p),
                    (a_p, b, c),
                    (a_p, b, c_p),
                    (a_p, b_p, c),
                    (a_p, b_p, c_p)
                ]
                
                idxs = []
                for t in transversals:
                    idx = TRIPLE_INDEX.get(tuple(sorted(t)))
                    if idx is not None:
                        idxs.append(idx)
                
                config_id = len(CONFIG_TRIPLES)
                CONFIG_TRIPLES.append(idxs)
                for idx in idxs:
                    TRIPLE_CONFIGS[idx].append(config_id)

precompute()

def is_valid_addition(triple_idx, in_set, config_counts):
    if triple_idx in in_set:
        return False
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        if config_counts[cfg_id] == 7:
            return False
    return True

def update_counts(triple_idx, delta, config_counts):
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        config_counts[cfg_id] += delta

def greedy_construct(num_restarts=50, time_limit=None):
    best_set = set()
    best_size = 0
    start_time = time.time()
    
    for r in range(num_restarts):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        current_set = set()
        config_counts = [0] * len(CONFIG_TRIPLES)
        
        for t_idx in order:
            if is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                update_counts(t_idx, 1, config_counts)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    return best_set, best_size

def local_search(initial_set, time_limit):
    current_set = initial_set.copy()
    config_counts = [0] * len(CONFIG_TRIPLES)
    
    for t_idx in current_set:
        update_counts(t_idx, 1, config_counts)
        
    best_set = current_set.copy()
    best_size = len(current_set)
    
    start_time = time.time()
    stuck_count = 0
    
    while (time.time() - start_time) < time_limit:
        improved = False
        
        # Try to add any valid triple
        candidates = [t for t in range(N_TRIPLES) if t not in current_set]
        random.shuffle(candidates)
        
        for t_idx in candidates:
            if is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                update_counts(t_idx, 1, config_counts)
                improved = True
        
        if improved:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            stuck_count = 0
            continue
        
        stuck_count += 1
        
        # Try 1-for-2 exchanges
        found_exchange = False
        for _ in range(100):
            if (time.time() - start_time) > time_limit:
                break
                
            if not current_set:
                break
            t_out = random.choice(list(current_set))
            
            current_set.remove(t_out)
            update_counts(t_out, -1, config_counts)
            
            added = []
            candidates = [t for t in range(N_TRIPLES) if t not in current_set and t != t_out]
            random.shuffle(candidates)
            
            for t1 in candidates:
                if is_valid_addition(t1, current_set, config_counts):
                    current_set.add(t1)
                    update_counts(t1, 1, config_counts)
                    added.append(t1)
                    break
            
            if len(added) == 1:
                candidates2 = [t for t in range(N_TRIPLES) if t not in current_set and t != t_out and t != added[0]]
                random.shuffle(candidates2)
                for t2 in candidates2:
                    if is_valid_addition(t2, current_set, config_counts):
                        current_set.add(t2)
                        update_counts(t2, 1, config_counts)
                        added.append(t2)
                        break
            
            if len(added) >= 2:
                found_exchange = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                break
            elif len(added) == 1:
                # Accept neutral move with probability to escape local optima
                if random.random() < 0.3:
                    found_exchange = True
                    break
                else:
                    current_set.remove(added[0])
                    update_counts(added[0], -1, config_counts)
                    current_set.add(t_out)
                    update_counts(t_out, 1, config_counts)
            else:
                current_set.add(t_out)
                update_counts(t_out, 1, config_counts)
        
        if found_exchange:
            stuck_count = 0
            continue
        
        # Larger perturbation
        if len(current_set) > 10:
            k = random.randint(5, 15)
            to_remove = random.sample(list(current_set), min(k, len(current_set)))
            for t in to_remove:
                current_set.remove(t)
                update_counts(t, -1, config_counts)
            
            candidates = [t for t in range(N_TRIPLES) if t not in current_set]
            random.shuffle(candidates)
            for t in candidates:
                if is_valid_addition(t, current_set, config_counts):
                    current_set.add(t)
                    update_counts(t, 1, config_counts)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
        
        # If still stuck after many iterations, do a fresh restart from best
        if stuck_count > 50:
            current_set = best_set.copy()
            config_counts = [0] * len(CONFIG_TRIPLES)
            for t_idx in current_set:
                update_counts(t_idx, 1, config_counts)
            stuck_count = 0

    return best_set, best_size

def construct():
    start_time = time.time()
    total_time_limit = 110.0
    
    greedy_time = min(25.0, total_time_limit * 0.25)
    best_set, best_size = greedy_construct(num_restarts=100, time_limit=greedy_time)
    
    current_time = time.time() - start_time
    if current_time > total_time_limit:
        return [list(ALL_TRIPLES[i]) for i in best_set]
    
    remaining_time = total_time_limit - current_time
    best_set, best_size = local_search(best_set, remaining_time)
    
    result = [list(ALL_TRIPLES[i]) for i in best_set]
    
    return result
