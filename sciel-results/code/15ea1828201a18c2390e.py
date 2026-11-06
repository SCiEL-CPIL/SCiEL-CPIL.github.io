"""Triple systems on 16 points without K(2,2,2) - greedy + simulated annealing."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _all_transversals(six_points):
    """For 6 points in 3 pairs, return the 8 transversal triples."""
    a, b, c, d, e, f = six_points
    return frozenset([
        frozenset((a, c, e)), frozenset((a, c, f)),
        frozenset((a, d, e)), frozenset((a, d, f)),
        frozenset((b, c, e)), frozenset((b, c, f)),
        frozenset((b, d, e)), frozenset((b, d, f)),
    ])

def _precompute():
    """Precompute all K(2,2,2) configurations and their transversals."""
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    # For each 6-point set, there are 15 ways to partition into 3 pairs
    # Each partition gives 8 transversals
    # Total: C(16,6) * 15 = 8008 * 15 = 120120 configurations
    
    # For each triple, store which configurations it belongs to
    # triple_to_configs: dict from tuple(triple) to list of config_ids
    
    configs = []  # list of (six_points_set, transversals_set)
    triple_to_configs = {t: [] for t in all_triples}
    
    for six in combinations(range(n), 6):
        six_set = frozenset(six)
        points = list(six)
        
        def gen_matchings(pts):
            if len(pts) == 0:
                yield []
                return
            first = pts[0]
            for i in range(1, len(pts)):
                second = pts[i]
                rest = pts[1:i] + pts[i+1:]
                for rm in gen_matchings(rest):
                    yield [(first, second)] + rm
            return
        
        for matching in gen_matchings(points):
            a, b = matching[0]
            c, d = matching[1]
            e, f = matching[2]
            transversals = [
                tuple(sorted((a, c, e))), tuple(sorted((a, c, f))),
                tuple(sorted((a, d, e))), tuple(sorted((a, d, f))),
                tuple(sorted((b, c, e))), tuple(sorted((b, c, f))),
                tuple(sorted((b, d, e))), tuple(sorted((b, d, f))),
            ]
            config_id = len(configs)
            configs.append((six_set, frozenset(transversals)))
            for t in transversals:
                triple_to_configs[t].append(config_id)
    
    return all_triples, triple_to_configs, configs

def construct():
    deadline = time.time() + 115  # safety margin
    
    all_triples, triple_to_configs, configs = _precompute()
    n_triples = len(all_triples)
    
    # Start with all triples through point 0
    initial = set(t for t in all_triples if 0 in t)
    current = set(initial)
    best = set(initial)
    best_size = len(current)
    
    # Config counts: how many transversals of each config are in current
    config_counts = [0] * len(configs)
    for t in current:
        for cid in triple_to_configs[t]:
            config_counts[cid] += 1
    
    def can_add(t):
        """Check if adding triple t would complete any K(2,2,2)."""
        for cid in triple_to_configs[t]:
            if config_counts[cid] >= 7:
                return False
        return True
    
    def add_triple(t):
        for cid in triple_to_configs[t]:
            config_counts[cid] += 1
        current.add(t)
    
    def remove_triple(t):
        for cid in triple_to_configs[t]:
            config_counts[cid] -= 1
        current.discard(t)
    
    # Greedy phase: add triples that don't conflict
    remaining = [t for t in all_triples if t not in current]
    # Sort by number of conflicts (fewer is better)
    remaining.sort(key=lambda t: len(triple_to_configs[t]))
    
    for t in remaining:
        if time.time() > deadline - 30:
            break
        if can_add(t):
            add_triple(t)
            if len(current) > best_size:
                best_size = len(current)
                best = set(current)
    
    # Simulated annealing phase
    if time.time() > deadline - 20:
        return [list(t) for t in best]
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while T > T_min and time.time() < deadline - 10:
        move_type = random.random()
        
        if move_type < 0.5:
            # Add a random triple not in current
            candidates = [t for t in all_triples if t not in current]
            if not candidates:
                T *= cooling
                continue
            t = random.choice(candidates)
            if can_add(t):
                add_triple(t)
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
        elif move_type < 0.8:
            # Remove a random triple from current
            if not current:
                T *= cooling
                continue
            t = random.choice(list(current))
            remove_triple(t)
        else:
            # Swap: remove one, add another
            if not current:
                T *= cooling
                continue
            t_remove = random.choice(list(current))
            candidates = [t for t in all_triples if t not in current]
            if not candidates:
                T *= cooling
                continue
            t_add = random.choice(candidates)
            
            remove_triple(t_remove)
            if can_add(t_add):
                add_triple(t_add)
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
            else:
                add_triple(t_remove)
        
        T *= cooling
    
    return [list(t) for t in best]

# EVOLVE-BLOCK-END
