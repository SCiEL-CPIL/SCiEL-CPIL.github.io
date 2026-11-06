"""Triple systems on 16 points without K(2,2,2) - greedy + simulated annealing."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _gen_matchings(pts):
    """Generate all perfect matchings of a list of even length."""
    if len(pts) == 0:
        yield []
        return
    first = pts[0]
    for i in range(1, len(pts)):
        second = pts[i]
        rest = pts[1:i] + pts[i+1:]
        for rm in _gen_matchings(rest):
            yield [(first, second)] + rm
        return

def _precompute():
    """Precompute all K(2,2,2) configurations and triple-to-config mapping."""
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(frozenset(t) for t in all_triples)
    
    # Enumerate all configurations
    # A configuration is: 6 points + a perfect matching into 3 pairs
    # Each config has 8 transversal triples
    configs_transversals = []  # list of frozensets of 8 triples
    triple_to_configs = {frozenset(t): [] for t in all_triples}
    
    for six in combinations(range(n), 6):
        points = list(six)
        for matching in _gen_matchings(points):
            a, b = matching[0]
            c, d = matching[1]
            e, f = matching[2]
            transversals = frozenset([
                frozenset((a, c, e)), frozenset((a, c, f)),
                frozenset((a, d, e)), frozenset((a, d, f)),
                frozenset((b, c, e)), frozenset((b, c, f)),
                frozenset((b, d, e)), frozenset((b, d, f)),
            ])
            config_id = len(configs_transversals)
            configs_transversals.append(transversals)
            for t in transversals:
                triple_to_configs[t].append(config_id)
    
    return all_triples, triple_set, configs_transversals, triple_to_configs

def construct():
    deadline = time.time() + 115
    
    all_triples, triple_set, configs_transversals, triple_to_configs = _precompute()
    n_configs = len(configs_transversals)
    
    # Start with all triples through point 0
    current = set(frozenset(t) for t in all_triples if 0 in t)
    best = set(current)
    best_size = len(current)
    
    # Config counts for current
    config_counts = [0] * n_configs
    for t in current:
        for cid in triple_to_configs[t]:
            config_counts[cid] += 1
    
    def can_add(t, counts):
        """Check if adding triple t would complete any K(2,2,2) given current counts."""
        for cid in triple_to_configs[t]:
            if counts[cid] >= 7:
                return False
        return True
    
    # Greedy phase: add triples in order of fewest conflicts
    remaining = [t for t in all_triples if t not in current]
    remaining.sort(key=lambda t: len(triple_to_configs[t]))
    
    for t in remaining:
        if time.time() > deadline - 40:
            break
        if can_add(t, config_counts):
            for cid in triple_to_configs[t]:
                config_counts[cid] += 1
            current.add(t)
            if len(current) > best_size:
                best_size = len(current)
                best = set(current)
    
    # Simulated annealing
    if time.time() > deadline - 25:
        return [list(t) for t in best]
    
    T = 1.0
    T_min = 0.005
    cooling = 0.9995
    
    while T > T_min and time.time() < deadline - 15:
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random triple
            candidates = [t for t in all_triples if t not in current]
            if not candidates:
                T *= cooling
                continue
            t = random.choice(candidates)
            if can_add(t, config_counts):
                for cid in triple_to_configs[t]:
                    config_counts[cid] += 1
                current.add(t)
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
        elif move_type < 0.7:
            # Remove a random triple
            if not current:
                T *= cooling
                continue
            t = random.choice(list(current))
            for cid in triple_to_configs[t]:
                config_counts[cid] -= 1
            current.discard(t)
        else:
            # Swap: remove one, try to add another
            if not current:
                T *= cooling
                continue
            t_remove = random.choice(list(current))
            candidates = [t for t in all_triples if t not in current]
            if not candidates:
                T *= cooling
                continue
            t_add = random.choice(candidates)
            
            # Temporarily remove
            for cid in triple_to_configs[t_remove]:
                config_counts[cid] -= 1
            current.discard(t_remove)
            
            if can_add(t_add, config_counts):
                for cid in triple_to_configs[t_add]:
                    config_counts[cid] += 1
                current.add(t_add)
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
            else:
                # Revert
                for cid in triple_to_configs[t_remove]:
                    config_counts[cid] += 1
                current.add(t_remove)
        
        T *= cooling
    
    return [list(t) for t in best]

# EVOLVE-BLOCK-END
