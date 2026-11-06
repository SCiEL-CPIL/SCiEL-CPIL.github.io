"""Triple systems on 16 points without K(2,2,2) - simulated annealing."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _all_transversals(six):
    """Return the 8 transversal triples of K(2,2,2) on 6 points."""
    a, b, c, d, e, f = six
    partitions = [
        ({a,b},{c,d},{e,f}),
        ({a,b},{c,e},{d,f}),
        ({a,b},{c,f},{d,e}),
        ({a,c},{b,d},{e,f}),
        ({a,c},{b,e},{d,f}),
        ({a,c},{b,f},{d,e}),
        ({a,d},{b,c},{e,f}),
        ({a,d},{b,e},{c,f}),
        ({a,d},{b,f},{c,e}),
        ({a,e},{b,c},{d,f}),
        ({a,e},{b,d},{c,f}),
        ({a,e},{b,f},{c,d}),
        ({a,f},{b,c},{d,e}),
        ({a,f},{b,d},{c,e}),
        ({a,f},{b,e},{c,d}),
    ]
    result = set()
    for p1, p2, p3 in partitions:
        for x in p1:
            for y in p2:
                for z in p3:
                    result.add((min(x,y,z), min(max(x,y),z), max(max(x,y),z)))
    return result

def _precompute_transversals(n=16):
    """Map from frozenset of 6 points -> list of transversal triples (as sorted tuples)."""
    trans_map = {}
    for six in combinations(range(n), 6):
        trans_map[frozenset(six)] = _all_transversals(six)
    return trans_map

def _would_complete(triple, present, trans_map, triple_to_sixes):
    """Check if adding `triple` would complete any K(2,2,2)."""
    for six in triple_to_sixes[triple]:
        # Check all transversals of this 6-set
        all_present = True
        for t in trans_map[six]:
            if t != triple and t not in present:
                all_present = False
                break
        if all_present:
            return True
    return False

def construct():
    n = 16
    deadline = time.time() + 115  # safety margin before 120s limit
    random.seed(42)
    
    all_triples = list(combinations(range(n), 3))
    triple_set_all = set(all_triples)
    
    # Precompute transversals for all 6-subsets
    trans_map = _precompute_transversals(n)
    
    # For each triple, which 6-sets contain it
    triple_to_sixes = {t: [] for t in all_triples}
    for six in combinations(range(n), 6):
        fsix = frozenset(six)
        for t in trans_map[fsix]:
            triple_to_sixes[t].append(fsix)
    
    # Greedy initialization: start empty, add triples greedily
    present = set()
    # Try adding triples in a random order
    random.shuffle(all_triples)
    for t in all_triples:
        if not _would_complete(t, present, trans_map, triple_to_sixes):
            present.add(t)
    
    best = set(present)
    best_size = len(present)
    
    # Simulated annealing
    T = 30.0
    T_min = 0.5
    cooling = 0.999
    
    while T > T_min and time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.5:
            # Add a random absent triple
            absent = [t for t in all_triples if t not in present]
            if not absent:
                continue
            t = random.choice(absent)
            if not _would_complete(t, present, trans_map, triple_to_sixes):
                present.add(t)
                if len(present) > best_size:
                    best = set(present)
                    best_size = len(present)
            else:
                # Accept with some probability to escape local optima
                if random.random() < 2.718281828 ** (-2.0 / T):
                    present.add(t)
        elif move_type < 0.8:
            # Remove a random present triple
            if not present:
                continue
            t = random.choice(list(present))
            present.discard(t)
        else:
            # Swap: remove one, add another
            if not present:
                continue
            t_remove = random.choice(list(present))
            absent = [t for t in all_triples if t not in present]
            if not absent:
                continue
            t_add = random.choice(absent)
            present.discard(t_remove)
            if not _would_complete(t_add, present, trans_map, triple_to_sixes):
                present.add(t_add)
                if len(present) > best_size:
                    best = set(present)
                    best_size = len(present)
            else:
                if random.random() < 2.718281828 ** (-2.0 / T):
                    present.add(t_add)
        
        T *= cooling
    
    # Final greedy improvement
    for t in all_triples:
        if t not in best:
            if not _would_complete(t, best, trans_map, triple_to_sixes):
                best.add(t)
    
    return [list(t) for t in best]

# EVOLVE-BLOCK-END
