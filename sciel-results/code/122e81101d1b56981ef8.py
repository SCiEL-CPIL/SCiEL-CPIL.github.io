"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START

def _independence_number(triples):
    """Compute the independence number (largest set containing no triple)."""
    n = 16
    triple_set = set()
    for t in triples:
        triple_set.add(frozenset(t))
    
    # Use a recursive branch-and-bound with memoization on bitmasks
    # Points 0..15 -> bitmask of 16 bits
    best = [0]
    
    # Precompute: for each point, which triples contain it
    # We'll use a different approach: iterate over all subsets is 2^16 = 65536, check each
    # That's fast enough.
    
    # For each subset, check if it contains any triple
    # A subset S is independent if no triple is fully contained in S
    
    # Bitmask representation
    def contains_triple(mask):
        for t in triple_set:
            # Check if all three points of t are in mask
            a, b, c = tuple(sorted(t))
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                return True
        return False
    
    # Enumerate all subsets from largest to smallest, return first independent
    # Actually, we want the max size independent set.
    # Use the fact that there are only 65536 subsets.
    # For each size from 16 down to 0, check if any subset of that size is independent.
    
    # More efficient: for each subset, precompute whether it's independent
    # Then find max popcount among independent subsets.
    
    # Even better: iterate subsets, maintain independence
    # Let's just do a simple approach: for each subset mask from 0 to 2^16-1,
    # check if it's independent, track max popcount.
    
    # 65536 subsets * up to ~80 triples check = ~5M ops, very fast.
    
    max_size = 0
    for mask in range(1 << 16):
        # Quick check: popcount
        pc = bin(mask).count('1')
        if pc <= max_size:
            continue
        if not contains_triple(mask):
            max_size = pc
            if max_size == 16:
                break
    return max_size


def _valid_system(triples):
    """Check if the system is valid: any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _greedy_initial(n=16):
    """Generate a random valid triple system greedily."""
    random.seed(42)  # Deterministic for reproducibility
    all_triples = list(itertools.combinations(range(n), 3))
    random.shuffle(all_triples)
    
    selected = []
    used_pairs = set()  # Track pairs used by selected triples
    
    for t in all_triples:
        # Check if this triple shares at most one point with each selected triple
        ok = True
        for s in selected:
            common = set(t) & set(s)
            if len(common) > 1:
                ok = False
                break
        if ok:
            selected.append(list(t))
    
    return selected


def _local_search(triples, time_limit=10.0):
    """Improve the system via local search: try removing and adding triples."""
    start_time = time.time()
    current = [list(t) for t in triples]
    current_alpha = _independence_number(current)
    best = list(current)
    best_alpha = current_alpha
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    while time.time() - start_time < time_limit:
        # Try random moves: remove a triple, or swap
        if random.random() < 0.5:
            # Remove a random triple
            if len(current) > 1:
                idx = random.randint(0, len(current) - 1)
                removed = current.pop(idx)
                new_alpha = _independence_number(current)
                if new_alpha <= best_alpha:
                    if new_alpha < best_alpha:
                        best = list(current)
                        best_alpha = new_alpha
                else:
                    current.append(removed)
        else:
            # Try to add a random triple
            t = random.choice(all_triples)
            t_list = list(t)
            # Check validity
            ok = True
            for s in current:
                if len(set(t_list) & set(s)) > 1:
                    ok = False
                    break
            if ok:
                current.append(t_list)
                new_alpha = _independence_number(current)
                if new_alpha <= best_alpha:
                    if new_alpha < best_alpha:
                        best = list(current)
                        best_alpha = new_alpha
                else:
                    current.pop()
    
    return best, best_alpha


def _simulated_annealing(triples, time_limit=60.0):
    """Simulated annealing to minimize independence number."""
    start_time = time.time()
    current = [list(t) for t in triples]
    current_alpha = _independence_number(current)
    best = list(current)
    best_alpha = current_alpha
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    temperature = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if temperature < 0.01:
            temperature = 1.0  # Reheat
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4 and len(current) > 1:
            # Remove a random triple
            idx = random.randint(0, len(current) - 1)
            removed = current.pop(idx)
            new_alpha = _independence_number(current)
            delta = new_alpha - current_alpha
            if delta <= 0 or random.random() < (1 - 1e-9) ** (delta / temperature):
                current_alpha = new_alpha
                if new_alpha < best_alpha:
                    best = list(current)
                    best_alpha = new_alpha
            else:
                current.append(removed)
        
        elif move_type < 0.7:
            # Add a random valid triple
            t = random.choice(all_triples)
            t_list = list(t)
            ok = True
            for s in current:
                if len(set(t_list) & set(s)) > 1:
                    ok = False
                    break
            if ok:
                current.append(t_list)
                new_alpha = _independence_number(current)
                delta = new_alpha - current_alpha
                if delta <= 0 or random.random() < (1 - 1e-9) ** (delta / temperature):
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best = list(current)
                        best_alpha = new_alpha
                else:
                    current.pop()
        
        elif move_type < 0.9 and len(current) > 1:
            # Swap: remove one, add one
            idx = random.randint(0, len(current) - 1)
            removed = current.pop(idx)
            t = random.choice(all_triples)
            t_list = list(t)
            ok = True
            for s in current:
                if len(set(t_list) & set(s)) > 1:
                    ok = False
                    break
            if ok:
                current.append(t_list)
                new_alpha = _independence_number(current)
                delta = new_alpha - current_alpha
                if delta <= 0 or random.random() < (1 - 1e-9) ** (delta / temperature):
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best = list(current)
                        best_alpha = new_alpha
                else:
                    current.pop()
                    current.append(removed)
            else:
                current.append(removed)
        
        else:
            # No-op or restart
            pass
        
        temperature *= cooling
    
    return best, best_alpha


def construct():
    """Construct a linear triple system on 16 points with small independence number."""
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Generate multiple random initial solutions
    best_triples = None
    best_alpha = 16  # Worst case
    
    # Try several random seeds for diversity
    for seed in range(20):
        if time.time() > deadline:
            break
        random.seed(seed)
        initial = _greedy_initial()
        alpha = _independence_number(initial)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in initial]
        
        # Local search on this initial
        if time.time() > deadline - 5:
            break
        improved, improved_alpha = _local_search(initial, time_limit=min(5.0, max(0.1, deadline - time.time())))
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_triples = improved
    
    # Simulated annealing on the best found so far
    if time.time() < deadline - 10 and best_triples is not None:
        sa_result, sa_alpha = _simulated_annealing(best_triples, time_limit=min(60.0, deadline - time.time() - 5))
        if sa_alpha < best_alpha:
            best_alpha = sa_alpha
            best_triples = sa_result
    
    # Final local search polish
    if time.time() < deadline - 2 and best_triples is not None:
        polished, polished_alpha = _local_search(best_triples, time_limit=min(5.0, max(0.1, deadline - time.time() - 1)))
        if polished_alpha < best_alpha:
            best_alpha = polished_alpha
            best_triples = polished
    
    if best_triples is None:
        # Fallback: return the simple 5-disjoint-triples system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples

# EVOLVE-BLOCK-END
