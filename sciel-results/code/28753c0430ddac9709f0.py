"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import copy

def _count_sum_representations(chosen):
    """Count how many times each sum appears as a+b with a<=b from chosen."""
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            b = chosen[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    return counts

def _max_representations(chosen):
    """Return the maximum number of representations of any sum."""
    if not chosen:
        return 0
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            b = chosen[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    return max(counts.values()) if counts else 0

def _is_valid(chosen):
    """Check if chosen is a valid B2[2] set (all sums have at most 2 reps)."""
    if not chosen:
        return True
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            b = chosen[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
            if counts[s] > 2:
                return False
    return True

def _greedy_with_order(order, chosen_set, counts):
    """Greedy add from a given order, starting with some initial set."""
    chosen = list(chosen_set)
    counts = dict(counts)
    for x in order:
        if x in counts_set(chosen):
            continue
        new_sums = [(x + y) for y in chosen] + [2 * x]
        trial = dict(counts)
        valid = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            counts = trial
    return chosen, counts

def counts_set(chosen):
    return set(chosen)

def _local_search(chosen, max_time=10):
    """Try to improve via swaps and additions."""
    chosen = list(chosen)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        counts = _count_sum_representations(chosen)
        # Try adding elements
        for x in range(1, 1001):
            if x in set(chosen):
                continue
            new_sums = [(x + y) for y in chosen] + [2 * x]
            trial = dict(counts)
            valid = True
            for s in new_sums:
                trial[s] = trial.get(s, 0) + 1
                if trial[s] > 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts = trial
                improved = True
                break
        # Try swaps
        for i, a in enumerate(chosen):
            for x in range(1, 1001):
                if x in set(chosen):
                    continue
                # Try removing a and adding x
                new_chosen = chosen[:i] + chosen[i+1:] + [x]
                if len(set(new_chosen)) != len(new_chosen):
                    continue
                if _is_valid(new_chosen):
                    chosen = new_chosen
                    improved = True
                    break
            if improved:
                break
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe
    
    best_size = 0
    best_solution = []
    
    # Multiple randomized greedy passes
    random.seed(42)
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Randomize the order with some bias
        candidates = list(range(1, 1001))
        
        # Different strategies for ordering
        strategy = iteration % 4
        if strategy == 0:
            random.shuffle(candidates)
        elif strategy == 1:
            # Bias towards smaller numbers first
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random() * x)
        elif strategy == 2:
            # Bias towards larger numbers first
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random() * (1001 - x))
        else:
            # Mix of small and large
            small = list(range(1, 501))
            large = list(range(501, 1001))
            random.shuffle(small)
            random.shuffle(large)
            candidates = small + large
            random.shuffle(candidates)
        
        # Greedy pass
        chosen = []
        counts = {}
        for x in candidates:
            if x in set(chosen):
                continue
            new_sums = [(x + y) for y in chosen] + [2 * x]
            trial = dict(counts)
            valid = True
            for s in new_sums:
                trial[s] = trial.get(s, 0) + 1
                if trial[s] > 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts = trial
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_solution = list(chosen)
        
        # Local search to improve
        if time.time() < deadline - 5:
            improved = _local_search(chosen, max_time=min(10, deadline - time.time() - 2))
            if len(improved) > best_size:
                best_size = len(improved)
                best_solution = improved
    
    # Final verification
    if best_solution and _is_valid(best_solution):
        return best_solution
    
    # Fallback: simple greedy
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = [(x + y) for y in chosen] + [2 * x]
        trial = dict(counts)
        valid = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            counts = trial
    
    return chosen

# EVOLVE-BLOCK-END
