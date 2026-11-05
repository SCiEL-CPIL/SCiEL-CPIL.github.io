"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 256
DEADLINE = 55.0  # seconds, safety margin before 60s kill

def _is_sidon(selected):
    """Check if a list of integers forms a valid Sidon set."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    for i in range(len(selected)):
        a = selected[i]
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(selected):
    """Compute all pairwise sums a+b with a<=b for a Sidon set."""
    sums = set()
    for i in range(len(selected)):
        a = selected[i]
        for j in range(i, len(selected)):
            b = selected[j]
            sums.add(a + b)
    return sums

def _is_maximal_fast(selected, sums):
    """Check if the set is maximal given precomputed sums."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if x can be added
        can_add = True
        for a in selected:
            if (a + x) in sums:
                can_add = False
                break
        if can_add and (2 * x) in sums:
            can_add = False
        if can_add:
            return False
    return True

def _greedy_order(order):
    """Greedy construction following a given order of candidates.
    Runs to completion to guarantee the resulting set is maximal."""
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added
        can_add = True
        for a in selected:
            if (a + x) in sums:
                can_add = False
                break
        if can_add and (2 * x) in sums:
            can_add = False
        if can_add:
            selected.append(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
    return selected, sums

def _local_search(selected, sums, time_limit):
    """Try to minimize the size of a maximal Sidon set via removal and swaps."""
    start = time.time()
    best = list(selected)
    best_sums = sums
    best_size = len(best)
    
    # Precompute all missing elements
    all_elements = set(range(1, N + 1))
    
    while time.time() < time_limit:
        improved = False
        
        # Phase 1: Try removing elements
        for i in range(len(best)):
            if time.time() > time_limit:
                break
            trial = best[:i] + best[i+1:]
            trial_set = set(trial)
            # Check maximality of trial
            trial_sums = _get_sums(trial)
            if _is_maximal_fast(trial, trial_sums):
                best = trial
                best_sums = trial_sums
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps (remove one, add one)
        missing = list(all_elements - set(best))
        random.shuffle(missing)
        
        for i in range(len(best)):
            if time.time() > time_limit:
                break
            removed = best[i]
            base = best[:i] + best[i+1:]
            
            for x in missing:
                if x in set(base):
                    continue
                trial = base + [x]
                # Check Sidon
                if not _is_sidon(trial):
                    continue
                # Check maximal
                trial_sums = _get_sums(trial)
                if _is_maximal_fast(trial, trial_sums):
                    best = trial
                    best_sums = trial_sums
                    best_size = len(best)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try more aggressive: random restarts within local search
            # or break out
            # Try a few random swaps even if they temporarily increase size (simulated annealing style)
            temp_improved = False
            for _ in range(50):
                if time.time() > time_limit:
                    break
                i = random.randrange(len(best))
                removed = best[i]
                base = best[:i] + best[i+1:]
                x = random.choice(missing)
                if x in set(base):
                    continue
                trial = base + [x]
                if not _is_sidon(trial):
                    continue
                trial_sums = _get_sums(trial)
                if _is_maximal_fast(trial, trial_sums):
                    # Accept even if same size (for diversity)
                    best = trial
                    best_sums = trial_sums
                    temp_improved = True
                    break
            
            if not temp_improved:
                # Stuck, try a different initial ordering
                break
    
    return best, best_sums

def construct():
    start_time = time.time()
    
    # Generate multiple initial maximal Sidon sets
    candidates = []
    
    # Ascending
    s, sums = _greedy_order(list(range(1, N + 1)))
    candidates.append((s, sums))
    
    # Descending
    if time.time() < DEADLINE:
        s, sums = _greedy_order(list(range(N, 0, -1)))
        candidates.append((s, sums))
    
    # Random orderings
    rng = random.Random(42)
    for seed in range(20):
        if time.time() > DEADLINE - 5.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s, sums = _greedy_order(order)
        candidates.append((s, sums))
    
    # Sort by size
    candidates.sort(key=lambda x: len(x[0]))
    
    # Take top candidates and do local search on each
    best = None
    best_sums = None
    best_size = float('inf')
    
    # Do local search on the smallest candidates
    for s, sums in candidates[:5]:
        if time.time() > DEADLINE - 3.0:
            break
        remaining = DEADLINE - 3.0 - time.time()
        if remaining < 1.0:
            break
        trial, trial_sums = _local_search(s, sums, time.time() + remaining)
        if len(trial) < best_size:
            best = trial
            best_sums = trial_sums
            best_size = len(trial)
    
    # If we haven't done local search on the best candidate yet
    if best is None:
        best, best_sums = candidates[0]
        best_size = len(best)
    
    # Additional local search on the best found
    if time.time() < DEADLINE - 2.0:
        remaining = DEADLINE - 2.0 - time.time()
        if remaining > 1.0:
            best, best_sums = _local_search(best, best_sums, time.time() + remaining)
            best_size = len(best)
    
    # Final verification
    if not _is_maximal_fast(best, _get_sums(best)):
        # Fallback
        best, best_sums = _greedy_order(list(range(1, N + 1)))
    
    return sorted(best)

# EVOLVE-BLOCK-END
