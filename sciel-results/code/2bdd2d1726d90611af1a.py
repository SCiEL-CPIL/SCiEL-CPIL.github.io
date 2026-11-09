"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

_DEADLINE = 320.0  # safety margin before 330s kill


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen):
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_iso(p, chosen[i], chosen[j]):
                return False
    return True


def _is_maximal(chosen, all_points):
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _greedy_random(all_points, rng, deadline):
    """Greedy construction with random point ordering."""
    points = list(all_points)
    rng.shuffle(points)
    chosen = []
    for p in points:
        if time.time() > deadline:
            break
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _local_search(chosen, all_points, rng, deadline):
    """Try to remove a point and re-add others to reduce size while staying maximal."""
    chosen_set = set(chosen)
    # Try removing each point and see if the set remains maximal (it won't be if we remove one,
    # but we can try to find a smaller maximal set by removing a point and then greedily filling)
    # Actually, removing a point makes it non-maximal. We need a different approach.
    # Try: remove a point, then greedily add back points that can be added, see if we end up smaller.
    best = list(chosen)
    
    for p in list(best):
        if time.time() > deadline:
            break
        # Remove p
        trial = [x for x in best if x != p]
        trial_set = set(trial)
        
        # Now greedily try to add points that can be added
        candidates = [q for q in all_points if q not in trial_set]
        rng.shuffle(candidates)
        for q in candidates:
            if time.time() > deadline:
                break
            if _can_add(q, trial):
                trial.append(q)
        
        if len(trial) < len(best):
            best = trial
    
    return best


def _simulated_annealing(chosen, all_points, rng, deadline):
    """SA-like search: remove a point, then greedily refill, accept if smaller or equal."""
    best = list(chosen)
    best_size = len(best)
    temperature = 1.0
    
    while time.time() < deadline:
        if len(best) <= 1:
            break
        
        # Remove a random point
        idx = rng.randint(0, len(best) - 1)
        p = best[idx]
        trial = best[:idx] + best[idx+1:]
        trial_set = set(trial)
        
        # Greedily refill
        candidates = [q for q in all_points if q not in trial_set]
        rng.shuffle(candidates)
        for q in candidates:
            if time.time() > deadline:
                break
            if _can_add(q, trial):
                trial.append(q)
        
        # Accept if smaller, or with some probability if equal (to escape local optima)
        if len(trial) < best_size:
            best = trial
            best_size = len(best)
            temperature = max(0.1, temperature * 0.95)
        elif len(trial) == best_size and rng.random() < 0.1:
            best = trial
            temperature = max(0.1, temperature * 0.98)
        else:
            temperature = max(0.1, temperature * 0.99)
    
    return best


def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5.0  # 5s safety margin
    
    rng = random.Random(42)
    
    best = None
    best_size = float('inf')
    
    # Phase 1: Multiple random greedy restarts
    restarts = 50
    for i in range(restarts):
        if time.time() > deadline:
            break
        chosen = _greedy_random(all_points, rng, deadline)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen
    
    if best is None:
        # Fallback: deterministic row-major
        chosen = []
        for p in all_points:
            if _can_add(p, chosen):
                chosen.append(p)
        return chosen
    
    # Phase 2: Local search on best
    if time.time() < deadline:
        best = _local_search(best, all_points, rng, deadline)
        best_size = len(best)
    
    # Phase 3: Simulated annealing on best
    if time.time() < deadline:
        best = _simulated_annealing(best, all_points, rng, deadline)
        best_size = len(best)
    
    # Phase 4: More random restarts with local search
    for i in range(30):
        if time.time() > deadline:
            break
        chosen = _greedy_random(all_points, rng, deadline)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen
            # Apply local search to new best
            if time.time() < deadline:
                best = _local_search(best, all_points, rng, deadline)
                best_size = len(best)
    
    return best
# EVOLVE-BLOCK-END
