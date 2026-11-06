import numpy as np
import time
import random

N = 18
NUM_INPUTS = 1 << N

def _precompute_inputs():
    mat = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
    for i in range(N):
        half = 1 << i
        period = 2 * half
        pattern = np.concatenate([np.zeros(half, dtype=np.uint8), np.ones(half, dtype=np.uint8)])
        mat[i] = np.tile(pattern, NUM_INPUTS // period)
    return mat

INPUTS = _precompute_inputs()

def _simulate(comparators):
    mat = INPUTS.copy()
    for (i, j) in comparators:
        mi = mat[i] & mat[j]
        mj = mat[i] | mat[j]
        mat[i] = mi
        mat[j] = mj
    return mat

def _is_valid(comparators):
    mat = _simulate(comparators)
    for i in range(N - 1):
        if np.any(mat[i] > mat[i + 1]):
            return False
    return True

def _count_violations(comparators):
    """Count total inversions across all inputs (0 means sorted)."""
    mat = _simulate(comparators)
    total = 0
    for i in range(N - 1):
        total += np.sum(mat[i] > mat[i + 1])
    return total

def _greedy_remove(net, deadline):
    """Remove comparators one by one if the network remains valid."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            trial = net[:idx] + net[idx+1:]
            if _is_valid(trial):
                net = trial
                changed = True
                break
    return net

def _greedy_add(net, deadline):
    """
    Greedy construction: start from empty, add comparators that reduce
    violation count the most. This is used as an alternative path.
    """
    current = list(net)
    violations = _count_violations(current)
    if violations == 0:
        return current
    
    # Candidate comparators: all pairs (i, j) with i < j
    candidates = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # We'll iterate: find the comparator that reduces violations the most
    # among those not yet in the network (or even if in, we can add more)
    # Actually, let's just try adding any comparator not yet present
    present = set(current)
    
    while violations > 0 and time.time() < deadline:
        best_gain = 0
        best_comp = None
        best_violations = violations
        
        for comp in candidates:
            if comp in present:
                continue
            trial = current + [comp]
            v = _count_violations(trial)
            gain = violations - v
            if gain > best_gain:
                best_gain = gain
                best_comp = comp
                best_violations = v
        
        if best_comp is None:
            break
        
        current.append(best_comp)
        present.add(best_comp)
        violations = best_violations
        
        # If no improvement, break
        if best_gain <= 0:
            break
    
    return current

def _swap_adjacent(net, idx, deadline):
    """Try swapping adjacent comparators at idx and idx+1."""
    if idx + 1 >= len(net):
        return None
    trial = net[:]
    trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
    if _is_valid(trial):
        return trial
    return None

def _random_reorder(net, rng):
    """Randomly shuffle the network and try to repair it."""
    shuffled = net[:]
    rng.shuffle(shuffled)
    if _is_valid(shuffled):
        return shuffled
    return None

def _local_search(net, deadline):
    """
    More aggressive local search: try removing, swapping, and reordering.
    """
    best = list(net)
    best_len = len(best)
    rng = random.Random(42)
    
    # Phase 1: Greedy removal (multiple passes)
    for _ in range(5):
        if time.time() > deadline:
            break
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(best) - 1, -1, -1):
                if time.time() > deadline:
                    break
                trial = best[:idx] + best[idx+1:]
                if _is_valid(trial):
                    best = trial
                    best_len = len(best)
                    changed = True
                    break
    
    # Phase 2: Try swapping adjacent comparators to enable further removals
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(len(best) - 1):
            if time.time() > deadline:
                break
            trial = _swap_adjacent(best, idx, deadline)
            if trial is not None:
                # Check if this swap enables further removals
                new_best = _greedy_remove(trial, deadline)
                if len(new_best) < best_len:
                    best = new_best
                    best_len = len(best)
                    improved = True
                    break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Initial: odd-even transposition network
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    if not _is_valid(net):
        return net
    
    # Phase 1: Greedy removal
    net = _greedy_remove(net, deadline)
    
    # Phase 2: Local search with swaps
    net = _local_search(net, deadline)
    
    # Phase 3: Try multiple random restarts with different initial orderings
    rng = random.Random(123)
    best_net = net
    best_len = len(net)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Create a perturbed version of the current best
        perturbed = best_net[:]
        # Randomly swap a few adjacent comparators
        for _ in range(min(10, len(perturbed))):
            idx = rng.randint(0, len(perturbed) - 2)
            perturbed[idx], perturbed[idx+1] = perturbed[idx+1], perturbed[idx]
        
        if _is_valid(perturbed):
            # Try to improve it
            improved = _local_search(perturbed, deadline)
            if len(improved) < best_len:
                best_net = improved
                best_len = len(best_net)
    
    return best_net
