import random
import time
import math
import numpy as np

def calc_max_disc_and_witnesses(signs, n=256, limit=None):
    """
    Calculates max discrepancy and returns the witnesses (progressions) achieving it.
    If limit is provided and exceeded, returns early with the current best witness.
    Uses vectorized prefix-sum approach for speed.
    """
    signs_arr = np.array(signs, dtype=np.int8)
    max_disc = 1
    best_witnesses = []
    for d in range(1, n):
        for r in range(d):
            # Get all indices in this AP
            idxs = np.arange(r, n, d)
            if len(idxs) < 2:
                continue
            # Compute prefix sums
            s = signs_arr[idxs]
            prefix = np.cumsum(s)
            # Include 0 as starting point
            prefix_full = np.concatenate([[0], prefix])
            min_p = int(np.min(prefix_full))
            max_p = int(np.max(prefix_full))
            disc = max_p - min_p
            if disc > max_disc:
                max_disc = disc
                if limit is not None and max_disc > limit:
                    return max_disc, [(d, r, min_p, max_p)]
                best_witnesses = [(d, r, min_p, max_p)]
            elif disc == max_disc:
                best_witnesses.append((d, r, min_p, max_p))
    return max_disc, best_witnesses

def get_indices_from_witnesses(witnesses, n=256):
    """Extract unique indices from a list of witnesses."""
    indices = set()
    for d, r, min_p, max_p in witnesses:
        idx = r
        while idx < n:
            indices.add(idx)
            idx += d
    return list(indices)

def calc_max_disc(signs, n=256, limit=None):
    """Quick max discrepancy calculation (returns just the value)."""
    signs_arr = np.array(signs, dtype=np.int8)
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            idxs = np.arange(r, n, d)
            if len(idxs) < 2:
                continue
            s = signs_arr[idxs]
            prefix = np.cumsum(s)
            prefix_full = np.concatenate([[0], prefix])
            disc = int(np.max(prefix_full) - np.min(prefix_full))
            if disc > max_disc:
                max_disc = disc
                if limit is not None and max_disc > limit:
                    return max_disc
    return max_disc

def greedy_repair(signs, n=256, deadline=None):
    """Greedy repair: repeatedly flip the sign that most reduces max discrepancy.
    Uses targeted approach: first try flipping indices in the worst AP."""
    current_disc, current_witnesses = calc_max_disc_and_witnesses(signs, n)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        best_gain = 0
        best_idx = -1
        # First, try flipping indices in the worst AP(s)
        hot_indices = get_indices_from_witnesses(current_witnesses, n)
        candidates = hot_indices if hot_indices else list(range(n))
        # Shuffle for diversity
        random.shuffle(candidates)
        for i in candidates:
            if deadline is not None and time.time() > deadline:
                break
            signs[i] *= -1
            new_disc = calc_max_disc(signs, n, limit=current_disc)
            signs[i] *= -1
            gain = current_disc - new_disc
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx >= 0 and best_gain > 0:
            signs[best_idx] *= -1
            current_disc -= best_gain
            improved = True
            current_witnesses, _ = calc_max_disc_and_witnesses(signs, n)
            # Re-fetch witnesses after improvement
            _, current_witnesses = calc_max_disc_and_witnesses(signs, n)
    return current_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    # Initial solution: use Thue-Morse for better starting point
    signs = []
    for i in range(n):
        # Thue-Morse sequence: (-1)^(popcount(i))
        pc = bin(i).count('1')
        signs.append(1 if pc % 2 == 0 else -1)
    
    # Initial full calculation
    current_disc, current_witnesses = calc_max_disc_and_witnesses(signs, n)
    best_disc = current_disc
    best_signs = signs[:]
    
    # Phase 1: Greedy repair to quickly reduce discrepancy
    if time.time() < deadline - 10:
        current_disc = greedy_repair(signs, n, deadline=deadline - 10)
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs[:]
    
    # Phase 2: Targeted Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline - 5:
        # 1. Select move based on current witnesses
        if not current_witnesses:
            # Fallback to random if no witnesses found (should not happen)
            i = random.randint(0, n - 1)
        else:
            hot_indices = get_indices_from_witnesses(current_witnesses, n)
            if not hot_indices:
                i = random.randint(0, n - 1)
            else:
                i = random.choice(hot_indices)
                
        # 2. Flip sign
        signs[i] *= -1
        
        # 3. Evaluate new state
        # Use limit to speed up rejection
        limit = current_disc + int(T * 5) + 1
        new_disc, new_witnesses = calc_max_disc_and_witnesses(signs, n, limit=limit)
        
        delta = new_disc - current_disc
        
        # 4. Accept or Reject
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_disc = new_disc
            current_witnesses = new_witnesses # Update witnesses for next move selection
            
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            # Reject: Revert sign
            signs[i] *= -1
            # Witnesses remain valid as state is unchanged
            
        # 5. Cool
        T *= cooling
        if T < T_min:
            T = T_min

    # Phase 3: Final polish with greedy repair
    if time.time() < deadline:
        signs[:] = best_signs[:]
        current_disc = greedy_repair(signs, n, deadline=deadline)
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs[:]

    return best_signs
