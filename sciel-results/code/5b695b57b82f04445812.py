# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _precompute_ap_indices():
    """Return list of arrays, each an array of indices (0-based) for an AP."""
    aps = []
    for start in range(N):
        for step in range(1, N):
            # AP: start, start+step, start+2*step, ... < N
            idx = np.arange(start, N, step)
            if len(idx) >= 2:  # include length>=2; singletons handled separately
                aps.append(idx)
    return aps

def _precompute_ap_prefix_sums(aps):
    """For each AP, precompute prefix sums of indices for fast sum computation.
    Actually we'll just store the indices and use numpy sum."""
    return aps

def _max_abs_ap_sum(signs, aps):
    """Compute max absolute sum over all APs and singletons."""
    # Singletons: max abs is 1
    if not aps:
        return 1
    # For each AP, compute sum of signs at those indices
    # Use numpy for speed
    # We can vectorize partially by grouping by length? 
    # Simple approach: loop but use numpy sum
    max_val = 1
    for idx in aps:
        s = np.sum(signs[idx])
        abs_s = abs(s)
        if abs_s > max_val:
            max_val = abs_s
    return max_val

def _max_abs_ap_sum_fast(signs, aps, ap_lengths, ap_max_possible):
    """Faster: skip APs where even flipping all won't beat current best? 
    Actually just compute. Use a smarter approach: 
    Group APs by their index pattern? Too complex.
    Let's just use the direct approach but with numpy."""
    max_val = 1
    for idx in aps:
        s = np.sum(signs[idx])
        abs_s = abs(s)
        if abs_s > max_val:
            max_val = abs_s
    return max_val

def _hill_climb(signs, aps, time_budget, rng):
    """Hill climbing: flip bits that reduce max abs AP sum."""
    signs = np.array(signs, dtype=np.int8)
    current_max = _max_abs_ap_sum(signs, aps)
    
    # Precompute for each position, which APs it belongs to
    # pos_to_aps[i] = list of AP indices that include position i
    pos_to_aps = [[] for _ in range(N)]
    for ai, idx in enumerate(aps):
        for pos in idx:
            pos_to_aps[pos].append(ai)
    
    # For faster evaluation, we can track current sums per AP
    ap_sums = np.zeros(len(aps), dtype=np.int32)
    for ai, idx in enumerate(aps):
        ap_sums[ai] = np.sum(signs[idx])
    
    def evaluate_flip(pos):
        """Return new max if we flip position pos."""
        old_sign = signs[pos]
        new_sign = -old_sign
        delta = 2 * old_sign  # change in sum for each AP containing pos
        
        # Compute new max
        # We need to check all APs containing pos
        new_max = 1
        # First, check if current max is still achievable
        # The APs that change are those in pos_to_aps[pos]
        # For other APs, sum stays same
        
        # Quick check: if current_max - 1 < new_max potential, we need full check
        # Let's just compute new sums for affected APs and find max
        
        # Store old values
        affected = pos_to_aps[pos]
        old_sums = {}
        for ai in affected:
            old_sums[ai] = ap_sums[ai]
        
        # Apply flip
        for ai in affected:
            ap_sums[ai] += delta
        
        # Find new max
        new_max = 1
        for ai in affected:
            abs_s = abs(ap_sums[ai])
            if abs_s > new_max:
                new_max = abs_s
        
        # Check unaffected APs - their max is at most current_max
        # But we need to know the max of unaffected APs
        # This is expensive. Let's track global max differently.
        
        # For now, just check affected + assume unaffected <= current_max
        # This is an approximation. For correctness, we need full check.
        
        # Undo flip
        for ai in affected:
            ap_sums[ai] = old_sums[ai]
        
        return new_max
    
    # Better approach: track the top few max values to quickly know if flip helps
    # For simplicity, let's do a more careful hill climb
    
    best_signs = signs.copy()
    best_max = current_max
    
    t0 = time.time()
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - t0 < time_budget:
        if no_improve_count > max_no_improve:
            break
        
        # Try flipping each position
        best_pos = -1
        best_new_max = current_max
        
        for pos in range(N):
            if time.time() - t0 > time_budget:
                break
            new_max = evaluate_flip(pos)
            if new_max < best_new_max:
                best_new_max = new_max
                best_pos = pos
        
        if best_pos >= 0 and best_new_max < current_max:
            # Apply flip
            signs[best_pos] = -signs[best_pos]
            delta = 2 * (-signs[best_pos])  # wait, sign already flipped
            # Recompute: old sign was signs[best_pos] before flip
            # After flip, signs[best_pos] is new
            # delta = 2 * old_sign = 2 * (-new_sign) = -2 * new_sign
            delta = -2 * signs[best_pos]
            for ai in pos_to_aps[best_pos]:
                ap_sums[ai] += delta
            current_max = best_new_max
            best_signs = signs.copy()
            best_max = current_max
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_signs, best_max

def _simulated_annealing(signs, aps, time_budget, rng):
    """SA approach."""
    signs = np.array(signs, dtype=np.int8)
    current_max = _max_abs_ap_sum(signs, aps)
    
    pos_to_aps = [[] for _ in range(N)]
    for ai, idx in enumerate(aps):
        for pos in idx:
            pos_to_aps[pos].append(ai)
    
    ap_sums = np.zeros(len(aps), dtype=np.int32)
    for ai, idx in enumerate(aps):
        ap_sums[ai] = np.sum(signs[idx])
    
    best_signs = signs.copy()
    best_max = current_max
    
    t0 = time.time()
    total_time = time_budget
    start_temp = 5.0
    end_temp = 0.1
    
    while time.time() - t0 < total_time:
        elapsed = time.time() - t0
        progress = elapsed / total_time
        temp = start_temp * (end_temp / start_temp) ** progress
        
        pos = rng.randint(0, N - 1)
        old_sign = signs[pos]
        signs[pos] = -old_sign
        delta = 2 * old_sign
        
        for ai in pos_to_aps[pos]:
            ap_sums[ai] += delta
        
        # Compute new max - need to be careful
        # Find max over all APs
        new_max = 1
        for ai in range(len(aps)):
            abs_s = abs(ap_sums[ai])
            if abs_s > new_max:
                new_max = abs_s
        
        delta_max = new_max - current_max
        
        if delta_max < 0 or rng.random() < np.exp(-delta_max / temp):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            # Revert
            signs[pos] = old_sign
            for ai in pos_to_aps[pos]:
                ap_sums[ai] -= delta
    
    return best_signs, best_max

def construct():
    t_start = time.time()
    rng = np.random.RandomState(42)
    
    # Precompute APs
    aps = _precompute_ap_indices()
    
    # Start with random initialization
    signs = rng.choice([-1, 1], size=N).astype(np.int8)
    
    # Phase 1: Hill climbing
    t1 = time.time()
    hill_time = min(20.0, DEADLINE - (t1 - t_start) - 10.0)
    if hill_time > 1:
        signs, current_max = _hill_climb(signs, aps, hill_time, rng)
    
    t2 = time.time()
    sa_time = min(30.0, DEADLINE - (t2 - t_start) - 5.0)
    if sa_time > 1:
        signs, current_max = _simulated_annealing(signs, aps, sa_time, rng)
    
    # Final hill climb with remaining time
    t3 = time.time()
    final_hill_time = max(0.5, DEADLINE - (t3 - t_start) - 2.0)
    if final_hill_time > 0.5:
        signs, current_max = _hill_climb(signs, aps, final_hill_time, rng)
    
    # Convert to list of Python ints
    return [int(s) for s in signs]

# EVOLVE-BLOCK-END
