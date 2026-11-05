"""Optimized two-colouring of [1,256] to minimize monochromatic 4-APs.

Strategy:
1. Precompute all 4-APs and, for each position, the set of APs it participates in.
2. Maintain a running count of monochromatic APs.
3. For each position i, compute the delta (change in mono count) if we flip bit i,
   by examining only the APs that contain i (not all APs).
4. Use a combination of:
   - Greedy local search (flip the best available move)
   - Simulated annealing (accept worse moves with probability exp(-delta/T))
   - Multiple random restarts
5. Use numpy for vectorized operations where possible.
"""
import time
import random
import numpy as np

N = 256

def _precompute_aps():
    """Return list of 4-tuples (a, a+d, a+2d, a+3d) for all valid 4-APs."""
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _build_pos_aps(aps):
    """For each position, store the indices of APs it belongs to, and its role (0,1,2,3)."""
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[d].append((idx, 3))
    return pos_aps

def _count_mono(coloring, aps):
    """Count monochromatic 4-APs."""
    count = 0
    for (a, b, c, d) in aps:
        if coloring[a] == coloring[b] == coloring[c] == coloring[d]:
            count += 1
    return count

def _thue_morse():
    """Generate Thue-Morse sequence of length 256."""
    tm = [0] * N
    for i in range(1, N):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return tm

def _delta_for_flip(coloring, pos, aps, pos_aps, mono_flags):
    """
    Compute the change in mono-AP count if we flip bit at position pos.
    Only checks APs that contain pos.
    
    coloring: list/array of 0/1
    mono_flags: boolean array where mono_flags[i] = True if AP i is monochromatic
    Returns: delta (positive means flipping increases count, negative means decreases)
    """
    delta = 0
    old_val = coloring[pos]
    new_val = 1 - old_val
    
    for (ap_idx, role) in pos_aps[pos]:
        a, b, c, d = aps[ap_idx]
        was_mono = mono_flags[ap_idx]
        
        # After flip, is this AP monochromatic?
        # The flipped position now has new_val
        # Check if all four positions have the same value
        vals = [coloring[a], coloring[b], coloring[c], coloring[d]]
        vals[role] = new_val
        is_mono_new = (vals[0] == vals[1] == vals[2] == vals[3])
        
        if was_mono and not is_mono_new:
            delta -= 1
        elif not was_mono and is_mono_new:
            delta += 1
        # else: no change
    
    return delta

def _local_search_greedy(coloring, aps, pos_aps, max_iters=2000, deadline=None):
    """
    Greedy local search: repeatedly flip the bit with the best (most negative) delta.
    Returns (best_coloring, best_count).
    """
    current = list(coloring)
    current_count = _count_mono(current, aps)
    
    # Precompute mono_flags
    mono_flags = np.zeros(len(aps), dtype=bool)
    for idx, (a, b, c, d) in enumerate(aps):
        mono_flags[idx] = (current[a] == current[b] == current[c] == current[d])
    
    best_count = current_count
    best_coloring = list(current)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the best flip
        best_delta = 0
        best_pos = -1
        for pos in range(N):
            d = _delta_for_flip(current, pos, aps, pos_aps, mono_flags)
            if d < best_delta:
                best_delta = d
                best_pos = pos
        
        if best_pos == -1:
            break  # No improving move
        
        # Apply the flip
        pos = best_pos
        old_val = current[pos]
        current[pos] = 1 - old_val
        current_count += best_delta
        
        # Update mono_flags for all APs containing pos
        for (ap_idx, role) in pos_aps[pos]:
            a, b, c, d = aps[ap_idx]
            mono_flags[ap_idx] = (current[a] == current[b] == current[c] == current[d])
        
        if current_count < best_count:
            best_count = current_count
            best_coloring = list(current)
    
    return best_coloring, best_count

def _simulated_annealing(coloring, aps, pos_aps, T_start=5.0, T_end=0.1, max_iters=5000, deadline=None, seed=None):
    """
    Simulated annealing to escape local minima.
    Returns (best_coloring, best_count).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    current = list(coloring)
    current_count = _count_mono(current, aps)
    
    mono_flags = np.zeros(len(aps), dtype=bool)
    for idx, (a, b, c, d) in enumerate(aps):
        mono_flags[idx] = (current[a] == current[b] == current[c] == current[d])
    
    best_count = current_count
    best_coloring = list(current)
    
    n_steps = max_iters
    for it in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (it / max(n_steps - 1, 1))
        
        # Pick a random position to flip
        pos = rng.randint(0, N - 1)
        delta = _delta_for_flip(current, pos, aps, pos_aps, mono_flags)
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept the move
            old_val = current[pos]
            current[pos] = 1 - old_val
            current_count += delta
            
            # Update mono_flags
            for (ap_idx, role) in pos_aps[pos]:
                a, b, c, d = aps[ap_idx]
                mono_flags[ap_idx] = (current[a] == current[b] == current[c] == current[d])
            
            if current_count < best_count:
                best_count = current_count
                best_coloring = list(current)
    
    return best_coloring, best_count

def _k_flip_search(coloring, aps, pos_aps, k=2, max_iters=2000, deadline=None):
    """
    Try flipping k bits simultaneously to escape local minima.
    Returns (best_coloring, best_count).
    """
    current = list(coloring)
    current_count = _count_mono(current, aps)
    
    mono_flags = np.zeros(len(aps), dtype=bool)
    for idx, (a, b, c, d) in enumerate(aps):
        mono_flags[idx] = (current[a] == current[b] == current[c] == current[d])
    
    best_count = current_count
    best_coloring = list(current)
    
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick k random positions to flip
        positions = rng.sample(range(N), k)
        
        # Compute total delta for flipping all k positions
        # We need to be careful: flipping multiple positions interacts
        # Compute delta by actually flipping and counting, then unflipping
        old_vals = [current[p] for p in positions]
        for p in positions:
            current[p] = 1 - current[p]
        
        # Update mono_flags for all affected APs
        affected_aps = set()
        for p in positions:
            for (ap_idx, role) in pos_aps[p]:
                affected_aps.add(ap_idx)
        
        new_count = 0
        for ap_idx in affected_aps:
            a, b, c, d = aps[ap_idx]
            mono_flags[ap_idx] = (current[a] == current[b] == current[c] == current[d])
        
        # Recompute full count (expensive but safe)
        new_count = _count_mono(current, aps)
        
        if new_count < best_count:
            best_count = new_count
            best_coloring = list(current)
            current_count = new_count
        else:
            # Undo the flip
            for p, ov in zip(positions, old_vals):
                current[p] = ov
            # Restore mono_flags
            for ap_idx in affected_aps:
                a, b, c, d = aps[ap_idx]
                mono_flags[ap_idx] = (current[a] == current[b] == current[c] == current[d])
    
    return best_coloring, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s buffer for safety
    
    aps = _precompute_aps()
    pos_aps = _build_pos_aps(aps)
    
    # Start with Thue-Morse
    best_coloring = _thue_morse()
    best_count = _count_mono(best_coloring, aps)
    
    # Phase 1: Greedy local search from Thue-Morse
    if time.time() < deadline - 60:
        lc, lc_count = _local_search_greedy(best_coloring, aps, pos_aps, max_iters=3000, deadline=deadline - 60)
        if lc_count < best_count:
            best_count = lc_count
            best_coloring = lc
    
    # Phase 2: Simulated annealing from best so far
    if time.time() < deadline - 40:
        sa, sa_count = _simulated_annealing(best_coloring, aps, pos_aps, T_start=3.0, T_end=0.05, max_iters=5000, deadline=deadline - 40, seed=42)
        if sa_count < best_count:
            best_count = sa_count
            best_coloring = sa
        # Follow up with greedy
        if time.time() < deadline - 35:
            lc2, lc2_count = _local_search_greedy(best_coloring, aps, pos_aps, max_iters=2000, deadline=deadline - 35)
            if lc2_count < best_count:
                best_count = lc2_count
                best_coloring = lc2
    
    # Phase 3: Multiple random restarts
    rng = random.Random(99)
    num_restarts = 0
    while time.time() < deadline - 15:
        # Generate a random starting coloring
        coloring = [rng.randint(0, 1) for _ in range(N)]
        init_count = _count_mono(coloring, aps)
        
        # Greedy local search
        lc, lc_count = _local_search_greedy(coloring, aps, pos_aps, max_iters=1500, deadline=deadline - 15)
        if lc_count < best_count:
            best_count = lc_count
            best_coloring = lc
        
        # SA from the greedy result
        if time.time() < deadline - 15:
            sa, sa_count = _simulated_annealing(best_coloring, aps, pos_aps, T_start=2.0, T_end=0.05, max_iters=2000, deadline=deadline - 15, seed=num_restarts + 100)
            if sa_count < best_count:
                best_count = sa_count
                best_coloring = sa
            # Greedy again
            if time.time() < deadline - 15:
                lc2, lc2_count = _local_search_greedy(best_coloring, aps, pos_aps, max_iters=1000, deadline=deadline - 15)
                if lc2_count < best_count:
                    best_count = lc2_count
                    best_coloring = lc2
        
        num_restarts += 1
        
        # Also try structured patterns periodically
        if num_restarts % 3 == 0:
            # Block patterns
            for block_size in [2, 3, 4, 5, 6, 7, 8]:
                coloring2 = [(i // block_size) % 2 for i in range(N)]
                c2 = _count_mono(coloring2, aps)
                if c2 < best_count:
                    best_count = c2
                    best_coloring = coloring2
                else:
                    lc3, lc3_count = _local_search_greedy(coloring2, aps, pos_aps, max_iters=500, deadline=deadline - 15)
                    if lc3_count < best_count:
                        best_count = lc3_count
                        best_coloring = lc3
    
    # Final polish: one more greedy pass
    if time.time() < deadline - 5:
        lc_final, lc_final_count = _local_search_greedy(best_coloring, aps, pos_aps, max_iters=1000, deadline=deadline - 5)
        if lc_final_count < best_count:
            best_count = lc_final_count
            best_coloring = lc_final
    
    return [int(c) for c in best_coloring]
