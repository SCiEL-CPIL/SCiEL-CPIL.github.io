"""Deterministic periodic structure search for Erdos minimum overlap."""
import time
import numpy as np

N = 100
DEADLINE = 145.0  # Safety margin

# EVOLVE-BLOCK-START
def _score(h):
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    """
    Robustly adjust h to sum to N/2 while keeping values in [0, 1].
    Uses a simple iterative clipping and redistribution.
    """
    target = N / 2.0
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    
    # Iterative correction
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            # Need to increase sum. Add to elements with most room (lowest values).
            # To avoid clipping issues, distribute evenly among those < 1.
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            # Sort indices by current value to prioritize filling low ones? 
            # Actually, uniform distribution among available room is safer for sum.
            # But if we just add diff/len(mask), we might clip some.
            # Let's do a proportional fill based on room.
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            # Scale the addition so we don't exceed room
            add_factor = min(1.0, diff / total_room)
            h[mask] += room * add_factor
        else:
            # Need to decrease sum. Subtract from elements with most value (highest values).
            mask = h > 1e-9
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            sub_factor = min(1.0, -diff / total_val)
            h[mask] -= val * sub_factor
            
    # Final exact fix if still off (should be negligible)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Find an index that can absorb the tiny difference
        if diff > 0:
            idx = np.argmin(h)
            if h[idx] < 1.0:
                h[idx] += diff
        else:
            idx = np.argmax(h)
            if h[idx] > 0.0:
                h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _generate_sawtooth(period, phase, slope):
    """
    Generate a sawtooth wave that wraps around.
    h[i] = (i/period + phase) % 1.0
    This creates a sharp discontinuity every 'period' steps.
    """
    x = np.arange(N)
    h = (x / period + phase) % 1.0
    # Optional: apply a slope scaling? The problem fixes sum to 0.5 average.
    # A pure sawtooth (0..1) has average 0.5.
    # If we scale it, we break the sum constraint unless we center it.
    # The constraint is sum(h) = 50. Average = 0.5.
    # Standard sawtooth (0..1) has average 0.5. So it's perfect.
    return h

def _generate_triangle(period, phase):
    """
    Generate a triangle wave oscillating between 0 and 1.
    Average is 0.5.
    """
    x = np.arange(N)
    # Map x to [0, period)
    t = (x / period + phase) % 1.0
    # Triangle wave: 0 -> 1 -> 0 over [0, 1]
    h = 1.0 - np.abs(2.0 * t - 1.0)
    return h

def _generate_binary_blocks(period, phase):
    """
    Generate a binary square wave.
    0 for first half of period, 1 for second half.
    Average is 0.5.
    """
    x = np.arange(N)
    t = (x / period + phase) % 1.0
    h = np.where(t < 0.5, 0.0, 1.0)
    return h

def _local_refine(h0, max_iters=200, step=0.05):
    """
    Simple coordinate descent / random step refinement.
    """
    h = h0.copy()
    current_score = _score(h)
    rng = np.random.RandomState(42)
    
    for _ in range(max_iters):
        # Pick a few random indices to perturb
        num_perturb = 3
        indices = rng.choice(N, size=num_perturb, replace=False)
        h_new = h.copy()
        for idx in indices:
            h_new[idx] += rng.uniform(-step, step)
        
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)
        
        if not _is_valid(h_new):
            continue
            
        new_score = _score(h_new)
        if new_score < current_score:
            current_score = new_score
            h = h_new
        else:
            # Revert
            pass
            
    return h, current_score

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE

    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    candidates = []
    
    # 1. Sawtooth waves
    # Periods that divide 100 or are close to factors
    periods = [2, 4, 5, 10, 20, 25, 50, 100]
    # Also try non-divisors to avoid perfect aliasing?
    # But for N=100, non-divisors might create messy boundaries.
    # Let's stick to divisors first for clean structure.
    for p in periods:
        # Try a few phases to shift the discontinuity
        for phase in [0.0, 0.1, 0.25, 0.5]:
            h = _generate_sawtooth(p, phase, 1.0)
            h = _fix_sum(h)
            if _is_valid(h):
                candidates.append(h)
                
    # 2. Triangle waves
    for p in periods:
        for phase in [0.0, 0.25, 0.5, 0.75]:
            h = _generate_triangle(p, phase)
            h = _fix_sum(h)
            if _is_valid(h):
                candidates.append(h)
                
    # 3. Binary Blocks
    for p in periods:
        for phase in [0.0, 0.25, 0.5, 0.75]:
            h = _generate_binary_blocks(p, phase)
            h = _fix_sum(h)
            if _is_valid(h):
                candidates.append(h)
                
    # 4. Mixed / Randomized Periodic
    # Generate random "periodic" patterns by choosing a period and random values
    # that repeat, ensuring average 0.5.
    rng = np.random.RandomState(100)
    for _ in range(20):
        p = rng.choice(periods)
        # Create a base pattern of length p
        base = rng.uniform(0, 1, p)
        # Adjust base to sum to p/2
        base = _fix_sum(np.tile(base, N // p + 1)[:N])
        # Actually, tiling and slicing might break sum slightly.
        # Better: create base of length p, fix sum to p/2, then tile.
        base = np.array([rng.uniform(0, 1) for _ in range(p)])
        # Fix base sum to p/2
        # Simple fix for small p
        target_base = p / 2.0
        diff = target_base - np.sum(base)
        base += diff / p
        base = np.clip(base, 0, 1)
        # If clipping broke sum, re-fix
        base = _fix_sum(base)
        
        h = np.tile(base, N // p + 1)[:N]
        h = _fix_sum(h)
        if _is_valid(h):
            candidates.append(h)

    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        if deadline_check():
            break
        s = _score(h)
        scored_candidates.append((s, h))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top candidates
    top_k = 5
    for s, h in scored_candidates[:top_k]:
        if deadline_check():
            break
        
        # Local refinement
        h_ref, s_ref = _local_refine(h, max_iters=300, step=0.1)
        
        if s_ref < best_score:
            best_score = s_ref
            best_h = h_ref
    
    # Final check
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
        
    return best_h
# EVOLVE-BLOCK-END
