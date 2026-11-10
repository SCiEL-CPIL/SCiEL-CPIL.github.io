import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Store as 0-indexed tuples
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(aps)

    # For each index, precompute the list of AP indices it belongs to
    # and for each such AP, which other 3 indices are in it
    index_to_aps = [[] for _ in range(N)]
    for ap_idx, (i1, i2, i3, i4) in enumerate(aps):
        for idx in (i1, i2, i3, i4):
            others = [x for x in (i1, i2, i3, i4) if x != idx]
            index_to_aps[idx].append((ap_idx, others))

    def count_mono(color):
        """Count monochromatic 4-APs."""
        count = 0
        for i1, i2, i3, i4 in aps:
            c = color[i1]
            if c == color[i2] == color[i3] == color[i4]:
                count += 1
        return count

    def get_delta(color, idx, new_val):
        """Compute change in mono count if we flip color[idx] to new_val."""
        old_val = color[idx]
        if old_val == new_val:
            return 0
        delta = 0
        for ap_idx, others in index_to_aps[idx]:
            # Check if the other 3 are all the same color
            o0, o1, o2 = others
            if color[o0] == color[o1] == color[o2]:
                # The other 3 are monochromatic with color c_others
                c_others = color[o0]
                if old_val == c_others:
                    # Currently mono. If new_val != c_others, it becomes non-mono.
                    if new_val != c_others:
                        delta -= 1
                else:
                    # Currently non-mono (idx is different from others).
                    # If new_val == c_others, it becomes mono.
                    if new_val == c_others:
                        delta += 1
        return delta

    def local_search(color, max_iter=1000):
        """Hill climbing: flip bits if it improves the score."""
        current_score = count_mono(color)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for idx in range(N):
                new_val = 1 - color[idx]
                delta = get_delta(color, idx, new_val)
                if delta < 0:
                    color[idx] = new_val
                    current_score += delta
                    improved = True
            iter_count += 1
        return current_score

    # Generate good initial configurations
    def random_init():
        return np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8)

    def block_init(block_size):
        return np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8)

    def stripe_init(period):
        return np.array([(i % period) < period // 2 for i in range(N)]).astype(np.int8)

    candidates = []
    
    # Block patterns
    for bs in [2, 3, 4, 5, 6, 7, 8, 16]:
        candidates.append(block_init(bs))
    
    # Stripe patterns
    for p in [3, 5, 7, 9, 11, 13]:
        candidates.append(stripe_init(p))
    
    # Random inits
    for _ in range(20):
        candidates.append(random_init())
    
    # Evaluate and refine candidates
    scored = []
    for c in candidates:
        c = c.copy()
        # Quick local search to get a local optimum
        sc = local_search(c, max_iter=100)
        scored.append((sc, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top 5 as starting points for SA
    top_starts = [c.copy() for _, c in scored[:5]]
    best_score = scored[0][0]
    best_color = scored[0][1].copy()

    # Simulated Annealing
    # We will run multiple SA runs, each with a different random seed and starting point
    # If time allows, we restart SA from the best found so far
    
    rng = random.Random()
    
    while time.time() < deadline - 10.0:
        # Pick a starting point: either one of the top starts or the current best
        if rng.random() < 0.5 and top_starts:
            start_color = top_starts[rng.randint(0, len(top_starts)-1)].copy()
        else:
            start_color = best_color.copy()
            
        # Perturb the start color slightly to avoid immediate cycles
        for _ in range(5):
            idx = rng.randint(0, N - 1)
            start_color[idx] = 1 - start_color[idx]
        
        color = start_color.copy()
        current_score = count_mono(color)
        
        # SA parameters
        T = 2.0
        T_min = 0.01
        cooling = 0.998
        max_iter = 50000
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Choose move: 1-bit flip (80%) or 3-bit flip (20%)
            if rng.random() < 0.8:
                idx = rng.randint(0, N - 1)
                new_val = 1 - color[idx]
                delta = get_delta(color, idx, new_val)
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    color[idx] = new_val
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()
            else:
                # 3-bit flip
                idxs = rng.sample(range(N), 3)
                idx1, idx2, idx3 = idxs
                v1 = 1 - color[idx1]
                v2 = 1 - color[idx2]
                v3 = 1 - color[idx3]
                
                # Compute combined delta
                # Note: If indices are in the same AP, the deltas are not independent.
                # However, for 4-APs, it's rare for 3 random indices to be in the same AP.
                # Even if they are, the error is small and SA can handle it.
                # To be safe, we can check if they are in the same AP, but for speed we assume independence.
                # A more robust way: apply one, compute delta, apply second, compute delta, etc.
                # But that requires 3 get_delta calls.
                delta1 = get_delta(color, idx1, v1)
                color[idx1] = v1
                delta2 = get_delta(color, idx2, v2)
                color[idx2] = v2
                delta3 = get_delta(color, idx3, v3)
                total_delta = delta1 + delta2 + delta3
                
                # Restore if rejected
                if total_delta < 0 or rng.random() < np.exp(-total_delta / T):
                    color[idx3] = v3
                    # idx1 and idx2 are already set
                    current_score += total_delta
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()
                else:
                    color[idx1] = 1 - v1
                    color[idx2] = 1 - v2
                    # idx3 is not set yet, so it's still old value
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # After each SA run, do a quick local search on the best found
        if time.time() < deadline - 5.0:
            local_search(best_color, max_iter=50)
            # Update best_score if local search improved it
            new_sc = count_mono(best_color)
            if new_sc < best_score:
                best_score = new_sc

    # Final exhaustive local search
    if time.time() < deadline - 1.0:
        local_search(best_color, max_iter=1000)
        best_score = count_mono(best_color)

    return [int(x) for x in best_color]
