"""Model-and-optimize approach: structured parameterization + local search refinement."""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110  # safety margin before 150s kill

    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # For each position, which APs does it participate in?
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps_arr[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))

    pos_ap_indices = []
    pos_ap_offsets = []
    for i in range(n):
        if pos_aps[i]:
            indices, offsets = zip(*pos_aps[i])
            pos_ap_indices.append(np.array(indices, dtype=np.int32))
            pos_ap_offsets.append(np.array(offsets, dtype=np.int8))
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_offsets.append(np.array([], dtype=np.int8))

    def count_mono_4ap(color):
        c = color[aps_arr]
        mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        return int(np.sum(mono))

    def delta_flip(color, pos):
        ap_idx = pos_ap_indices[pos]
        if len(ap_idx) == 0:
            return 0
        sub = color[aps_arr[ap_idx]]
        off = pos_ap_offsets[pos]
        # Count current monochromatic APs involving this position
        # For each AP, check if all 4 positions have same color as color[pos]
        c0 = color[pos]
        # Check if the other 3 positions match c0
        # sub has shape (len(ap_idx), 4), off tells which column is 'pos'
        # We need: for each row, the other 3 entries equal c0
        # Equivalent: count rows where exactly 3 entries (excluding off) equal c0
        # And currently the row is mono iff all 4 equal (i.e., off entry also equals c0, which it does by definition)
        # After flip, pos becomes 1-c0, so mono iff other 3 all equal 1-c0
        other = sub.copy()
        other[np.arange(len(ap_idx)), off] = c0  # set pos to c0 (its current value)
        # curr_mono: all 4 equal -> other 3 equal c0 (and pos is c0)
        curr_mono = np.all(other == c0, axis=1)
        curr_count = int(np.sum(curr_mono))
        # After flip: pos = 1-c0, mono iff other 3 all equal 1-c0
        new_mono = np.all(other == (1 - c0), axis=1)
        new_count = int(np.sum(new_mono))
        return new_count - curr_count

    def local_search(color, score, time_budget=5.0):
        """Greedy local search with random restarts within budget."""
        start = time.time()
        # First do standard greedy passes
        improved = True
        while improved and time.time() - start < time_budget * 0.6:
            improved = False
            for pos in range(n):
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        # If time remains, try some random perturbations + re-optimize
        if time.time() - start < time_budget * 0.8:
            rng = np.random.default_rng(42)
            for _ in range(3):
                if time.time() - start >= time_budget:
                    break
                # Perturb a few random positions
                for _ in range(5):
                    pos = rng.integers(0, n)
                    color[pos] = 1 - color[pos]
                    score += delta_flip(color, pos)
                # Re-optimize
                improved = True
                while improved and time.time() - start < time_budget:
                    improved = False
                    for pos in range(n):
                        delta = delta_flip(color, pos)
                        if delta < 0:
                            color[pos] = 1 - color[pos]
                            score += delta
                            improved = True
        return color, score

    def simulated_annealing(color, score, time_budget=10.0):
        """Vectorized simulated annealing with batched random number generation."""
        start = time.time()
        T_start = 2.0
        T_end = 0.01
        rng = np.random.default_rng(777)
        
        # Pre-generate random positions and random numbers in batches
        batch_size = 10000
        total_iters = 0
        
        while time.time() - start < time_budget:
            # Determine how many iterations we can do in this batch
            remaining_time = time_budget - (time.time() - start)
            # Estimate iterations per second based on elapsed
            if total_iters > 0:
                elapsed = time.time() - start
                iters_per_sec = total_iters / max(elapsed, 0.001)
            else:
                iters_per_sec = 50000  # initial estimate
            this_batch = min(batch_size, int(remaining_time * iters_per_sec * 0.9))
            if this_batch < 100:
                break
            
            # Generate batch of random positions and random numbers
            positions = rng.integers(0, n, size=this_batch)
            rand_vals = rng.random(size=this_batch)
            
            # Compute temperature schedule for this batch
            elapsed_frac = (time.time() - start) / time_budget
            T_curr = T_start * (T_end / T_start) ** min(1.0, elapsed_frac)
            
            for i in range(this_batch):
                pos = positions[i]
                delta = delta_flip(color, pos)
                if delta <= 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                else:
                    # Accept with probability exp(-delta/T)
                    if rand_vals[i] < np.exp(-delta / max(T_curr, 0.001)):
                        color[pos] = 1 - color[pos]
                        score += delta
                total_iters += 1
                
                # Periodically check time
                if i % 2000 == 0 and time.time() - start >= time_budget:
                    break
        
        return color, score

    def generate_structured_colorings():
        """Generate a set of structured candidate colorings (model-based)."""
        candidates = []

        # Thue-Morse
        tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(tm.copy())

        # Bit-reversal Thue-Morse
        br = np.array([bin(i)[::-1].count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(br.copy())

        # Modular patterns: color[i] = (i mod p) < q for various p, q
        for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            for q in range(1, p):
                c = np.array([(i % p) < q for i in range(n)], dtype=np.int8)
                candidates.append(c)
            # Also flipped
            for q in range(1, p):
                c = np.array([(i % p) >= q for i in range(n)], dtype=np.int8)
                candidates.append(c)

        # Walsh functions: color[i] = bit k of i
        for k in range(8):
            c = np.array([(i >> k) & 1 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # XOR combinations of bits (Walsh-Hadamard basis)
        for mask in range(1, 256):
            c = np.array([bin(i & mask).count('1') % 2 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # Linear phase: color[i] = (a*i + b) mod 2
        for a in range(1, 8):
            for b in range(2):
                c = np.array([(a * i + b) % 2 for i in range(n)], dtype=np.int8)
                candidates.append(c)

        # Quadratic: color[i] = (a*i*i + b*i + c) mod 2
        for a in range(1, 4):
            for b in range(2):
                for c in range(2):
                    col = np.array([(a * i * i + b * i + c) % 2 for i in range(n)], dtype=np.int8)
                    candidates.append(col)

        # Sine-wave quantization
        for freq in [1, 2, 3, 4, 5, 7, 8, 11]:
            for phase in range(4):
                vals = np.sin(2 * np.pi * freq * np.arange(n) / n + phase * np.pi / 2)
                c = (vals > 0).astype(np.int8)
                candidates.append(c)

        # Random starting points
        rng = np.random.default_rng(12345)
        for _ in range(20):
            c = rng.integers(0, 2, n).astype(np.int8)
            candidates.append(c)

        return candidates

    best_color = None
    best_score = float('inf')

    # Evaluate all structured candidates, keep the best few for refinement
    candidates = generate_structured_colorings()

    # Score all candidates
    scored = []
    for c in candidates:
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))

    scored.sort(key=lambda x: x[0])

    # Take top candidates for refinement
    top_k = min(20, len(scored))
    remaining_time = deadline - time.time()
    
    # Phase 1: Quick local search on top candidates to find good basins
    quick_ls_results = []
    for rank in range(min(top_k, 10)):
        if time.time() > deadline - 30:
            break
        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()
        
        # Quick local search
        color_ls, score_ls = local_search(color.copy(), score, time_budget=2.0)
        if score_ls < best_score:
            best_score = score_ls
            best_color = color_ls.copy()
        quick_ls_results.append((score_ls, color_ls.copy()))
    
    # Phase 2: SA on the best quick-LS results
    quick_ls_results.sort(key=lambda x: x[0])
    sa_starts = min(5, len(quick_ls_results))
    
    for rank in range(sa_starts):
        if time.time() > deadline - 20:
            break
        score, color = quick_ls_results[rank]
        
        # SA with generous budget
        time_remaining = deadline - time.time() - 10
        if time_remaining < 3:
            break
        sa_budget = min(12.0, time_remaining * 0.6)
        color_sa, score_sa = simulated_annealing(color.copy(), score, time_budget=sa_budget)
        if score_sa < best_score:
            best_score = score_sa
            best_color = color_sa.copy()
        
        # Final local search after SA
        time_remaining = deadline - time.time() - 5
        if time_remaining > 2:
            color_final, score_final = local_search(color_sa.copy(), score_sa, time_budget=min(4.0, time_remaining * 0.5))
            if score_final < best_score:
                best_score = score_final
                best_color = color_final.copy()
    
    # Phase 3: Final polish on the overall best
    if best_color is not None and time.time() < deadline - 5:
        color_final, score_final = local_search(best_color.copy(), best_score, time_budget=deadline - time.time() - 5)
        if score_final < best_score:
            best_score = score_final
            best_color = color_final

    if best_color is None:
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    return [int(c) for c in best_color]
