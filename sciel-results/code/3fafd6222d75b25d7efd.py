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
        curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
        curr_count = int(np.sum(curr_mono))
        # Compute new count without full copy: flip the offset column
        new_mono = curr_mono.copy()
        for j in range(len(ap_idx)):
            row = j
            o = off[j]
            # Check if flipping position 'o' in this AP breaks or creates mono
            if curr_mono[row]:
                # Currently mono: flipping any one position breaks it
                new_mono[row] = False
            else:
                # Currently not mono: check if flipping makes it mono
                # Count how many positions differ from position 0
                vals = sub[row]
                # After flip, position o becomes 1 - vals[o]
                # Check if all 4 would be equal
                if o == 0:
                    # New val at pos 0 is 1-vals[0]; check if all others equal that
                    new_val = 1 - vals[0]
                    if vals[1] == new_val and vals[2] == new_val and vals[3] == new_val:
                        new_mono[row] = True
                elif o == 1:
                    new_val = 1 - vals[1]
                    if vals[0] == new_val and vals[2] == new_val and vals[3] == new_val:
                        new_mono[row] = True
                elif o == 2:
                    new_val = 1 - vals[2]
                    if vals[0] == new_val and vals[1] == new_val and vals[3] == new_val:
                        new_mono[row] = True
                elif o == 3:
                    new_val = 1 - vals[3]
                    if vals[0] == new_val and vals[1] == new_val and vals[2] == new_val:
                        new_mono[row] = True
        new_count = int(np.sum(new_mono))
        return new_count - curr_count

    def local_search(color, score, time_budget=5.0):
        """Greedy local search: keep flipping if it improves. Uses random order for diversity."""
        improved = True
        start = time.time()
        rng = np.random.default_rng(42)
        while improved and time.time() - start < time_budget:
            improved = False
            order = rng.permutation(n)
            for pos in order:
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        return color, score

    def simulated_annealing(color, score, time_budget=10.0):
        """Simulated annealing with single and multi-flip moves, optimized for speed."""
        start = time.time()
        T_start = 4.0
        T_end = 0.005
        sa_best_score = score
        sa_best_color = color.copy()
        rng = np.random.default_rng(99)

        # Precompute a mapping from ap_index to row position for multi-flip
        # Build a dict: ap_idx -> row in the affected_aps array (rebuilt each time)
        
        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.5

            # Choose move type: 75% single flip, 25% multi-flip
            if rng.random() < 0.75:
                # Single flip
                pos = rng.integers(0, n)
                delta = delta_flip(color, pos)
                if delta <= 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                else:
                    if rng.random() < np.exp(-delta / max(T, 0.001)):
                        color[pos] = 1 - color[pos]
                        score += delta
            else:
                # Multi-flip: pick 2-4 random distinct positions
                k = rng.integers(2, 5)
                positions = rng.choice(n, size=k, replace=False)
                # Compute total delta by flipping all at once
                affected_aps = set()
                for p in positions:
                    if len(pos_ap_indices[p]) > 0:
                        affected_aps.update(pos_ap_indices[p].tolist())
                if affected_aps:
                    ap_list = np.array(sorted(affected_aps), dtype=np.int32)
                    # Build index map: ap_idx -> row
                    ap_to_row = {int(ap_list[i]): i for i in range(len(ap_list))}
                    sub = color[aps_arr[ap_list]].copy()
                    # Count current mono
                    curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
                    curr_count = int(np.sum(curr_mono))
                    # Apply flips
                    for p in positions:
                        off_arr = pos_ap_offsets[p]
                        idx_arr = pos_ap_indices[p]
                        for j in range(len(idx_arr)):
                            row = ap_to_row.get(int(idx_arr[j]))
                            if row is not None:
                                sub[row, off_arr[j]] = 1 - sub[row, off_arr[j]]
                    new_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
                    new_count = int(np.sum(new_mono))
                    delta = new_count - curr_count
                    if delta <= 0:
                        for p in positions:
                            color[p] = 1 - color[p]
                        score += delta
                    else:
                        if rng.random() < np.exp(-delta / max(T, 0.001)):
                            for p in positions:
                                color[p] = 1 - color[p]
                            score += delta

            # Track best
            if score < sa_best_score:
                sa_best_score = score
                sa_best_color = color.copy()

        # Return best seen during SA
        return sa_best_color, sa_best_score

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

        # XOR combinations of bits (Walsh-Hadamard basis) - select diverse masks
        for mask in range(1, 256, 7):  # Sample every 7th mask for diversity
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

        # Rademacher-like: random sign patterns with structure
        rng = np.random.default_rng(12345)
        for _ in range(30):
            c = rng.integers(0, 2, n).astype(np.int8)
            candidates.append(c)
        
        # Block patterns: alternating blocks of varying sizes
        for block_size in [4, 8, 16, 32, 64]:
            pattern = np.zeros(n, dtype=np.int8)
            for i in range(n):
                pattern[i] = (i // block_size) % 2
            candidates.append(pattern)
            candidates.append(1 - pattern)

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

    # Take top candidates for refinement - more candidates with shorter budgets
    top_k = min(30, len(scored))
    remaining_time = deadline - time.time()
    time_per_candidate = max(0.8, remaining_time / (top_k + 5))

    for rank in range(top_k):
        if time.time() > deadline - 15:
            break
        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()

        # Local search refinement
        ls_budget = min(3.0, time_per_candidate * 0.30)
        color_ls, score_ls = local_search(color.copy(), score, time_budget=ls_budget)
        if score_ls < best_score:
            best_score = score_ls
            best_color = color_ls.copy()

        # Simulated annealing from best local search result
        time_remaining = deadline - time.time() - 15
        if time_remaining > 2:
            sa_budget = min(8.0, time_remaining * 0.50)
            color_sa, score_sa = simulated_annealing(color_ls.copy(), score_ls, time_budget=sa_budget)
            if score_sa < best_score:
                best_score = score_sa
                best_color = color_sa.copy()

            # Final local search after SA
            ls2_budget = min(3.0, time_remaining * 0.25)
            color_final, score_final = local_search(color_sa.copy(), score_sa, time_budget=ls2_budget)
            if score_final < best_score:
                best_score = score_final
                best_color = color_final.copy()

    # Final polish on the overall best: multiple rounds of LS + SA
    if best_color is not None and time.time() < deadline - 8:
        # Round 1: Local search
        if time.time() < deadline - 6:
            ls_budget = min(3.0, (deadline - time.time() - 6) * 0.3)
            color_final, score_final = local_search(best_color.copy(), best_score, time_budget=ls_budget)
            if score_final < best_score:
                best_score = score_final
                best_color = color_final.copy()

        # Round 2: SA
        if time.time() < deadline - 5:
            sa_budget = min(8.0, deadline - time.time() - 5)
            color_sa_f, score_sa_f = simulated_annealing(best_color.copy(), best_score, time_budget=sa_budget)
            if score_sa_f < best_score:
                best_score = score_sa_f
                best_color = color_sa_f.copy()

        # Round 3: Final local search
        if time.time() < deadline - 3:
            ls_budget = deadline - time.time() - 3
            color_final2, score_final2 = local_search(best_color.copy(), best_score, time_budget=ls_budget)
            if score_final2 < best_score:
                best_score = score_final2
                best_color = color_final2.copy()

    if best_color is None:
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    return [int(c) for c in best_color]
