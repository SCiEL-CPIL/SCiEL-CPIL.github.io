"""Max-SAT formulation with vectorized noise-injected local search and dynamic temperature."""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115  # safety margin before 150s kill

    # Precompute all 4-term arithmetic progressions (0-indexed: positions 0..255)
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
        new_sub = sub.copy()
        new_sub[np.arange(len(ap_idx)), off] = 1 - new_sub[np.arange(len(ap_idx)), off]
        new_mono = (new_sub[:, 1] == new_sub[:, 0]) & (new_sub[:, 2] == new_sub[:, 0]) & (new_sub[:, 3] == new_sub[:, 0])
        new_count = int(np.sum(new_mono))
        return new_count - curr_count

    def local_search(color, score, time_budget=5.0):
        """Greedy local search: keep flipping if it improves."""
        improved = True
        start = time.time()
        while improved and time.time() - start < time_budget:
            improved = False
            for pos in range(n):
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        return color, score

    def maxsat_local_search(color, score, time_budget=15.0):
        """
        Max-SAT style local search with weighted clauses and dynamic temperature.
        Each 4-AP is a clause that is 'violated' (monochromatic). We want to minimize
        the number of violated clauses. We use a noise-injected approach where we
        consider flipping positions that are in many violated clauses, with dynamic
        temperature to escape local minima.
        """
        start = time.time()
        T_start = 3.0
        T_end = 0.01
        best_score = score
        best_color = color.copy()
        
        # Track violation count per position (how many mono APs it's part of)
        # This helps prioritize flips
        
        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0.0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.2
            
            # Compute which APs are currently monochromatic
            c = color[aps_arr]
            mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
            num_violated = int(np.sum(mono))
            
            if num_violated == 0:
                break
            
            # Compute violation weight per position: how many violated APs it's in
            violation_weights = np.zeros(n, dtype=np.int32)
            violated_aps = np.where(mono)[0]
            for ap_idx in violated_aps:
                p0, p1, p2, p3 = aps_arr[ap_idx]
                violation_weights[p0] += 1
                violation_weights[p1] += 1
                violation_weights[p2] += 1
                violation_weights[p3] += 1
            
            # Choose a position to flip: biased toward high violation weight
            # with some randomness controlled by temperature
            if np.random.random() < 0.7:
                # Biased selection: pick from positions with non-zero violation weight
                candidates = np.where(violation_weights > 0)[0]
                if len(candidates) == 0:
                    candidates = np.arange(n)
                # Weighted random selection
                weights = violation_weights[candidates].astype(np.float64)
                weights = weights ** 2  # Amplify preference for high-weight positions
                total_w = weights.sum()
                if total_w > 0:
                    probs = weights / total_w
                    pos = candidates[np.random.choice(len(candidates), p=probs)]
                else:
                    pos = candidates[np.random.randint(len(candidates))]
            else:
                # Random position
                pos = np.random.randint(0, n)
            
            delta = delta_flip(color, pos)
            
            if delta <= 0:
                color[pos] = 1 - color[pos]
                score += delta
            else:
                # Accept with probability based on temperature
                accept_prob = np.exp(-delta / max(T, 0.001))
                if np.random.random() < accept_prob:
                    color[pos] = 1 - color[pos]
                    score += delta
            
            # Track best
            if score < best_score:
                best_score = score
                best_color = color.copy()
                
        return best_color, best_score

    def multi_flip_sa(color, score, time_budget=10.0):
        """Simulated annealing with multi-flip moves for escaping local minima."""
        start = time.time()
        T_start = 4.0
        T_end = 0.005
        best_score = score
        best_color = color.copy()

        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0.0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.5

            # Choose move type: 60% single flip, 40% multi-flip
            if np.random.random() < 0.6:
                pos = np.random.randint(0, n)
                delta = delta_flip(color, pos)
                if delta <= 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                else:
                    if np.random.random() < np.exp(-delta / max(T, 0.001)):
                        color[pos] = 1 - color[pos]
                        score += delta
            else:
                # Multi-flip: pick 2-4 random distinct positions
                k = np.random.randint(2, 5)
                positions = np.random.choice(n, size=k, replace=False)
                affected_aps = set()
                for p in positions:
                    if len(pos_ap_indices[p]) > 0:
                        affected_aps.update(pos_ap_indices[p].tolist())
                if affected_aps:
                    ap_list = np.array(sorted(affected_aps), dtype=np.int32)
                    sub = color[aps_arr[ap_list]]
                    curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
                    curr_count = int(np.sum(curr_mono))
                    new_sub = sub.copy()
                    for p in positions:
                        off_arr = pos_ap_offsets[p]
                        idx_arr = pos_ap_indices[p]
                        for j, ap_i in enumerate(idx_arr):
                            row_in_new = np.where(ap_list == ap_i)[0]
                            if len(row_in_new) > 0:
                                new_sub[row_in_new[0], off_arr[j]] = 1 - new_sub[row_in_new[0], off_arr[j]]
                    new_mono = (new_sub[:, 1] == new_sub[:, 0]) & (new_sub[:, 2] == new_sub[:, 0]) & (new_sub[:, 3] == new_sub[:, 0])
                    new_count = int(np.sum(new_mono))
                    delta = new_count - curr_count
                    if delta <= 0:
                        for p in positions:
                            color[p] = 1 - color[p]
                        score += delta
                    else:
                        if np.random.random() < np.exp(-delta / max(T, 0.001)):
                            for p in positions:
                                color[p] = 1 - color[p]
                            score += delta

            if score < best_score:
                best_score = score
                best_color = color.copy()

        return best_color, best_score

    def generate_structured_colorings():
        """Generate a set of structured candidate colorings."""
        candidates = []

        # Thue-Morse
        tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(tm.copy())

        # Bit-reversal Thue-Morse
        br = np.array([bin(i)[::-1].count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(br.copy())

        # Modular patterns
        for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            for q in range(1, p):
                c = np.array([(i % p) < q for i in range(n)], dtype=np.int8)
                candidates.append(c)

        # Walsh functions
        for k in range(8):
            c = np.array([(i >> k) & 1 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # XOR combinations of bits
        for mask in range(1, 256):
            c = np.array([bin(i & mask).count('1') % 2 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # Random starting points
        rng = np.random.default_rng(42)
        for _ in range(30):
            c = rng.integers(0, 2, n).astype(np.int8)
            candidates.append(c)

        return candidates

    best_color = None
    best_score = float('inf')

    # Evaluate all structured candidates
    candidates = generate_structured_colorings()
    scored = []
    for c in candidates:
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))

    scored.sort(key=lambda x: x[0])

    # Take top candidates for refinement
    top_k = min(25, len(scored))
    
    for rank in range(top_k):
        if time.time() > deadline - 25:
            break
        
        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()

        # Phase 1: Greedy local search
        ls_budget = min(3.0, (deadline - time.time() - 25) * 0.15)
        if ls_budget > 0.5:
            color_ls, score_ls = local_search(color.copy(), score, time_budget=ls_budget)
            if score_ls < best_score:
                best_score = score_ls
                best_color = color_ls.copy()
        else:
            color_ls, score_ls = color.copy(), score

        # Phase 2: Max-SAT local search with dynamic temperature
        msat_budget = min(10.0, (deadline - time.time() - 25) * 0.35)
        if msat_budget > 1.0:
            color_msat, score_msat = maxsat_local_search(color_ls.copy(), score_ls, time_budget=msat_budget)
            if score_msat < best_score:
                best_score = score_msat
                best_color = color_msat.copy()
        else:
            color_msat, score_msat = color_ls.copy(), score_ls

        # Phase 3: Multi-flip SA for escaping local minima
        sa_budget = min(8.0, (deadline - time.time() - 20) * 0.3)
        if sa_budget > 1.0:
            color_sa, score_sa = multi_flip_sa(color_msat.copy(), score_msat, time_budget=sa_budget)
            if score_sa < best_score:
                best_score = score_sa
                best_color = color_sa.copy()
        else:
            color_sa, score_sa = color_msat.copy(), score_msat

        # Phase 4: Final local search polish
        final_ls_budget = min(3.0, deadline - time.time() - 15)
        if final_ls_budget > 0.5:
            color_final, score_final = local_search(color_sa.copy(), score_sa, time_budget=final_ls_budget)
            if score_final < best_score:
                best_score = score_final
                best_color = color_final.copy()

    # Final polish on the overall best
    if best_color is not None and time.time() < deadline - 10:
        final_budget = deadline - time.time() - 8
        color_final, score_final = local_search(best_color.copy(), best_score, time_budget=final_budget * 0.5)
        if score_final < best_score:
            best_score = score_final
            best_color = color_final.copy()

        if time.time() < deadline - 12:
            sa_final_budget = min(8.0, deadline - time.time() - 12)
            color_sa_f, score_sa_f = maxsat_local_search(best_color.copy(), best_score, time_budget=sa_final_budget)
            if score_sa_f < best_score:
                best_score = score_sa_f
                best_color = color_sa_f.copy()

            if time.time() < deadline - 6:
                color_final2, score_final2 = local_search(best_color.copy(), best_score, time_budget=deadline - time.time() - 6)
                if score_final2 < best_score:
                    best_score = score_final2
                    best_color = color_final2.copy()

    if best_color is None:
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    return [int(c) for c in best_color]
