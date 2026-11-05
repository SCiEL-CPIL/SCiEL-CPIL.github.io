import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 118.0  # safety margin before 120s (evaluator kills at 150s, but be safe)

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute balls (neighbors) as lists of indices
    nb_count = 1 + n * (q - 1)
    balls = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        balls[idx] = nb_list

    # Precompute ball bitsets (each ball as a boolean vector over space)
    # For space_size=2187, a bitset is 2187 bools ~ 2KB; 2187 of them ~ 4.4MB, fine.
    ball_bits = np.zeros((space_size, space_size), dtype=np.int8)
    for c in range(space_size):
        ball_bits[c, balls[c]] = 1

    rng = np.random.default_rng(12345)

    # --- Greedy construction ---
    uncovered = np.ones(space_size, dtype=np.int8)
    total_uncovered = space_size
    code_indices = []
    selected_mask = np.zeros(space_size, dtype=bool)

    while total_uncovered > 0 and (time.time() - t_start) < deadline - 50.0:
        # Score: number of currently uncovered words covered by candidate c
        scores = ball_bits @ uncovered.astype(np.int32)
        scores[selected_mask] = -1
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
        code_indices.append(best_idx)
        selected_mask[best_idx] = True
        uncovered[balls[best_idx]] = 0
        total_uncovered -= int(np.sum(ball_bits[best_idx] & uncovered.astype(np.int8)))
        # Recompute total properly
        total_uncovered = int(np.sum(uncovered))

    # Ensure full coverage
    while total_uncovered > 0 and (time.time() - t_start) < deadline - 45.0:
        scores = ball_bits @ uncovered.astype(np.int32)
        scores[selected_mask] = -1
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
        code_indices.append(best_idx)
        selected_mask[best_idx] = True
        uncovered[balls[best_idx]] = 0
        total_uncovered = int(np.sum(uncovered))

    # --- Greedy removal to tighten ---
    cov = np.zeros(space_size, dtype=np.int32)
    for c_idx in code_indices:
        cov[balls[c_idx]] += 1

    active = [True] * len(code_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 40.0:
        changed = False
        for i in range(len(code_indices)):
            if not active[i]:
                continue
            c_idx = code_indices[i]
            if np.min(cov[balls[c_idx]]) >= 2:
                active[i] = False
                cov[balls[c_idx]] -= 1
                changed = True

    current_indices = [code_indices[i] for i in range(len(code_indices)) if active[i]]
    best_code = current_indices[:]
    best_size = len(best_code)

    # --- Helper functions for local search ---
    def compute_cov(indices):
        c = np.zeros(space_size, dtype=np.int32)
        for idx in indices:
            c[balls[idx]] += 1
        return c

    def greedy_remove(cov, indices):
        """Remove all redundantly-covered codewords."""
        active = [True] * len(indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(indices)):
                if not active[i]:
                    continue
                idx = indices[i]
                if np.min(cov[balls[idx]]) >= 2:
                    active[i] = False
                    cov[balls[idx]] -= 1
                    changed = True
        return [indices[i] for i in range(len(indices)) if active[i]]

    def re_greedy_cover(cov, existing_indices):
        """Given a partial cover (cov) and existing codewords, greedily add codewords to cover all."""
        indices = list(existing_indices)
        unc = (cov == 0).astype(np.int8)
        total_unc = int(np.sum(unc))
        sel_mask = np.zeros(space_size, dtype=bool)
        for idx in indices:
            sel_mask[idx] = True
        while total_unc > 0 and (time.time() - t_start) < deadline - 5.0:
            scores = ball_bits @ unc
            scores[sel_mask] = -1
            best = int(np.argmax(scores))
            if scores[best] <= 0:
                break
            indices.append(best)
            sel_mask[best] = True
            unc[balls[best]] = 0
            total_unc = int(np.sum(unc))
        return indices

    def rarest_uncovered_swap(cov, indices, T, rng_local):
        """
        Find a swap that covers the rarest uncovered word.
        Returns (new_indices, new_cov) if accepted, else None.
        """
        unc = (cov == 0).astype(np.int8)
        if np.sum(unc) == 0:
            return None  # already valid; try deletion instead

        # Find the rarest uncovered word (covered by fewest existing codewords... but it's uncovered, so 0)
        # Instead: pick a random uncovered word, find candidates that cover it
        unc_list = np.where(unc > 0)[0]
        if len(unc_list) == 0:
            return None
        target = int(unc_list[int(rng_local.integers(0, len(unc_list)))])

        # Find all codewords whose ball contains target
        # ball_bits[:, target] gives which codewords cover target
        candidates = np.where(ball_bits[:, target] > 0)[0]
        # Exclude already-selected
        sel_set = set(indices)
        candidates = [c for c in candidates if c not in sel_set]
        if not candidates:
            return None

        # Pick a random candidate to add
        new_c = candidates[int(rng_local.integers(0, len(candidates)))]

        # Pick a codeword to remove: prefer one that covers fewest unique words
        # (i.e., its ball is mostly covered by others)
        # Compute for each current codeword: how many words would become uncovered if removed
        # That's: count of words in ball where cov==1
        remove_candidates = list(indices)
        if len(remove_candidates) < 2:
            return None

        # Score each removal candidate by "damage": number of words in its ball with cov==1
        # We want to remove the one with least damage
        best_rem = None
        best_damage = float('inf')
        for rc in remove_candidates:
            b = balls[rc]
            damage = int(np.sum(cov[b] == 1))
            if damage < best_damage:
                best_damage = damage
                best_rem = rc

        if best_damage > 0 and T < 0.5:
            # Only accept if damage is small
            pass

        # Tentative swap: remove best_rem, add new_c
        new_cov = cov.copy()
        new_cov[balls[best_rem]] -= 1
        new_cov[balls[new_c]] += 1

        new_unc = int(np.sum(new_cov == 0))
        old_unc = int(np.sum(unc))

        delta = (new_unc - old_unc) * 5  # penalty weight for uncovered count
        if delta <= 0 or (T > 0 and rng_local.random() < np.exp(-delta / T)):
            new_indices = [c for c in indices if c != best_rem]
            new_indices.append(new_c)
            return (new_indices, new_cov)
        return None

    # --- Local search loop ---
    cur_indices = best_code[:]
    cur_cov = compute_cov(cur_indices)
    cur_size = len(cur_indices)

    T_start = 3.0
    T_end = 0.1
    sa_budget = deadline - (time.time() - t_start) - 8.0
    if sa_budget < 10.0:
        sa_budget = 10.0

    step = 0
    stall_count = 0
    max_stall = 300

    while (time.time() - t_start) < deadline - 8.0:
        elapsed_frac = 1.0 - (deadline - 8.0 - (time.time() - t_start)) / sa_budget
        elapsed_frac = max(0.0, min(1.0, elapsed_frac))
        T = T_start * (T_end / T_start) ** elapsed_frac

        r = rng.random()
        improved = False

        if r < 0.35:
            # Deletion pass: remove redundant codewords
            new_indices = greedy_remove(cur_cov.copy(), cur_indices)
            if len(new_indices) < cur_size:
                cur_indices = new_indices
                cur_cov = compute_cov(cur_indices)
                cur_size = len(cur_indices)
                improved = True
                stall_count = 0
                if cur_size < best_size:
                    best_size = cur_size
                    best_code = cur_indices[:]

        elif r < 0.65:
            # Rarest-uncovered swap
            result = rarest_uncovered_swap(cur_cov, cur_indices, T, rng)
            if result is not None:
                new_indices, new_cov = result
                cur_indices = new_indices
                cur_cov = new_cov
                cur_size = len(cur_indices)
                improved = True
                stall_count = 0
                if cur_size < best_size and int(np.sum(cur_cov == 0)) == 0:
                    best_size = cur_size
                    best_code = cur_indices[:]

        elif r < 0.85:
            # Kick + re-greedy: remove k random codewords, re-cover greedily
            k = max(1, min(5, cur_size // 5))
            remove_set = set(rng.choice(cur_indices, size=k, replace=False).tolist())
            remaining = [c for c in cur_indices if c not in remove_set]
            # Compute coverage of remaining
            rem_cov = np.zeros(space_size, dtype=np.int32)
            for c in remaining:
                rem_cov[balls[c]] += 1
            # Re-greedy cover
            new_indices = re_greedy_cover(rem_cov, remaining)
            if len(new_indices) < cur_size:
                cur_indices = new_indices
                cur_cov = compute_cov(cur_indices)
                cur_size = len(cur_indices)
                improved = True
                stall_count = 0
                if cur_size < best_size and int(np.sum(cur_cov == 0)) == 0:
                    best_size = cur_size
                    best_code = cur_indices[:]

        else:
            # Random 2-swap for diversity
            if cur_size < 2:
                step += 1
                continue
            idx_a = int(rng.integers(0, cur_size))
            idx_b = int(rng.integers(0, cur_size))
            if idx_a == idx_b:
                step += 1
                continue
            rem_a = cur_indices[idx_a]
            rem_b = cur_indices[idx_b]
            new_a = int(rng.integers(0, space_size))
            new_b = int(rng.integers(0, space_size))
            if new_a in cur_indices or new_b in cur_indices or new_a == new_b:
                step += 1
                continue

            new_cov = cur_cov.copy()
            new_cov[balls[rem_a]] -= 1
            new_cov[balls[rem_b]] -= 1
            new_cov[balls[new_a]] += 1
            new_cov[balls[new_b]] += 1

            new_unc = int(np.sum(new_cov == 0))
            old_unc = int(np.sum(cur_cov == 0))
            delta = (new_unc - old_unc) * 5
            if delta <= 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                new_indices = [c for c in cur_indices if c != rem_a and c != rem_b]
                new_indices.extend([new_a, new_b])
                cur_indices = new_indices
                cur_cov = new_cov
                cur_size = len(cur_indices)
                improved = True
                stall_count = 0
                if cur_size < best_size and int(np.sum(cur_cov == 0)) == 0:
                    best_size = cur_size
                    best_code = cur_indices[:]

        if not improved:
            stall_count += 1
            if stall_count > max_stall:
                # Force a kick
                k = max(1, min(4, cur_size // 5))
                remove_set = set(rng.choice(cur_indices, size=k, replace=False).tolist())
                remaining = [c for c in cur_indices if c not in remove_set]
                rem_cov = np.zeros(space_size, dtype=np.int32)
                for c in remaining:
                    rem_cov[balls[c]] += 1
                new_indices = re_greedy_cover(rem_cov, remaining)
                cur_indices = new_indices
                cur_cov = compute_cov(cur_indices)
                cur_size = len(cur_indices)
                stall_count = 0

        step += 1

    # --- Final cleanup ---
    final_indices = best_code[:]
    final_cov = compute_cov(final_indices)

    # Ensure validity
    if int(np.sum(final_cov == 0)) > 0:
        final_indices = re_greedy_cover(final_cov, final_indices)
        final_cov = compute_cov(final_indices)

    # Final greedy removal
    final_indices = greedy_remove(final_cov, final_indices)
    final_cov = compute_cov(final_indices)

    # Verify
    if len(final_indices) == 0:
        return [idx_to_word(i) for i in range(space_size)]

    if int(np.sum(final_cov == 0)) > 0:
        # Fallback to initial greedy
        init_cov = compute_cov(code_indices)
        if int(np.sum(init_cov == 0)) == 0:
            final_indices = code_indices
        else:
            final_indices = list(range(space_size))

    # Ensure distinctness
    seen = set()
    unique = []
    for c in final_indices:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return [idx_to_word(i) for i in unique]
