"""
Structural rewrite: batched local search with vectorized marginal-gain computation.
Instead of random restarts + SA + greedy, we run many short batched local-search
schedules from diverse seeds. Each schedule repeatedly:
  1. For every position, compute the delta if flipped (vectorized over all 256 positions).
  2. Apply the most negative delta (or a set of them) to reduce the count.
  3. Repeat until no improvement or time budget.
We also include a "multi-flip" mode that flips the top-k most beneficial positions
simultaneously to escape single-flip plateaus.
"""
import time
import random
import numpy as np


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit

    # Precompute all 4-term APs
    aps = []
    for d in range(1, 86):
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            aps.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    aps = np.array(aps, dtype=np.int32)
    n_aps = len(aps)

    # For each AP, store the 4 indices
    ap_i0 = aps[:, 0]
    ap_i1 = aps[:, 1]
    ap_i2 = aps[:, 2]
    ap_i3 = aps[:, 3]

    # Build adjacency: for each position, list of (ap_idx, role)
    pos_aps = [[] for _ in range(256)]
    for idx in range(n_aps):
        pos_aps[ap_i0[idx]].append((idx, 0))
        pos_aps[ap_i1[idx]].append((idx, 1))
        pos_aps[ap_i2[idx]].append((idx, 2))
        pos_aps[ap_i3[idx]].append((idx, 3))

    # Precompute per-position AP index arrays and role arrays for fast access
    pos_ap_indices = []
    pos_ap_roles = []
    for pos in range(256):
        idxs = [x[0] for x in pos_aps[pos]]
        roles = [x[1] for x in pos_aps[pos]]
        pos_ap_indices.append(np.array(idxs, dtype=np.int32))
        pos_ap_roles.append(np.array(roles, dtype=np.int32))

    def count_mono(colors):
        c0 = colors[ap_i0]
        c1 = colors[ap_i1]
        c2 = colors[ap_i2]
        c3 = colors[ap_i3]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())

    def compute_all_deltas(colors):
        """
        For each position 0..255, compute the delta in mono count if we flip that position.
        Returns an array of length 256 with deltas.
        """
        deltas = np.zeros(256, dtype=np.int32)
        for pos in range(256):
            old_color = colors[pos]
            new_color = 1 - old_color
            ap_idxs = pos_ap_indices[pos]
            roles = pos_ap_roles[pos]
            if len(ap_idxs) == 0:
                continue
            # Get colors of all 4 positions in each AP
            c0 = colors[ap_i0[ap_idxs]]
            c1 = colors[ap_i1[ap_idxs]]
            c2 = colors[ap_i2[ap_idxs]]
            c3 = colors[ap_i3[ap_idxs]]
            # Currently mono?
            was_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
            # After flip: the color at 'pos' changes to new_color
            # For each AP, check if all 4 would be new_color
            # We need to check: the three OTHER positions all equal new_color
            # Build a mask for each role
            # For role 0: check c1==new_color, c2==new_color, c3==new_color
            # For role 1: check c0==new_color, c2==new_color, c3==new_color
            # etc.
            # More efficiently: create a 4xN boolean matrix where entry [r, j] is True if the
            # color at role r in AP j equals new_color, then check all 4 are True.
            # But we can do it per-role with masks.
            is_mono_new = np.zeros(len(ap_idxs), dtype=bool)
            for r in range(4):
                mask = (roles == r)
                if not np.any(mask):
                    continue
                # For these APs, pos is at role r, so we need the other 3 to be new_color
                # The other roles are 0,1,2,3 except r
                all_true = np.ones(len(ap_idxs), dtype=bool)
                for rr in range(4):
                    if rr == r:
                        continue
                    # Get color at role rr for these APs
                    if rr == 0:
                        cc = c0
                    elif rr == 1:
                        cc = c1
                    elif rr == 2:
                        cc = c2
                    else:
                        cc = c3
                    all_true &= (cc == new_color)
                # Only apply to APs where role == r
                is_mono_new[mask] = all_true[mask]

            delta = np.sum(~was_mono & is_mono_new) - np.sum(was_mono & ~is_mono_new)
            deltas[pos] = delta
        return deltas

    def batched_local_search(colors, current_count, max_rounds=200, k=1):
        """
        Repeatedly compute all deltas, flip the k most beneficial positions, until no improvement.
        """
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            deltas = compute_all_deltas(colors)
            # Find positions with negative delta
            neg_mask = deltas < 0
            if not np.any(neg_mask):
                break
            # Sort by delta (most negative first)
            neg_positions = np.where(neg_mask)[0]
            neg_deltas = deltas[neg_positions]
            order = np.argsort(neg_deltas)
            neg_positions = neg_positions[order]
            neg_deltas = neg_deltas[order]

            # Flip up to k positions
            n_flip = min(k, len(neg_positions))
            for i in range(n_flip):
                pos = neg_positions[i]
                colors[pos] = 1 - colors[pos]
                current_count += neg_deltas[i]

            improved = True
            rounds += 1

        return colors, current_count

    def multi_flip_local_search(colors, current_count, max_rounds=100, k=5):
        """
        Flip the top-k most beneficial positions simultaneously each round.
        This can escape plateaus that single-flip can't.
        """
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            deltas = compute_all_deltas(colors)
            neg_mask = deltas < 0
            if not np.any(neg_mask):
                break
            neg_positions = np.where(neg_mask)[0]
            neg_deltas = deltas[neg_positions]
            order = np.argsort(neg_deltas)
            neg_positions = neg_positions[order]
            neg_deltas = neg_deltas[order]

            n_flip = min(k, len(neg_positions))
            # Apply all flips simultaneously (compute delta assuming only those positions change)
            # For simplicity, apply sequentially but this is an approximation
            for i in range(n_flip):
                pos = neg_positions[i]
                colors[pos] = 1 - colors[pos]
                current_count += neg_deltas[i]

            improved = True
            rounds += 1

        return colors, current_count

    # Main search loop
    best_colors = None
    best_count = float('inf')

    # Try multiple random seeds
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        random.seed(seed)
        colors = np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)
        current_count = count_mono(colors)

        # Phase 1: single-flip local search
        colors, current_count = batched_local_search(colors, current_count, max_rounds=100, k=1)

        # Phase 2: multi-flip local search
        colors, current_count = multi_flip_local_search(colors, current_count, max_rounds=50, k=5)

        # Phase 3: more single-flip
        colors, current_count = batched_local_search(colors, current_count, max_rounds=100, k=1)

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # If we've hit a good count, try to refine further
        if best_count <= 30:
            # Extra refinement rounds
            colors, current_count = batched_local_search(colors, current_count, max_rounds=50, k=1)
            colors, current_count = multi_flip_local_search(colors, current_count, max_rounds=30, k=3)
            colors, current_count = batched_local_search(colors, current_count, max_rounds=50, k=1)
            if current_count < best_count:
                best_count = current_count
                best_colors = colors.copy()

    # Final refinement on best found
    if best_colors is not None and time.time() < deadline - 10:
        colors = best_colors.copy()
        current_count = best_count
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            colors, current_count = batched_local_search(colors, current_count, max_rounds=50, k=1)
            colors, current_count = multi_flip_local_search(colors, current_count, max_rounds=20, k=3)
            colors, current_count = batched_local_search(colors, current_count, max_rounds=50, k=1)
            if current_count < best_count:
                best_count = current_count
                best_colors = colors.copy()

    return [int(c) for c in best_colors]
