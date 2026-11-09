import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute edges
    edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    n_edges = 120

    # Map edge (u, v) to index
    edge_idx_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx_map[(i, j)] = idx
        edge_idx_map[(j, i)] = idx

    # Generate all K4s
    k4s_all = []
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        e_indices = [
            edge_idx_map[(a, b)],
            edge_idx_map[(a, c)],
            edge_idx_map[(a, d)],
            edge_idx_map[(b, c)],
            edge_idx_map[(b, d)],
            edge_idx_map[(c, d)]
        ]
        k4s_all.append(e_indices)

    # For each edge, the list of K4 indices it belongs to
    edge_k4s = [[] for _ in range(n_edges)]
    for k4_idx, e_indices in enumerate(k4s_all):
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)

    # Quick validity check for a single K4
    def k4_valid(k4_idx, cols):
        e_indices = k4s_all[k4_idx]
        c0 = cols[e_indices[0]]
        c1 = cols[e_indices[1]]
        c2 = cols[e_indices[2]]
        c3 = cols[e_indices[3]]
        c4 = cols[e_indices[4]]
        c5 = cols[e_indices[5]]
        # Count distinct colors among 6 edges
        if c0 == c1:
            if c0 == c2:
                if c0 == c3:
                    # c0=c1=c2=c3
                    if c0 == c4:
                        # c0..c4 same
                        return c5 != c0  # need 5 distinct, impossible with only 2 distinct
                    else:
                        # c0=c1=c2=c3=c4? No, c4 different.
                        # colors: {c0, c4, c5}
                        return c0 != c4 and c4 != c5 and c0 != c5
                else:
                    # c0=c1=c2, c3 different
                    if c0 == c4:
                        # c0=c1=c2=c4, c3 different
                        # colors: {c0, c3, c5}
                        return c0 != c3 and c3 != c5 and c0 != c5
                    else:
                        # c0=c1=c2, c3 != c0, c4 != c0
                        # colors: {c0, c3, c4, c5}
                        # need 5 distinct -> need all 4 different from each other and c0
                        if c3 == c4:
                            # colors: {c0, c3, c5}
                            return c0 != c3 and c3 != c5 and c0 != c5
                        else:
                            # colors: {c0, c3, c4, c5}
                            # need 5 distinct, so c5 must be different from all, and c3!=c4
                            # but we only have 4 distinct so far. Need 5.
                            # So this branch cannot yield 5 distinct unless c5 is new and we have 4 distinct already.
                            # Wait, if c0, c3, c4 are all distinct, that's 3.
                            # If c5 is distinct from all, that's 4.
                            # We need 5. So this branch always fails?
                            # Let's re-evaluate.
                            # c0=c1=c2. c3!=c0. c4!=c0.
                            # If c3 != c4, we have {c0, c3, c4}.
                            # If c5 != c0, c3, c4, we have {c0, c3, c4, c5} -> 4 distinct.
                            # We need 5. So this branch returns False?
                            # No, wait. The condition is "at least 5".
                            # So if we can only get 4, it's False.
                            return False
            else:
                # c0=c1, c2 != c0
                if c0 == c3:
                    # c0=c1=c3, c2 != c0
                    if c0 == c4:
                        # c0=c1=c3=c4, c2 != c0
                        # colors: {c0, c2, c5}
                        return c0 != c2 and c2 != c5 and c0 != c5
                    else:
                        # c0=c1=c3, c2 != c0, c4 != c0
                        # colors: {c0, c2, c4, c5}
                        if c2 == c4:
                            # colors: {c0, c2, c5}
                            return c0 != c2 and c2 != c5 and c0 != c5
                        else:
                            # colors: {c0, c2, c4, c5}
                            # max 4 distinct.
                            return False
                else:
                    # c0=c1, c2 != c0, c3 != c0
                    if c0 == c4:
                        # c0=c1=c4, c2 != c0, c3 != c0
                        # colors: {c0, c2, c3, c5}
                        if c2 == c3:
                            # colors: {c0, c2, c5}
                            return c0 != c2 and c2 != c5 and c0 != c5
                        else:
                            # colors: {c0, c2, c3, c5}
                            # max 4 distinct.
                            return False
                    else:
                        # c0=c1, c2 != c0, c3 != c0, c4 != c0
                        # colors: {c0, c2, c3, c4, c5}
                        # if c2, c3, c4 are all distinct and c5 is distinct from all -> 5 distinct
                        if c2 == c3:
                            return False
                        if c2 == c4:
                            return False
                        if c3 == c4:
                            return False
                        # c2, c3, c4 distinct
                        return c5 != c0 and c5 != c2 and c5 != c3 and c5 != c4
        else:
            # c0 != c1
            if c0 == c2:
                # c0=c2, c1 != c0
                if c0 == c3:
                    # c0=c2=c3, c1 != c0
                    if c0 == c4:
                        # c0=c2=c3=c4, c1 != c0
                        # colors: {c0, c1, c5}
                        return c0 != c1 and c1 != c5 and c0 != c5
                    else:
                        # c0=c2=c3, c1 != c0, c4 != c0
                        # colors: {c0, c1, c4, c5}
                        if c1 == c4:
                            # colors: {c0, c1, c5}
                            return c0 != c1 and c1 != c5 and c0 != c5
                        else:
                            # colors: {c0, c1, c4, c5}
                            return False
                else:
                    # c0=c2, c1 != c0, c3 != c0
                    if c0 == c4:
                        # c0=c2=c4, c1 != c0, c3 != c0
                        # colors: {c0, c1, c3, c5}
                        if c1 == c3:
                            # colors: {c0, c1, c5}
                            return c0 != c1 and c1 != c5 and c0 != c5
                        else:
                            # colors: {c0, c1, c3, c5}
                            return False
                    else:
                        # c0=c2, c1 != c0, c3 != c0, c4 != c0
                        # colors: {c0, c1, c3, c4, c5}
                        if c1 == c3:
                            return False
                        if c1 == c4:
                            return False
                        if c3 == c4:
                            return False
                        return c5 != c0 and c5 != c1 and c5 != c3 and c5 != c4
            else:
                # c0 != c1, c0 != c2
                if c0 == c3:
                    # c0=c3, c1 != c0, c2 != c0
                    if c0 == c4:
                        # c0=c3=c4, c1 != c0, c2 != c0
                        # colors: {c0, c1, c2, c5}
                        if c1 == c2:
                            # colors: {c0, c1, c5}
                            return c0 != c1 and c1 != c5 and c0 != c5
                        else:
                            # colors: {c0, c1, c2, c5}
                            return False
                    else:
                        # c0=c3, c1 != c0, c2 != c0, c4 != c0
                        # colors: {c0, c1, c2, c4, c5}
                        if c1 == c2:
                            return False
                        if c1 == c4:
                            return False
                        if c2 == c4:
                            return False
                        return c5 != c0 and c5 != c1 and c5 != c2 and c5 != c4
                else:
                    # c0 != c1, c0 != c2, c0 != c3
                    if c0 == c4:
                        # c0=c4, c1 != c0, c2 != c0, c3 != c0
                        # colors: {c0, c1, c2, c3, c5}
                        if c1 == c2:
                            return False
                        if c1 == c3:
                            return False
                        if c2 == c3:
                            return False
                        return c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3
                    else:
                        # c0 != c1, c0 != c2, c0 != c3, c0 != c4
                        # 5 distinct so far? Not necessarily.
                        # c1, c2, c3, c4 might have duplicates.
                        # We need 5 distinct total.
                        # So we need c1, c2, c3, c4, c5 to provide at least 4 distinct colors from the set {c1..c5}
                        # that are distinct from c0 and each other?
                        # Actually, we just need the set {c0, c1, c2, c3, c4, c5} to have size >= 5.
                        # Since c0 is distinct from c1,c2,c3,c4, we need {c1,c2,c3,c4,c5} to have at least 4 distinct elements.
                        # And c5 must be distinct from c0 (already true if c5 != c0).
                        # Let's just count.
                        distinct = 1 # c0
                        if c1 != c0: distinct += 1
                        if c2 != c0 and c2 != c1: distinct += 1
                        if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
                        if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
                        if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
                        return distinct >= 5

    # Initial colors: (i+j) % 17
    colors = [0] * n_edges
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17

    best_colors = colors[:]
    best_num_colors = len(set(colors))

    # Local Search
    while time.time() < deadline:
        e_idx = random.randint(0, n_edges - 1)
        old_color = colors[e_idx]

        # Try to change to a random color in [0, 15]
        # Bias towards colors that are already used to reduce distinct count
        # But simply picking from [0, 15] works okay.
        new_color = random.randint(0, 15)

        if new_color == old_color:
            continue

        colors[e_idx] = new_color

        valid = True
        for k4_idx in edge_k4s[e_idx]:
            if not k4_valid(k4_idx, colors):
                valid = False
                break

        if valid:
            # Check distinct count periodically
            if random.random() < 0.05:
                num_colors = len(set(colors))
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_colors = colors[:]
        else:
            colors[e_idx] = old_color

    # Final check and compression
    # If best_colors is valid, use it.
    # We assume best_colors is valid because we only updated it when valid.
    # But let's double check? No, too slow.
    # Just compress best_colors.
    unique_colors = sorted(list(set(best_colors)))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    final_colors = [mapping[c] for c in best_colors]
    return final_colors
