import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin for 120s limit

    n = 11
    num_colorings = 1 << n
    full_mask = (1 << num_colorings) - 1

    # Generate all 5-element subsets
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)

    # Precompute masks for each edge
    # mask[e] is a bitset of colorings where edge e is monochromatic
    edge_masks = [0] * num_edges
    for i, e in enumerate(all_edges):
        mask_vs = 0
        for v in e:
            mask_vs |= (1 << v)
        mask = 0
        for c in range(num_colorings):
            if (c & mask_vs) == 0 or (c & mask_vs) == mask_vs:
                mask |= (1 << c)
        edge_masks[i] = mask

    # Count how many edges cover each coloring
    color_edge_count = [0] * num_colorings
    for i in range(num_edges):
        m = edge_masks[i]
        while m:
            lsb = m & -m
            c_idx = lsb.bit_length() - 1
            color_edge_count[c_idx] += 1
            m ^= lsb

    best_edges_count = num_edges
    best_chosen = None

    # Deterministic greedy set cover
    def greedy_set_cover():
        uncovered = full_mask
        chosen = []
        remaining = set(range(num_edges))
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = -1
            for idx in remaining:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > best_count:
                    best_count = c
                    best_idx = idx
            if best_idx == -1:
                break
            chosen.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            remaining.discard(best_idx)
        return chosen

    # Initial greedy
    best_chosen = greedy_set_cover()
    best_edges_count = len(best_chosen)

    # Iterative improvement: try to reduce the set cover
    def try_improve(chosen, time_limit):
        chosen_set = set(chosen)
        counts = [0] * num_colorings
        for idx in chosen:
            m = edge_masks[idx]
            while m:
                lsb = m & -m
                c_idx = lsb.bit_length() - 1
                counts[c_idx] += 1
                m ^= lsb

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Try removing edges one by one
            for idx in chosen[:]:
                m = edge_masks[idx]
                can_remove = True
                while m:
                    lsb = m & -m
                    c_idx = lsb.bit_length() - 1
                    if counts[c_idx] == 1:
                        can_remove = False
                        break
                    m ^= lsb

                if can_remove:
                    m = edge_masks[idx]
                    while m:
                        lsb = m & -m
                        c_idx = lsb.bit_length() - 1
                        counts[c_idx] -= 1
                        m ^= lsb
                    chosen.remove(idx)
                    chosen_set.discard(idx)
                    improved = True
                    break

        # Try 1-for-2 swaps: remove 2 edges, add 1
        if time.time() < time_limit and len(chosen) > 51:
            for i in range(len(chosen)):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, len(chosen)):
                    if time.time() > time_limit:
                        break
                    idx_i = chosen[i]
                    idx_j = chosen[j]

                    # Compute which colorings are uniquely covered by these two
                    m_i = edge_masks[idx_i]
                    m_j = edge_masks[idx_j]
                    m_both = m_i | m_j

                    # Check if we can find an edge that covers all uniquely-covered colorings
                    # and doesn't introduce new uncovered colorings
                    uncovered_after_removal = 0
                    temp = m_both
                    while temp:
                        lsb = temp & -temp
                        c_idx = lsb.bit_length() - 1
                        if counts[c_idx] == 1:
                            uncovered_after_removal |= lsb
                        temp ^= lsb

                    if uncovered_after_removal == 0:
                        # Can remove both without adding anything
                        m = edge_masks[idx_i]
                        while m:
                            lsb = m & -m
                            c_idx = lsb.bit_length() - 1
                            counts[c_idx] -= 1
                            m ^= lsb
                        m = edge_masks[idx_j]
                        while m:
                            lsb = m & -m
                            c_idx = lsb.bit_length() - 1
                            counts[c_idx] -= 1
                            m ^= lsb
                        chosen.remove(idx_j)
                        chosen.remove(idx_i)
                        improved = True
                        break

                    # Try to find an edge that covers uncovered_after_removal
                    best_add = -1
                    best_penalty = -1
                    for add_idx in range(num_edges):
                        if add_idx in chosen_set:
                            continue
                        m_add = edge_masks[add_idx]
                        if (uncovered_after_removal & ~m_add) != 0:
                            continue
                        # Check penalty: how many colorings get over-covered
                        penalty = (m_add & ~m_both).bit_count()
                        if penalty > best_penalty:
                            best_penalty = penalty
                            best_add = add_idx

                    if best_add != -1 and best_penalty <= 2:
                        # Do the swap
                        m = edge_masks[idx_i]
                        while m:
                            lsb = m & -m
                            c_idx = lsb.bit_length() - 1
                            counts[c_idx] -= 1
                            m ^= lsb
                        m = edge_masks[idx_j]
                        while m:
                            lsb = m & -m
                            c_idx = lsb.bit_length() - 1
                            counts[c_idx] -= 1
                            m ^= lsb
                        chosen.remove(idx_j)
                        chosen.remove(idx_i)

                        m = edge_masks[best_add]
                        while m:
                            lsb = m & -m
                            c_idx = lsb.bit_length() - 1
                            counts[c_idx] += 1
                            m ^= lsb
                        chosen.append(best_add)
                        chosen_set.add(best_add)
                        improved = True
                        break

                if improved:
                    break

        return chosen, len(chosen)

    # Run improvement loop
    while time.time() < deadline and best_edges_count > 51:
        current_time = time.time()
        improved_chosen, new_count = try_improve(best_chosen[:], deadline)
        if new_count < best_edges_count:
            best_edges_count = new_count
            best_chosen = improved_chosen
        if best_edges_count <= 51:
            break

    result_edges = [list(all_edges[i]) for i in best_chosen]
    return {'n': n, 'edges': result_edges}
