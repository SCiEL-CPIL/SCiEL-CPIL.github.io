import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_count = float('inf')
    best_edges = None
    best_n = None

    # Precompute for each n
    for n in range(8, 15):
        if time.time() > deadline - 5:
            break

        total_colorings = 1 << n
        all_ones = (1 << total_colorings) - 1

        # Precompute M0 and M1
        M0 = [0] * n
        M1 = [0] * n
        for c in range(total_colorings):
            for v in range(n):
                if (c >> v) & 1:
                    M1[v] |= (1 << c)
                else:
                    M0[v] |= (1 << c)

        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)

        # Precompute edge masks
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)

        # Run multiple restarts
        max_restarts = 200
        restarts_done = 0

        for r in range(max_restarts):
            if time.time() > deadline - 3:
                break
            restarts_done += 1

            random.seed(1000 + n * 100 + r)

            # Strategy 1: Greedy set cover
            if r % 3 == 0:
                covered = 0
                selected_indices = []
                remaining = list(range(num_edges))
                random.shuffle(remaining)

                while covered != all_ones and remaining:
                    best_gain = -1
                    best_idx = -1
                    for idx in remaining:
                        gain = (edge_masks[idx] & ~covered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx
                    if best_gain == 0:
                        break
                    selected_indices.append(best_idx)
                    covered |= edge_masks[best_idx]
                    remaining.remove(best_idx)

                if covered == all_ones:
                    # Local deletion
                    improved = True
                    max_iters = 50
                    iter_count = 0
                    while improved and iter_count < max_iters and time.time() < deadline - 3:
                        improved = False
                        iter_count += 1
                        indices_to_check = list(range(len(selected_indices)))
                        random.shuffle(indices_to_check)
                        for i in indices_to_check:
                            if i >= len(selected_indices):
                                continue
                            e_idx = selected_indices[i]
                            new_covered = 0
                            for j in range(len(selected_indices)):
                                if j != i:
                                    new_covered |= edge_masks[selected_indices[j]]
                            if new_covered == all_ones:
                                selected_indices.pop(i)
                                covered = new_covered
                                improved = True
                                break

                    if len(selected_indices) < best_count:
                        best_count = len(selected_indices)
                        best_edges = [list(all_edges[i]) for i in selected_indices]
                        best_n = n

            # Strategy 2: Random initialization + local search with swaps
            elif r % 3 == 1:
                # Start with a random subset of edges
                target_size = 40 + (r % 10)
                selected_indices = random.sample(range(num_edges), min(target_size, num_edges))

                # Compute coverage
                covered = 0
                for idx in selected_indices:
                    covered |= edge_masks[idx]

                # Local search: try to add edges that help cover, remove redundant
                improved = True
                max_iters = 30
                iter_count = 0
                while improved and iter_count < max_iters and time.time() < deadline - 3:
                    improved = False
                    iter_count += 1

                    # First, try removing redundant edges
                    indices_to_check = list(range(len(selected_indices)))
                    random.shuffle(indices_to_check)
                    for i in indices_to_check:
                        if i >= len(selected_indices):
                            continue
                        e_idx = selected_indices[i]
                        new_covered = 0
                        for j in range(len(selected_indices)):
                            if j != i:
                                new_covered |= edge_masks[selected_indices[j]]
                        if new_covered == all_ones:
                            selected_indices.pop(i)
                            covered = new_covered
                            improved = True
                            break

                    if improved:
                        continue

                    # If not fully covered, try adding edges
                    if covered != all_ones:
                        # Find edges that add the most new coverage
                        uncovered = ~covered & all_ones
                        best_add = -1
                        best_gain = 0
                        random_subset = random.sample(range(num_edges), min(100, num_edges))
                        for idx in random_subset:
                            if idx in selected_indices:
                                continue
                            gain = (edge_masks[idx] & uncovered).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_add = idx
                        if best_add >= 0 and best_gain > 0:
                            selected_indices.append(best_add)
                            covered |= edge_masks[best_add]
                            improved = True

                if covered == all_ones and len(selected_indices) < best_count:
                    best_count = len(selected_indices)
                    best_edges = [list(all_edges[i]) for i in selected_indices]
                    best_n = n

            # Strategy 3: Start from greedy, then do swap-based local search
            else:
                # Greedy to get a solution
                covered = 0
                selected_indices = []
                remaining = list(range(num_edges))
                random.shuffle(remaining)

                while covered != all_ones and remaining:
                    best_gain = -1
                    best_idx = -1
                    for idx in remaining:
                        gain = (edge_masks[idx] & ~covered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx
                    if best_gain == 0:
                        break
                    selected_indices.append(best_idx)
                    covered |= edge_masks[best_idx]
                    remaining.remove(best_idx)

                if covered == all_ones:
                    # Swap-based local search
                    improved = True
                    max_iters = 40
                    iter_count = 0
                    while improved and iter_count < max_iters and time.time() < deadline - 3:
                        improved = False
                        iter_count += 1

                        # Try removing
                        indices_to_check = list(range(len(selected_indices)))
                        random.shuffle(indices_to_check)
                        for i in indices_to_check:
                            if i >= len(selected_indices):
                                continue
                            e_idx = selected_indices[i]
                            new_covered = 0
                            for j in range(len(selected_indices)):
                                if j != i:
                                    new_covered |= edge_masks[selected_indices[j]]
                            if new_covered == all_ones:
                                selected_indices.pop(i)
                                covered = new_covered
                                improved = True
                                break

                        if improved:
                            continue

                        # Try swaps: remove one, add one different
                        if len(selected_indices) > 10:
                            swap_found = False
                            for i in range(len(selected_indices)):
                                if swap_found or time.time() > deadline - 3:
                                    break
                                removed_idx = selected_indices[i]
                                temp_covered = 0
                                for j in range(len(selected_indices)):
                                    if j != i:
                                        temp_covered |= edge_masks[selected_indices[j]]
                                if temp_covered == all_ones:
                                    continue  # Already redundant, handle above

                                # Try adding random edges
                                candidates = random.sample(range(num_edges), min(50, num_edges))
                                best_add = -1
                                best_new_size = len(selected_indices)  # We want smaller
                                for add_idx in candidates:
                                    if add_idx == removed_idx:
                                        continue
                                    if add_idx in selected_indices:
                                        continue
                                    # New set = current minus removed_idx plus add_idx
                                    new_size = len(selected_indices)  # same size
                                    # Check if this swap helps (we want to eventually remove more)
                                    # For now, just track if we can maintain coverage
                                    new_covered = temp_covered | edge_masks[add_idx]
                                    if new_covered == all_ones:
                                        # This swap maintains coverage; we'll use it later
                                        # For now, skip swaps that don't reduce size
                                        pass
                                # Skip complex swap logic for now

                    if covered == all_ones and len(selected_indices) < best_count:
                        best_count = len(selected_indices)
                        best_edges = [list(all_edges[i]) for i in selected_indices]
                        best_n = n

    if best_edges is None:
        # Fallback: all edges on n=9
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
