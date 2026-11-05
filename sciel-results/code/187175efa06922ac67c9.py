import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before the 150s kill

    def now():
        return time.time()

    # ---- Precompute all t-subsets ----
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # ---- Precompute all k-blocks ----
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # ---- Incidence structures ----
    block_to_subsets = []          # list of np.array[int32]
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx, block in enumerate(all_blocks):
        subs = []
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            subs.append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)
        block_to_subsets.append(np.array(subs, dtype=np.int32))

    # Convert subset_to_blocks to numpy arrays for fast vectorized updates
    subset_blocks_arr = [np.array(lst, dtype=np.int32) for lst in subset_to_blocks]

    # ---- Weighted gain (Gonzalez-Park heuristic) ----
    # weight[u] = 1 / (# blocks covering u): rare subsets get higher priority
    weights = np.empty(n_subsets, dtype=np.float64)
    for u in range(n_subsets):
        weights[u] = 1.0 / len(subset_blocks_arr[u])

    uncovered = np.ones(n_subsets, dtype=bool)
    remaining = n_subsets
    gains = np.zeros(n_blocks, dtype=np.float64)

    # Initial gains: gains[b] = sum of weights of all (currently uncovered) subsets in b
    for u in range(n_subsets):
        np.add.at(gains, subset_blocks_arr[u], weights[u])

    blocks_found = []
    used_blocks = set()
    coverage_count = np.zeros(n_subsets, dtype=np.int32)

    # ---- Greedy construction ----
    while remaining > 0:
        if now() > deadline - 12.0:
            break

        b_star = int(np.argmax(gains))
        if gains[b_star] <= 0:
            break

        blocks_found.append(list(all_blocks[b_star]))
        used_blocks.add(b_star)

        # Mark newly-covered subsets
        newly = uncovered[block_to_subsets[b_star]]
        newly_idx = block_to_subsets[b_star][newly]
        if newly_idx.size:
            uncovered[newly_idx] = False
            remaining -= newly_idx.size
            # Decrement gain for every block containing a newly-covered subset
            w_new = weights[newly_idx]
            for b in subset_blocks_arr:
                np.subtract.at(gains, b, np.take(w_new,
                            np.searchsorted(block_to_subsets[b], newly_idx,
                                            'left')[
                                np.searchsorted(block_to_subsets[b], newly_idx,
                                                'left') < block_to_subsets[b].size
                                ]) if False else 0)  # placeholder (unused)

            # Fast path: for each newly covered subset, subtract its weight from all blocks containing it
            for u in newly_idx:
                np.subtract.at(gains, subset_blocks_arr[u], weights[u])

        # Update coverage count
        np.add.at(coverage_count, block_to_subsets[b_star], 1)

    # ---- Polish pass 1: targeted completion of any leftover uncovered subsets ----
    if remaining > 0 and now() < deadline - 6.0:
        uncov_indices = np.where(uncovered)[0]
        for u_idx in uncov_indices:
            if now() > deadline - 3.0:
                break
            if not uncovered[u_idx]:
                continue
            # pick the unused block with max gain that covers u_idx
            best_b = -1
            best_g = -1.0
            for b in subset_blocks_arr[u_idx]:
                if b not in used_blocks and gains[b] > best_g:
                    best_g = gains[b]
                    best_b = b
            if best_b < 0:
                # fallback: any unused block covering it
                for b in subset_blocks_arr[u_idx]:
                    if b not in used_blocks:
                        best_b = b
                        break
            if best_b < 0:
                continue
            blocks_found.append(list(all_blocks[best_b]))
            used_blocks.add(best_b)
            newly = uncovered[block_to_subsets[best_b]]
            newly_idx = block_to_subsets[best_b][newly]
            if newly_idx.size:
                uncovered[newly_idx] = False
                remaining -= newly_idx.size
                for u in newly_idx:
                    np.subtract.at(gains, subset_blocks_arr[u], weights[u])
            np.add.at(coverage_count, block_to_subsets[best_b], 1)

    # ---- Polish pass 2: repeated redundant-block removal ----
    # A block is removable if every t-subset it covers is covered by >= 2 blocks.
    def try_remove(b_idx):
        nonlocal coverage_count
        if coverage_count[block_to_subsets[b_idx]].min() <= 1:
            return False
        np.subtract.at(coverage_count, block_to_subsets[b_idx], 1)
        # remove from blocks_found
        content = list(all_blocks[b_idx])
        for i, bl in enumerate(blocks_found):
            if bl == content:
                del blocks_found[i]
                break
        used_blocks.discard(b_idx)
        return True

    # Iterate until no more removals (fixed point)
    changed = True
    while changed and now() < deadline - 3.0:
        changed = False
        for b_idx in list(used_blocks):
            if try_remove(b_idx):
                changed = True

    # ---- Polish pass 3: swap-based local search ----
    # Try dropping one block, then greedily re-covering the newly-exposed subsets.
    # Keep the change only if the net block count decreases.
    if now() < deadline - 4.0:
        for b_idx in list(used_blocks):
            if now() > deadline - 2.0:
                break
            # simulate removal
            subs = block_to_subsets[b_idx]
            if coverage_count[subs].min() <= 1:
                continue  # would break coverage
            # temporarily remove
            np.subtract.at(coverage_count, subs, 1)
            exposed = subs[coverage_count[subs] == 0]
            # re-cover exposed subsets with best unused blocks
            saved_blocks = len(blocks_found)
            added = []
            ok = True
            for u in exposed:
                best_b = -1
                best_g = -1.0
                for b in subset_blocks_arr[u]:
                    if b not in used_blocks and gains[b] > best_g:
                        best_g = gains[b]
                        best_b = b
                if best_b < 0:
                    ok = False
                    break
                added.append(best_b)
            if not ok or len(added) >= 1:
                # undo everything
                np.add.at(coverage_count, subs, 1)
                continue
            # accept: remove b_idx, add nothing (exposed were already covered? impossible)
            # In practice exposed>0 requires additions; accept only if net improves.
            np.add.at(coverage_count, subs, 1)

    return blocks_found
