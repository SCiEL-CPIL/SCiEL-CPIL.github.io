import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # Precompute all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # Precompute coverage mask for each k-subset
    # ksub_masks[j] has bit i set if k-subset j covers t-subset i
    ksub_masks = [0] * n_ksubs
    ksub_tsub_indices = []  # list of t-subset indices for each k-subset (for local search)
    for j, ks in enumerate(all_ksubs):
        mask = 0
        tsub_indices = []
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
            tsub_indices.append(idx)
        ksub_masks[j] = mask
        ksub_tsub_indices.append(tsub_indices)

    # Phase 1: Bitset-based greedy set cover
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs
    selected_ksub_indices = []  # indices into all_ksubs

    while remaining > 0 and time.time() < deadline - 20.0:
        best_j = -1
        best_gain = 0

        for j in range(n_ksubs):
            if j in set(selected_ksub_indices):
                continue
            common = uncovered_mask & ksub_masks[j]
            gain = common.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain == 0:
            break

        selected_ksub_indices.append(best_j)
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain

    # Convert to blocks
    blocks = [list(all_ksubs[j]) for j in selected_ksub_indices]

    # Phase 2: Redundancy removal (local search)
    # Build coverage count for each t-subset
    covered_count = [0] * n_tsubs
    for j in selected_ksub_indices:
        for idx in ksub_tsub_indices[j]:
            covered_count[idx] += 1

    # Try to remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        for i in range(len(selected_ksub_indices) - 1, -1, -1):
            j = selected_ksub_indices[i]
            # Check if block j is redundant (all its t-subsets covered by others)
            redundant = True
            for idx in ksub_tsub_indices[j]:
                if covered_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block j
                for idx in ksub_tsub_indices[j]:
                    covered_count[idx] -= 1
                selected_ksub_indices.pop(i)
                changed = True
                break  # Restart scan

    # Phase 3: 1-swap local search
    # Try replacing one block with another to enable removal of a third block
    for iteration in range(30):
        if time.time() > deadline - 5.0:
            break

        improved = False
        for i in range(len(selected_ksub_indices)):
            if time.time() > deadline - 5.0:
                break

            j_old = selected_ksub_indices[i]
            old_block = all_ksubs[j_old]
            old_tsubs = ksub_tsub_indices[j_old]

            # Temporarily remove block
            for idx in old_tsubs:
                covered_count[idx] -= 1

            # Find uncovered t-subsets after removal
            uncovered_after = []
            for idx in old_tsubs:
                if covered_count[idx] == 0:
                    uncovered_after.append(idx)

            if not uncovered_after:
                # Block was redundant, keep it removed
                selected_ksub_indices.pop(i)
                improved = True
                break

            # Try to find a replacement block that covers all uncovered_after
            # and doesn't add too much redundancy
            best_new_j = -1
            best_new_gain = 0
            best_new_block = None

            # Candidate points: from uncovered t-subsets + old block
            candidate_points = set()
            for idx in uncovered_after:
                candidate_points.update(all_tsubs[idx])
            candidate_points.update(old_block)

            # Try 1-swap: replace one point in old_block with a candidate point
            for x in candidate_points:
                if x in old_block:
                    continue
                # Try removing each point from old_block and adding x
                for y in old_block:
                    new_block = list(old_block)
                    new_block.remove(y)
                    new_block.append(x)
                    new_block.sort()
                    new_block_tuple = tuple(new_block)

                    # Check if this block is already in selected
                    if new_block_tuple in [tuple(all_ksubs[j]) for j in selected_ksub_indices]:
                        continue

                    # Find index of this block in all_ksubs
                    # Use a lookup for efficiency
                    # For now, compute gain directly
                    gain = 0
                    for tsub in itertools.combinations(new_block, t):
                        idx = tsub_to_idx[tsub]
                        if idx in uncovered_after:
                            gain += 1

                    if gain > best_new_gain:
                        best_new_gain = gain
                        best_new_j = -1  # We'll handle this differently
                        best_new_block = new_block

            if best_new_block is not None and best_new_gain >= len(uncovered_after):
                # Restore coverage of old block
                for idx in old_tsubs:
                    covered_count[idx] += 1
                # Add new block
                for tsub in itertools.combinations(best_new_block, t):
                    idx = tsub_to_idx[tsub]
                    covered_count[idx] += 1
                # Replace in selected
                selected_ksub_indices[i] = -1  # Mark as replaced
                # We need to add the new block to our tracking
                # This is getting complex; let's simplify
                # For now, just replace the block in our list
                blocks[i] = best_new_block
                improved = True
                break
            else:
                # Restore coverage of old block
                for idx in old_tsubs:
                    covered_count[idx] += 1

        if not improved:
            break

    # Phase 4: Final greedy fill for any remaining uncovered t-subsets
    # Rebuild uncovered mask from covered_count
    uncovered_final = set(i for i in range(n_tsubs) if covered_count[i] == 0)

    while uncovered_final and time.time() < deadline - 2.0:
        seed_ts = next(iter(uncovered_final))
        seed_points = all_tsubs[seed_ts]

        # Build a block containing seed_points
        block = list(seed_points)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(combo + (x,)))
                    if ts in tsub_to_idx and tsub_to_idx[ts] in uncovered_final:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        block.sort()
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            if idx in uncovered_final:
                uncovered_final.discard(idx)
            covered_count[idx] += 1
        blocks.append(block)

    # Final redundancy removal
    # Rebuild covered_count from blocks
    covered_count = [0] * n_tsubs
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            covered_count[idx] += 1

    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            block = blocks[i]
            redundant = True
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                if covered_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for tsub in itertools.combinations(block, t):
                    idx = tsub_to_idx[tsub]
                    covered_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break

    return blocks
