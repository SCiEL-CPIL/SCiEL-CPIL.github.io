import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Precompute all t-subsets as bitmasks for fast coverage tracking
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-blocks and their t-subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, store the list of t-subset indices it covers
    block_tsub_indices = []
    for b in all_blocks:
        tsub_ids = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsub_indices.append(tsub_ids)

    # Cyclic group Z_v: orbits of blocks under rotation
    # A block {a1, a2, ..., ak} has orbit { {a1+i, a2+i, ..., ak+i} mod v : i=0..v-1 }
    # Two blocks are in the same orbit if one is a rotation of the other.

    def canonical_form(block):
        """Find the lexicographically smallest rotation of the block."""
        best = None
        for i in range(v):
            rotated = tuple(sorted((x + i) % v for x in block))
            if best is None or rotated < best:
                best = rotated
        return best

    # Group blocks into orbits
    orbit_map = {}  # canonical form -> list of block indices
    for i, b in enumerate(all_blocks):
        cf = canonical_form(b)
        if cf not in orbit_map:
            orbit_map[cf] = []
        orbit_map[cf].append(i)

    # For each orbit, compute the union of t-subsets covered by all blocks in the orbit
    orbit_tsub_union = {}  # canonical form -> set of t-subset indices covered by orbit
    for cf, indices in orbit_map.items():
        union_set = set()
        for idx in indices:
            union_set.update(block_tsub_indices[idx])
        orbit_tsub_union[cf] = union_set

    # Stage 1: Greedy orbit selection
    # Select orbits to maximize coverage of t-subsets
    uncovered = np.ones(num_tsubs, dtype=bool)
    orbit_coverage_count = np.zeros(num_tsubs, dtype=np.int32)
    selected_orbits = []  # list of canonical forms
    selected_block_indices = []

    # Precompute orbit gain: number of currently uncovered t-subsets in each orbit
    # Initially all t-subsets are uncovered
    orbit_gains = {}
    for cf, union in orbit_tsub_union.items():
        orbit_gains[cf] = len(union)

    # Greedy: repeatedly pick orbit with max gain
    while time.time() < deadline - 30:
        # Find orbit with max gain
        best_cf = None
        best_gain = -1
        for cf, gain in orbit_gains.items():
            if gain > best_gain:
                best_gain = gain
                best_cf = cf

        if best_cf is None or best_gain <= 0:
            break

        selected_orbits.append(best_cf)
        # Add all blocks from this orbit
        for idx in orbit_map[best_cf]:
            selected_block_indices.append(idx)

        # Update coverage and gains
        for ts_idx in orbit_tsub_union[best_cf]:
            if uncovered[ts_idx]:
                uncovered[ts_idx] = False
                orbit_coverage_count[ts_idx] = 1
                # Update gains for all other orbits that cover this t-subset
                # This is expensive; do it lazily or skip for speed
            else:
                orbit_coverage_count[ts_idx] += 1

        # Recompute gains (simplified: just subtract 1 for each newly covered t-subset)
        # For efficiency, we'll recompute at the end or use a simpler approach
        # Let's just break after a few orbits to save time for repair
        if len(selected_orbits) >= 10:
            break

    # Stage 2: Greedy repair with individual blocks
    # Build coverage count for selected blocks
    coverage = np.zeros(num_tsubs, dtype=np.int32)
    for idx in selected_block_indices:
        for ts_idx in block_tsub_indices[idx]:
            coverage[ts_idx] += 1

    # Find uncovered t-subsets
    uncovered_mask = coverage == 0
    if not np.any(uncovered_mask):
        # Already covered!
        pass
    else:
        # Greedy: pick blocks that cover the most uncovered t-subsets
        # gains[i] = number of uncovered t-subsets in block i
        gains = np.zeros(num_blocks, dtype=np.int32)
        for i in range(num_blocks):
            gains[i] = sum(1 for ts_idx in block_tsub_indices[i] if uncovered_mask[ts_idx])

        selected_set = set(selected_block_indices)

        while np.any(uncovered_mask) and time.time() < deadline - 15:
            max_gain = np.max(gains)
            if max_gain <= 0:
                break

            best_idx = np.argmax(gains)
            if best_idx in selected_set:
                # Already selected, skip
                gains[best_idx] = -1
                continue

            selected_block_indices.append(best_idx)
            selected_set.add(best_idx)

            # Update coverage and gains
            for ts_idx in block_tsub_indices[best_idx]:
                coverage[ts_idx] += 1
                if uncovered_mask[ts_idx]:
                    uncovered_mask[ts_idx] = False
                    # Decrement gain for all blocks covering this t-subset
                    for other_idx in range(num_blocks):
                        if other_idx != best_idx and uncovered_mask[ts_idx] == False:
                            # Check if other block covers this t-subset
                            if ts_idx in block_tsub_indices[other_idx]:
                                gains[other_idx] -= 1

    # Stage 3: Local search - remove redundant blocks
    selected_set = set(selected_block_indices)

    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(selected_block_indices) - 1, -1, -1):
            b_idx = selected_block_indices[i]
            is_redundant = True
            for ts_idx in block_tsub_indices[b_idx]:
                # Count selected blocks covering ts_idx
                count = 0
                for other_idx in selected_set:
                    if ts_idx in block_tsub_indices[other_idx]:
                        count += 1
                        if count > 1:
                            break
                if count <= 1:
                    is_redundant = False
                    break

            if is_redundant:
                selected_block_indices.pop(i)
                selected_set.remove(b_idx)
                changed = True

    # Convert to final blocks
    final_blocks = [all_blocks[i] for i in selected_block_indices]

    return final_blocks
