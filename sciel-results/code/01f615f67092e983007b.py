import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices (35 each) for fast iteration
block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Weighted greedy weights: rare subsets get higher priority
weights = [1.0 / f for f in subset_frequencies]


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 112  # Safety margin below 120s kill

    def weighted_greedy():
        """Weighted greedy: pick block maximizing sum of weights over uncovered subsets it covers."""
        uncovered_mask = (1 << N_SUBSETS) - 1
        chosen = []
        while uncovered_mask:
            best_idx = -1
            best_gain = 0.0
            for i in range(N_BLOCKS):
                gain = 0.0
                for bit in block_bit_indices[i]:
                    if (uncovered_mask >> bit) & 1:
                        gain += weights[bit]
                        if gain <= best_gain:
                            pass
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx < 0 or best_gain == 0.0:
                break
            chosen.append(best_idx)
            uncovered_mask &= ~all_masks[best_idx]
        return chosen

    # 1. Weighted greedy initialization
    current_blocks = weighted_greedy()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # Build counts (list of ints, fast)
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def add_block(idx):
        current_blocks.append(idx)
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def remove_block(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            counts[bit] -= 1
        current_blocks.pop(pos)

    def is_redundant(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True

    def find_redundant():
        for pos in range(len(current_blocks)):
            if is_redundant(pos):
                return pos
        return -1

    # 2. Simulated Annealing
    # Energy = number of blocks (minimize)
    # Moves: swap a random block with a candidate that improves or maintains coverage
    T_init = 10.0
    T_final = 0.1
    cooling_rate = 0.995
    max_steps = 5000

    temperature = T_init
    step = 0
    best_energy = best_len
    current_energy = len(current_blocks)

    while time.time() < deadline and step < max_steps:
        # Phase A: prune all redundant blocks first
        progress = True
        while progress:
            progress = False
            pos = find_redundant()
            if pos >= 0:
                remove_block(pos)
                progress = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                current_energy = len(current_blocks)
                continue

        if len(current_blocks) == 0:
            break

        # Phase B: Simulated Annealing move
        # Pick a random block to potentially swap
        i = random.randint(0, len(current_blocks) - 1)
        idx_i = current_blocks[i]
        
        # Find subsets that would become uncovered if we remove this block
        uncovered_if_removed = []
        for bit in block_bit_indices[idx_i]:
            if counts[bit] == 1:
                uncovered_if_removed.append(bit)

        if not uncovered_if_removed:
            # Block is redundant - remove it (improvement)
            remove_block(i)
            current_energy = len(current_blocks)
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = current_blocks.copy()
            step += 1
            temperature *= cooling_rate
            continue

        # Need to find a replacement block that covers all uncovered_if_removed
        # Intersect candidate sets
        if len(uncovered_if_removed) > 4:
            # Too many constraints, try adding a random block as perturbation
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                add_block(c)
                current_energy = len(current_blocks)
                step += 1
                temperature *= cooling_rate
            continue

        # Find candidate blocks covering all uncovered bits
        cands = None
        for b in uncovered_if_removed:
            lst = blocks_containing_subset[b]
            if cands is None:
                cands = set(lst)
            else:
                cands.intersection_update(lst)
            if not cands:
                break

        if cands:
            cands.discard(idx_i)
            if cands:
                # Choose a random candidate
                j = random.choice(list(cands))
                if j not in current_blocks:
                    # Perform swap: remove i, add j
                    remove_block(i)
                    add_block(j)
                    current_energy = len(current_blocks)
                    
                    # Accept if better, or with probability exp(-delta/T) if worse
                    if current_energy <= best_energy:
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_blocks = current_blocks.copy()
                    else:
                        # Check if this is an improvement over previous state
                        # For simplicity, always accept swaps (they maintain coverage)
                        pass
                    
                    step += 1
                    temperature *= cooling_rate
                    continue

        # No valid swap found - try perturbation: add a random block
        c = random.randint(0, N_BLOCKS - 1)
        if c not in current_blocks:
            add_block(c)
            current_energy = len(current_blocks)
        
        step += 1
        temperature *= cooling_rate

    # 3. Final prune pass
    final_counts = [0] * N_SUBSETS
    for idx in best_blocks:
        for bit in block_bit_indices[idx]:
            final_counts[bit] += 1
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for pos in range(len(best_blocks)):
            idx = best_blocks[pos]
            redundant = True
            for bit in block_bit_indices[idx]:
                if final_counts[bit] == 1:
                    redundant = False
                    break
            if redundant:
                for bit in block_bit_indices[idx]:
                    final_counts[bit] -= 1
                best_blocks.pop(pos)
                changed = True
                break

    return [list(all_blocks[i]) for i in best_blocks]
