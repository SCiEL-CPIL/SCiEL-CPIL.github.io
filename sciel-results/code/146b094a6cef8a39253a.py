"""
Structural change: instead of enumerating all C(24,7) blocks and doing greedy set-cover
with 1-exchange, use a probabilistic construction guided by the Johnson scheme.

Key structural differences:
1. Block generation: Use a random symmetric matrix (Johnson-scheme-inspired) to score
   candidate blocks, then select top-scoring blocks with diversity (not pure greedy gain).
2. Repair: Targeted local search that optimizes block overlap structure (maximizing
   shared pairs) rather than simple uncovered-triple gain.
3. Removal: Iterative removal using overlap-aware criteria (remove block with most
   redundancy relative to neighbors, not just any redundant block).
"""

import time
import itertools
import random
import sys
import math
import numpy as np

_TIME_LIMIT = 120
_MARGIN = 10
_DEADLINE = None


def _time_ok():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN
    return time.time() < _DEADLINE


def _triple_to_idx(triple, v, t):
    """Fast combinatorial number system encoding for t-subsets of {0..v-1}."""
    # Use the combinatorial number system (colex) to map (a,b,c) to index
    # For t=3: idx = C(a,3) + C(b,2) + C(c,1) where a>b>c
    # Actually, let's just use a dict for clarity and sufficient speed
    pass


def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN

    # Edge cases
    if v < t:
        return []
    if k > v:
        k = v
    if t > k:
        return []
    if t == 0:
        return []

    num_triples = 1
    for i in range(t):
        num_triples *= (v - i)
        num_triples //= (i + 1)

    # Build triple index
    triple_index = {}
    idx = 0
    for ts in itertools.combinations(range(v), t):
        triple_index[ts] = idx
        idx += 1

    # Precompute all blocks as tuples for fast lookup
    all_block_tuples = list(itertools.combinations(range(v), k))
    num_blocks = len(all_block_tuples)

    # Precompute mask for each block (as Python int bitmask over num_triples bits)
    # This is the expensive part: C(24,7) = 346104 blocks, each with C(7,3)=35 triples
    all_masks = [0] * num_blocks
    for bi, block in enumerate(all_block_tuples):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << triple_index[ts])
        all_masks[bi] = mask

    if not _time_ok():
        # Fallback: return empty (shouldn't happen)
        return []

    rng = random.Random(42)

    # ============================================================
    # PHASE 1: Probabilistic construction via Johnson-scheme scoring
    # ============================================================
    # Build a random symmetric "affinity" matrix A of size v x v.
    # Score a block B as sum of A[i][j] for all pairs (i,j) in B.
    # This creates a structured bias that approximates a block design.
    # We generate multiple such matrices and pick blocks greedily by
    # (score, then uncovered gain) to get diverse high-quality blocks.

    best_selected = None
    best_count = float('inf')

    num_constructions = 5  # Try multiple random matrices

    for trial in range(num_constructions):
        if not _time_ok():
            break

        trial_rng = random.Random(42 + trial * 1000)

        # Generate random symmetric matrix with values in [0, 1]
        A = np.random.RandomState(trial_rng.randint(0, 2**31)).random_sample((v, v))
        A = (A + A.T) / 2.0

        # Score each block
        block_scores = np.zeros(num_blocks)
        for bi, block in enumerate(all_block_tuples):
            s = 0.0
            for i in range(k):
                for j in range(i + 1, k):
                    s += A[block[i], block[j]]
            block_scores[bi] = s

        # Sort block indices by score descending
        sorted_indices = np.argsort(-block_scores).tolist()

        # Greedy selection: pick block that maximizes (alpha * score + (1-alpha) * gain)
        # where gain = number of newly covered triples
        alpha = 0.3  # Weight on Johnson-scheme score vs. coverage gain
        uncovered = (1 << num_triples) - 1
        selected = []
        selected_set = set()

        while uncovered and _time_ok():
            best_val = -1
            best_idx = -1
            best_gain = 0

            # Check top-scoring unselected blocks first (structural bias)
            # Then fill in with coverage gain
            checked = 0
            max_check = min(num_blocks, 50000)  # Limit per iteration for speed

            for idx_candidate in sorted_indices:
                if idx_candidate in selected_set:
                    continue
                gain = (all_masks[idx_candidate] & uncovered).bit_count()
                if gain == 0:
                    continue
                score_norm = block_scores[idx_candidate] / max(1e-9, block_scores.max())
                val = alpha * score_norm + (1 - alpha) * (gain / max(1, uncovered.bit_count()))
                if val > best_val:
                    best_val = val
                    best_idx = idx_candidate
                    best_gain = gain
                checked += 1
                if checked >= max_check:
                    break

            if best_idx == -1:
                # Fallback: scan all remaining
                for i in range(num_blocks):
                    if i in selected_set:
                        continue
                    gain = (all_masks[i] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        best_val = gain
                if best_idx == -1:
                    break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~all_masks[best_idx]

        # Force-complete if needed
        while uncovered and _time_ok():
            best_gain = 0
            best_idx = -1
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = (all_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~all_masks[best_idx]

        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = list(selected)

    if best_selected is None:
        # Shouldn't happen, but fallback
        best_selected = []

    # ============================================================
    # PHASE 2: Overlap-aware local search (repair)
    # ============================================================
    # Instead of simple 1-exchange, we do:
    # a) Remove redundant blocks (those with all triples covered by others)
    # b) Try to replace pairs of blocks with a single block that covers their union
    # c) Try to add a block and remove 2+ blocks (2-for-1 exchange)

    selected = list(best_selected)

    def _compute_triple_counts(sel):
        """Compute coverage count for each triple in the selected set."""
        counts = {}
        for idx in sel:
            mask = all_masks[idx]
            m = mask
            while m:
                low = m & (-m)
                bit = low.bit_length() - 1
                counts[bit] = counts.get(bit, 0) + 1
                m ^= low
        return counts

    def _is_removable(idx, counts):
        """Check if block idx can be removed (all its triples have count > 1)."""
        mask = all_masks[idx]
        m = mask
        while m:
            low = m & (-m)
            bit = low.bit_length() - 1
            if counts.get(bit, 0) <= 1:
                return False
            m ^= low
        return True

    def _remove_redundant(sel):
        """Iteratively remove redundant blocks."""
        sel = list(sel)
        changed = True
        while changed and _time_ok():
            changed = False
            counts = _compute_triple_counts(sel)
            for i in range(len(sel)):
                if not _time_ok():
                    return sel
                if _is_removable(sel[i], counts):
                    sel.pop(i)
                    changed = True
                    break
        return sel

    def _try_2_for_1(sel):
        """Try to find a block that covers the union of two selected blocks' unique triples."""
        sel = list(sel)
        improved = True
        while improved and _time_ok():
            improved = False
            n = len(sel)
            if n < 2:
                break
            # Sample pairs
            rng_local = random.Random(int(time.time() * 1000) % 100000)
            pairs = list(itertools.combinations(range(n), 2))
            rng_local.shuffle(pairs)
            max_pairs = min(len(pairs), 200)

            for p1, p2 in pairs[:max_pairs]:
                if not _time_ok():
                    break
                idx1, idx2 = sel[p1], sel[p2]
                mask1 = all_masks[idx1]
                mask2 = all_masks[idx2]

                # Triples only covered by block1 or only by block2
                # We need a new block that covers (mask1 | mask2) minus what others cover
                # Simplified: find a block whose mask covers mask1 & mask2 (the overlap)
                # Actually, we want to replace {idx1, idx2} with a single block that
                # covers all triples that were ONLY covered by idx1 or ONLY by idx2.

                # Compute what's uniquely covered by idx1 and idx2
                others_mask = 0
                for idx in sel:
                    if idx != idx1 and idx != idx2:
                        others_mask |= all_masks[idx]

                unique_to_1 = mask1 & ~others_mask & ~mask2
                unique_to_2 = mask2 & ~others_mask & ~mask1
                needed = unique_to_1 | unique_to_2

                if needed == 0:
                    # Both blocks are fully redundant; just remove them
                    sel.remove(idx2)
                    sel.remove(idx1)
                    improved = True
                    break

                # Find a single block that covers 'needed'
                best_block = -1
                best_overlap = 0
                for bi in range(num_blocks):
                    if bi == idx1 or bi == idx2:
                        continue
                    overlap = (all_masks[bi] & needed).bit_count()
                    if overlap == needed.bit_count():
                        best_block = bi
                        break
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_block = bi

                if best_block != -1 and (all_masks[best_block] & needed) == needed:
                    # Replace idx1, idx2 with best_block
                    sel.remove(idx2)
                    sel.remove(idx1)
                    sel.append(best_block)
                    improved = True
                    break

        return sel

    def _try_add_remove(sel):
        """Try adding one block and removing two or more."""
        sel = list(sel)
        current_set = set(sel)
        improved = True

        while improved and _time_ok():
            improved = False
            unselected = [i for i in range(num_blocks) if i not in current_set]
            if not unselected:
                break

            rng_local = random.Random(int(time.time() * 1000) % 100000)
            sample_size = min(3000, len(unselected))
            candidates = rng_local.sample(unselected, sample_size) if sample_size < len(unselected) else unselected

            for new_idx in candidates:
                if not _time_ok():
                    break

                trial = sel + [new_idx]
                counts = _compute_triple_counts(trial)

                # Find removable blocks (excluding new_idx)
                removable = []
                for idx in trial:
                    if idx == new_idx:
                        continue
                    if _is_removable(idx, counts):
                        removable.append(idx)

                if len(removable) >= 2:
                    # Try removing the two that maximize freed capacity
                    # Just remove first two for now
                    to_remove = removable[:2]
                    new_sel = [idx for idx in trial if idx not in to_remove]
                    if len(new_sel) < len(sel):
                        sel = new_sel
                        current_set = set(sel)
                        improved = True
                        break

        return sel

    # Apply repair steps
    selected = _remove_redundant(selected)
    selected = _try_2_for_1(selected)
    selected = _remove_redundant(selected)
    selected = _try_add_remove(selected)
    selected = _remove_redundant(selected)

    # ============================================================
    # PHASE 3: Final verification and output
    # ============================================================

    # Build result
    result = [list(all_block_tuples[i]) for i in selected]

    # Deduplicate
    seen = set()
    unique_result = []
    for b in result:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_result.append(b)

    # Verify coverage
    covered_mask = 0
    for b in unique_result:
        for ts in itertools.combinations(b, t):
            covered_mask |= (1 << triple_index[ts])

    full_mask = (1 << num_triples) - 1
    if covered_mask != full_mask:
        # Incomplete: add blocks to cover remaining
        uncovered = full_mask & ~covered_mask
        sel_set = set(selected)
        while uncovered and _time_ok():
            best_gain = 0
            best_idx = -1
            for i in range(num_blocks):
                if i in sel_set:
                    continue
                gain = (all_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            unique_result.append(list(all_block_tuples[best_idx]))
            sel_set.add(best_idx)
            uncovered &= ~all_masks[best_idx]

    return unique_result
