"""Covering design C(v,k,t): minimise the number of k-subsets of v points
so that every t-subset lies in at least one block.

Design family: bitset-accelerated greedy seed + simulated annealing.
State = list of blocks.  Coverage of the C(v,t) t-subsets is tracked as a
Python integer bitmask (one bit per t-subset), so whole-block coverage updates
are O(1) integer ops and "is this t-subset in block B" is a precomputed
membership test.  This keeps each SA move cheap so thousands of iterations fit
inside the wall-clock budget.
"""
import itertools
import time
import math
import random


def _combo_code(sub, n):
    code = 0
    for x in sub:
        code |= (1 << (x * n))
    return code


def _encode_t_subsets(v, t):
    n = v - 1
    code_to_idx = {}
    i = 0
    for combo in itertools.combinations(range(v), t):
        code = 0
        for x in combo:
            code |= (1 << (x * n))
        code_to_idx[code] = i
        i += 1
    return code_to_idx, n


def _block_mask(block, code_to_idx, n):
    m = 0
    for sub in itertools.combinations(block, t):
        m |= (1 << code_to_idx[_combo_code(sub, n)])
    return m


def _counts_from_masks(masks, code_to_idx, n, num_subsets):
    counts = [0] * num_subsets
    for m in masks:
        mm = m
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            counts[i] += 1
            mm ^= lb
    return counts


def _or_masks(masks):
    c = 0
    for m in masks:
        c |= m
    return c


def construct(v, k, t):
    deadline = time.time() + 112.0          # safety margin vs 150s kill
    rng = random.Random(12345)

    code_to_idx, n = _encode_t_subsets(v, t)
    num_subsets = len(code_to_idx)
    full_mask = (1 << num_subsets) - 1

    # point_masks[x] = bitset of t-subsets containing x
    point_masks = []
    for x in range(v):
        m = 0
        for sub in itertools.combinations(range(v), t):
            if x in sub:
                m |= (1 << code_to_idx[_combo_code(sub, n)])
        point_masks.append(m)

    # ---- greedy seed ----
    def greedy_add(covered, blocks, block_masks, counts, deadline_guard=True):
        while covered != full_mask:
            if deadline_guard and time.time() > deadline - 45.0:
                return covered, blocks, block_masks, counts
            uncovered = full_mask ^ covered
            best_x, best_g = -1, -1
            for x in range(v):
                g = (point_masks[x] & uncovered).bit_count()
                if g > best_g:
                    best_g, best_x = g, x
            if best_x < 0:
                return covered, blocks, block_masks, counts
            block = [best_x]
            lc = point_masks[best_x]
            for _ in range(k - 1):
                bp, bg = -1, -1
                for y in range(v):
                    if y in block:
                        continue
                    g = (point_masks[y] & uncovered & ~lc).bit_count()
                    if g > bg:
                        bg, bp = g, y
                if bp < 0:
                    break
                block.append(bp)
                lc |= point_masks[bp]
            if len(block) < k:
                return covered, blocks, block_masks, counts
            block.sort()
            if block in blocks:
                continue
            m = _block_mask(block, code_to_idx, n)
            covered |= m
            for sub in itertools.combinations(block, t):
                counts[code_to_idx[_combo_code(sub, n)]] += 1
            blocks.append(block)
            block_masks.append(m)
        return covered, blocks, block_masks, counts

    blocks = []
    block_masks = []
    counts = [0] * num_subsets
    covered = 0
    covered, blocks, block_masks, counts = greedy_add(covered, blocks, block_masks, counts)

    # ---- simulated annealing ----
    cur_cost = len(block_masks) + 3 * (full_mask ^ covered).bit_count()
    best_blocks, best_masks = list(blocks), list(block_masks)
    best_covered, best_cost = covered, cur_cost

    T, T_min, cool = 1.0, 0.02, 0.9995

    def accept(delta):
        if delta <= 0:
            return True
        return rng.random() < min(1.0, math.exp(-delta / T))

    while time.time() < deadline - 2.0 and T > T_min:
        r = rng.random()
        if r < 0.40 and len(block_masks) > 1:
            # REMOVE a block that has no unique t-subset (always valid)
            i = rng.randrange(len(block_masks))
            has_unique = False
            for sub in itertools.combinations(blocks[i], t):
                if counts[code_to_idx[_combo_code(sub, n)]] == 1:
                    has_unique = True
                    break
            if not has_unique:
                m = block_masks[i]
                saved_block, saved_mask = blocks[i], block_masks[i]
                for sub in itertools.combinations(saved_block, t):
                    counts[code_to_idx[_combo_code(sub, n)]] -= 1
                del blocks[i]
                del block_masks[i]
                covered = _or_masks(block_masks)
                new_cost = len(block_masks) + 3 * (full_mask ^ covered).bit_count()
                d = new_cost - cur_cost
                if d <= 0 or accept(d):
                    cur_cost = new_cost
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_blocks, best_masks = list(blocks), list(block_masks)
                        best_covered = covered
                else:
                    # revert
                    blocks.insert(i, saved_block)
                    block_masks.insert(i, saved_mask)
                    for sub in itertools.combinations(saved_block, t):
                        counts[code_to_idx[_combo_code(sub, n)]] += 1
                    covered = _or_masks(block_masks)
        elif r < 0.78:
            # ADD a block covering the rarest uncovered subsets
            uncovered = full_mask ^ covered
            if uncovered:
                best_x, best_g = -1, -1
                for x in range(v):
                    g = (point_masks[x] & uncovered).bit_count()
                    if g > best_g:
                        best_g, best_x = g, x
                if best_x >= 0:
                    block = [best_x]
                    lc = point_masks[best_x]
                    for _ in range(k - 1):
                        bp, bg = -1, -1
                        for y in range(v):
                            if y in block:
                                continue
                            g = (point_masks[y] & uncovered & ~lc).bit_count()
                            if g > bg:
                                bg, bp = g, y
                        if bp < 0:
                            break
                        block.append(bp)
                        lc |= point_masks[bp]
                    if len(block) == k:
                        block.sort()
                        if block not in blocks:
                            m = _block_mask(block, code_to_idx, n)
                            covered |= m
                            for sub in itertools.combinations(block, t):
                                counts[code_to_idx[_combo_code(sub, n)]] += 1
                            blocks.append(block)
                            block_masks.append(m)
                            new_cost = len(block_masks) + 3 * (full_mask ^ covered).bit_count()
                            d = new_cost - cur_cost
                            if d <= 0 or accept(d):
                                cur_cost = new_cost
                                if cur_cost < best_cost:
                                    best_cost = cur_cost
                                    best_blocks, best_masks = list(blocks), list(block_masks)
                                    best_covered = covered
                            else:
                                # revert
                                block_masks.pop()
                                blocks.pop()
                                for sub in itertools.combinations(block, t):
                                    counts[code_to_idx[_combo_code(sub, n)]] -= 1
                                covered = _or_masks(block_masks)
        T *= cool

    # Return the best valid covering found.
    if best_covered == full_mask:
        return best_blocks
    if covered == full_mask:
        return blocks
    return blocks
