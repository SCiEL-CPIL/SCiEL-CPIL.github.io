"""Randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math


def _precompute_block_tsubsets(block, t):
    """Pre-compute all t-subsets of a block as frozensets."""
    return [frozenset(c) for c in itertools.combinations(block, t)]


def _greedy_cover_fast(v, k, t, rng):
    """Fast greedy construction using bitmask-based coverage tracking."""
    # Enumerate all t-subsets and assign integer IDs
    all_tsubs = []
    tsub_to_id = {}
    idx = 0
    for comb in itertools.combinations(range(v), t):
        tsub_to_id[frozenset(comb)] = idx
        all_tsubs.append(frozenset(comb))
        idx += 1
    total_t = len(all_tsubs)

    # Bitmask for coverage: bit i set means t-subset i is covered
    # For v=24, t=3: C(24,3) = 2024 bits -> ~26 uint64 words
    num_words = (total_t + 63) // 64
    covered = [0] * num_words

    def is_covered(tid):
        return bool(covered[tid >> 6] & (1 << (tid & 63)))

    def mark_covered(tid):
        covered[tid >> 6] |= (1 << (tid & 63))

    blocks = []
    block_set = set()

    while True:
        # Find an uncovered t-subset
        unc_found = False
        for w in range(num_words):
            inv = ~covered[w] & ((1 << 64) - 1)
            if inv:
                # Find first set bit
                bit = (inv & (-inv)).bit_length() - 1
                tid = w * 64 + bit
                if tid < total_t:
                    seed = all_tsubs[tid]
                    unc_found = True
                    break
        if not unc_found:
            break

        # Build a block greedily starting from seed
        block = list(seed)
        rng.shuffle(block)

        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)

            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    tid = tsub_to_id.get(fs)
                    if tid is not None and not is_covered(tid):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for fs in _precompute_block_tsubsets(block, t):
                tid = tsub_to_id[fs]
                mark_covered(tid)

    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 118.0

    # Pre-compute all t-subsets for quick lookup
    all_tsubsets_list = [frozenset(c) for c in itertools.combinations(range(v), t)]
    total_t = len(all_tsubsets_list)

    # Build initial valid covering using fast bitmask greedy
    blocks = _greedy_cover_fast(v, k, t, rng)
    
    # Pre-compute t-subsets for each block
    block_tsubs = [_precompute_block_tsubsets(b, t) for b in blocks]
    
    def compute_counts(blk_list, blk_tsubs_list):
        counts = {}
        for tsubs in blk_tsubs_list:
            for fs in tsubs:
                counts[fs] = counts.get(fs, 0) + 1
        return counts

    counts = compute_counts(blocks, block_tsubs)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    uncovered = total_t - len(counts)
    
    def apply_block_change(tsubs, sign):
        """Apply adding/removing a block's t-subsets to counts. Returns delta_uncovered."""
        delta_uncovered = 0
        for fs in tsubs:
            old_c = counts.get(fs, 0)
            new_c = old_c + sign
            if new_c == 0:
                if fs in counts:
                    del counts[fs]
                if old_c == 1:
                    delta_uncovered += 1
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
                counts[fs] = new_c
        return delta_uncovered

    # Simulated annealing parameters
    T0 = 15.0
    T_min = 0.005
    cooling_rate = 0.9992
    T = T0
    
    no_improve_iter = 0
    max_no_improve = 400
    step = 0
    
    while time.time() - start_time < time_limit:
        step += 1
        
        # 1. Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            tsubs_i = block_tsubs[i]
            redundant = True
            for fs in tsubs_i:
                if counts.get(fs, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(tsubs_i, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    no_improve_iter = 0
                    break
                else:
                    apply_block_change(tsubs_i, 1)
                    uncovered -= delta_u
        
        if removed_any:
            continue
        
        # 2. Try swaps with simulated annealing acceptance
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_tsubs = block_tsubs[i]
            
            critical = []
            for fs in old_tsubs:
                if counts.get(fs, 0) == 1:
                    critical.append(fs)
            
            if not critical:
                # Block is fully redundant in coverage sense; try random swap
                new_block_set = set(rng.sample(range(v), k))
            else:
                points_needed = set()
                for fs in critical:
                    points_needed.update(fs)
                
                if len(points_needed) > k:
                    # Can't cover all critical points in one block; pick random subset
                    points_needed = set(rng.sample(list(points_needed), min(k, len(points_needed))))
                
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        c = counts.get(fs, 0)
                        if c == 0:
                            gain += 10
                        elif c == 1:
                            gain += 1
                    new_block_set.add(x)
            
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                if new_block != old_block:
                    new_tsubs = _precompute_block_tsubsets(list(new_block), t)
                    
                    delta_u_remove = apply_block_change(old_tsubs, -1)
                    delta_u_add = apply_block_change(new_tsubs, 1)
                    
                    new_uncovered = uncovered + delta_u_remove + delta_u_add
                    
                    # Energy: uncovered count (lower is better)
                    old_energy = uncovered
                    new_energy = new_uncovered
                    delta_e = new_energy - old_energy
                    
                    accept = False
                    if delta_e <= 0:
                        accept = True
                    elif T > T_min:
                        if delta_e < 80:  # only accept small increases
                            prob = math.exp(-delta_e / T)
                            accept = (rng.random() < prob)
                    
                    if accept:
                        blocks[i] = new_block
                        block_tsubs[i] = new_tsubs
                        uncovered = new_uncovered
                        no_improve_iter = 0
                    else:
                        apply_block_change(new_tsubs, -1)
                        apply_block_change(old_tsubs, 1)
                        uncovered = old_energy
        
        # 3. Perturbation if stuck
        if no_improve_iter > max_no_improve:
            if len(blocks) > 1:
                old_blocks = list(blocks)
                old_block_tsubs = [list(ts) for ts in block_tsubs]
                old_counts = dict(counts)
                old_uncovered = uncovered
                
                i = rng.randrange(len(blocks))
                b = blocks[i]
                tsubs_i = block_tsubs[i]
                
                delta_u = apply_block_change(tsubs_i, -1)
                uncovered += delta_u
                blocks.pop(i)
                block_tsubs.pop(i)
                
                unc_list = []
                for c in all_tsubsets_list:
                    if counts.get(c, 0) == 0:
                        unc_list.append(c)
                
                while unc_list:
                    seed = rng.choice(unc_list)
                    new_b = list(seed)
                    rng.shuffle(new_b)
                    
                    while len(new_b) < k:
                        best_x = None
                        best_gain = -1
                        cands = [x for x in range(v) if x not in new_b]
                        rng.shuffle(cands)
                        for x in cands:
                            gain = 0
                            for rest in itertools.combinations(new_b, t-1):
                                fs = frozenset(list(rest) + [x])
                                if counts.get(fs, 0) == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_b.append(best_x)
                    
                    new_b.sort()
                    new_b_tuple = tuple(new_b)
                    
                    if new_b_tuple not in set(blocks):
                        new_tsubs = _precompute_block_tsubsets(new_b, t)
                        blocks.append(new_b_tuple)
                        block_tsubs.append(new_tsubs)
                        delta_u_add = apply_block_change(new_tsubs, 1)
                        uncovered += delta_u_add
                        
                        unc_list = []
                        for c in all_tsubsets_list:
                            if counts.get(c, 0) == 0:
                                unc_list.append(c)
                    
                if uncovered == 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve_iter = 0
                else:
                    blocks = old_blocks
                    block_tsubs = old_block_tsubs
                    counts = old_counts
                    uncovered = old_uncovered
                    no_improve_iter += 1
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1
        
        # Cool down
        T *= cooling_rate

    # Final polish pass: repeatedly try all removals and targeted swaps
    if time.time() - start_time < time_limit - 2.0:
        polish_time = time_limit - (time.time() - start_time)
        polish_deadline = time.time() + max(0.0, polish_time - 1.0)
        
        # Use current best state
        if uncovered > 0:
            blocks = list(best_blocks)
            block_tsubs = [_precompute_block_tsubsets(b, t) for b in blocks]
            counts = compute_counts(blocks, block_tsubs)
            uncovered = total_t - len(counts)
        
        while time.time() < polish_deadline:
            improved = False
            
            # Try all removals
            for i in range(len(blocks) - 1, -1, -1):
                tsubs_i = block_tsubs[i]
                redundant = True
                for fs in tsubs_i:
                    if counts.get(fs, 0) <= 1:
                        redundant = False
                        break
                if redundant:
                    delta_u = apply_block_change(tsubs_i, -1)
                    uncovered += delta_u
                    if uncovered == 0:
                        blocks.pop(i)
                        block_tsubs.pop(i)
                        improved = True
                    else:
                        apply_block_change(tsubs_i, 1)
                        uncovered -= delta_u
            
            if not improved:
                # Try targeted swaps: for each block, try swapping with a few random alternatives
                for i in range(len(blocks)):
                    if time.time() >= polish_deadline:
                        break
                    old_block = blocks[i]
                    old_tsubs = block_tsubs[i]
                    
                    critical = []
                    for fs in old_tsubs:
                        if counts.get(fs, 0) == 1:
                            critical.append(fs)
                    
                    if not critical:
                        continue
                    
                    points_needed = set()
                    for fs in critical:
                        points_needed.update(fs)
                    
                    if len(points_needed) > k:
                        continue
                    
                    new_block_set = set(points_needed)
                    cands = [x for x in range(v) if x not in new_block_set]
                    rng.shuffle(cands)
                    
                    # Try a few candidates
                    for x in cands[:5]:
                        if len(new_block_set) >= k:
                            break
                        new_block_set.add(x)
                    
                    if len(new_block_set) == k:
                        new_block = tuple(sorted(new_block_set))
                        if new_block != old_block:
                            new_tsubs = _precompute_block_tsubsets(list(new_block), t)
                            delta_u_remove = apply_block_change(old_tsubs, -1)
                            delta_u_add = apply_block_change(new_tsubs, 1)
                            new_uncovered = uncovered + delta_u_remove + delta_u_add
                            if new_uncovered <= uncovered:
                                blocks[i] = new_block
                                block_tsubs[i] = new_tsubs
                                uncovered = new_uncovered
                                improved = True
                            else:
                                apply_block_change(new_tsubs, -1)
                                apply_block_change(old_tsubs, 1)
            
            if not improved:
                break
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
