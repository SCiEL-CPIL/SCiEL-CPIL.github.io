"""Simulated-annealing local search for C(24,7,3) covering design."""
import itertools
import time
import random
import math
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 145

    # Precompute all t-subsets and index
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {ts: i for i, ts in enumerate(all_t)}

    # For each point, list of t-subsets containing it
    point_tsubs = [[] for _ in range(v)]
    for ts in all_t:
        for p in ts:
            point_tsubs[p].append(t_index[ts])

    # For each pair of points, list of t-subsets containing both (for t=3: triples {a,b,c})
    # For t=3: pair (a,b) -> indices of triples containing a and b
    pair_tsubs = {}
    for i in range(v):
        for j in range(i + 1, v):
            lst = []
            for c in range(v):
                if c == i or c == j:
                    continue
                ts = tuple(sorted((i, j, c)))
                lst.append(t_index[ts])
            pair_tsubs[(i, j)] = lst

    def fast_greedy(rng, time_limit):
        """Fast greedy: repeatedly pick block covering most uncovered t-subsets."""
        uncovered = [1] * n_t
        uncovered_count = n_t
        blocks = []
        # Precompute for each k-subset its coverage? Too many (C(24,7)=346104).
        # Instead, greedy: start from an uncovered t-subset, grow block.
        
        while uncovered_count > 0 and time.time() < time_limit:
            # Find an uncovered t-subset
            # Use a list of uncovered indices for speed
            # For simplicity, scan (n_t=2024 is small)
            idx = -1
            for i in range(n_t):
                if uncovered[i]:
                    idx = i
                    break
            if idx == -1:
                break
            
            seed = set(all_t[idx])
            block = list(seed)
            block_set = set(block)
            
            # Grow to size k
            for _ in range(k - t):
                if time.time() > time_limit:
                    break
                best_gain = -1
                best_x = None
                for x in range(v):
                    if x in block_set:
                        continue
                    # Gain: number of uncovered t-subsets containing x and (t-1) from block
                    gain = 0
                    # For t=3: t-subsets containing x and 2 from block
                    for a, b in itertools.combinations(block, t - 1):
                        ts = tuple(sorted((a, b, x)))
                        ti = t_index[ts]
                        if uncovered[ti]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None and rng.random() < 0.3:
                        best_x = x
                if best_x is None or best_gain <= 0:
                    # Pad randomly
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block_set.add(rng.choice(remaining))
                        block.append(rng.choice(remaining))
                    break
                block_set.add(best_x)
                block.append(best_x)
            
            if len(block) != k:
                remaining = [x for x in range(v) if x not in block_set]
                for x in remaining:
                    if len(block) >= k:
                        break
                    block.append(x)
                    block_set.add(x)
            
            block = sorted(block)
            blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                ti = t_index[ts]
                if uncovered[ti]:
                    uncovered[ti] = 0
                    uncovered_count -= 1
        
        return blocks

    def count_uncovered(blocks):
        """Return number of uncovered t-subsets and the list."""
        covered = [0] * n_t
        for block in blocks:
            block_set = set(block)
            for ts in itertools.combinations(block, t):
                ti = t_index[ts]
                covered[ti] = 1
        uncovered = [i for i in range(n_t) if not covered[i]]
        return len(uncovered), uncovered

    def simulated_annealing(blocks, time_limit, rng):
        """SA on block multiset: 1-swap and 2-opt moves."""
        # Ensure blocks are lists of sorted ints
        blocks = [list(b) for b in blocks]
        n_blocks = len(blocks)
        
        # Precompute coverage counts: for each t-subset, how many blocks cover it
        cover_count = [0] * n_t
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[t_index[ts]] += 1
        
        uncovered_count = sum(1 for c in cover_count if c == 0)
        
        best_blocks = [list(b) for b in blocks]
        best_uncovered = uncovered_count
        best_count = n_blocks
        
        # SA parameters
        T_start = 2.0
        T_end = 0.01
        n_iter = max(1, int((time_limit - (time.time() - (deadline - time_limit))) * 50))
        # Actually, let's just iterate until time limit
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max(1, n_iter))
        
        while time.time() < time_limit:
            # Pick a move
            move_type = rng.random()
            if move_type < 0.5:
                # 1-swap: replace one block with a new random block
                if n_blocks == 0:
                    break
                i = rng.randrange(n_blocks)
                old_block = blocks[i]
                old_set = set(old_block)
                
                # Generate new block: biased towards covering uncovered t-subsets
                # Pick a random uncovered t-subset, build block around it
                unc_list = [i2 for i2 in range(n_t) if cover_count[i2] == 0]
                if unc_list:
                    seed_ts = all_t[rng.choice(unc_list)]
                    new_block_set = set(seed_ts)
                else:
                    # All covered; random block
                    new_block_set = set(rng.sample(range(v), k))
                
                while len(new_block_set) < k:
                    # Add point that maximizes coverage of currently-uncovered t-subsets
                    best_gain = -1
                    best_x = None
                    for x in range(v):
                        if x in new_block_set:
                            continue
                        gain = 0
                        for ts in itertools.combinations(sorted(new_block_set), t - 1):
                            ts_new = tuple(sorted(ts + (x,)))
                            if ts_new in t_index:
                                ti = t_index[ts_new]
                                if cover_count[ti] == 0:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                        elif gain == best_gain and best_x is not None and rng.random() < 0.3:
                            best_x = x
                    if best_x is None:
                        remaining = [x for x in range(v) if x not in new_block_set]
                        if remaining:
                            best_x = rng.choice(remaining)
                        else:
                            break
                    new_block_set.add(best_x)
                
                new_block = sorted(new_block_set)
                if new_block == old_block:
                    continue
                
                # Compute delta: change in uncovered count
                # Remove old_block: for each t-subset in old_block, decrement cover_count
                delta_unc = 0
                affected = []
                for ts in itertools.combinations(old_block, t):
                    ti = t_index[ts]
                    cover_count[ti] -= 1
                    affected.append(ti)
                    if cover_count[ti] == 0:
                        delta_unc += 1
                
                # Add new_block: for each t-subset in new_block, increment cover_count
                for ts in itertools.combinations(new_block, t):
                    ti = t_index[ts]
                    cover_count[ti] += 1
                    if ti in affected:
                        # Already decremented; if it went back to >=1, undo delta
                        if cover_count[ti] >= 1:
                            delta_unc -= 1
                    else:
                        if cover_count[ti] == 1:
                            delta_unc -= 1
                
                uncovered_count += delta_unc
                
                # Accept/reject
                if delta_unc <= 0 or rng.random() < math.exp(-delta_unc / T):
                    blocks[i] = new_block
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best_blocks = [list(b) for b in blocks]
                        best_count = n_blocks
                else:
                    # Revert
                    for ts in itertools.combinations(new_block, t):
                        ti = t_index[ts]
                        cover_count[ti] -= 1
                    for ts in itertools.combinations(old_block, t):
                        ti = t_index[ts]
                        cover_count[ti] += 1
                    uncovered_count -= delta_unc
                
                T *= cooling
                
            elif move_type < 0.8:
                # 2-opt: replace two blocks with two new blocks
                if n_blocks < 2:
                    T *= cooling
                    continue
                i, j = rng.sample(range(n_blocks), 2)
                old_bi = blocks[i]
                old_bj = blocks[j]
                
                # Generate new blocks around uncovered t-subsets
                unc_list = [i2 for i2 in range(n_t) if cover_count[i2] == 0]
                new_bi_set = set()
                new_bj_set = set()
                
                if len(unc_list) >= 2:
                    ts1 = all_t[rng.choice(unc_list)]
                    ts2 = all_t[rng.choice([x for x in unc_list if x != unc_list.index(ts1)] if len(unc_list) > 1 else unc_list)]
                    new_bi_set = set(ts1)
                    new_bj_set = set(ts2)
                elif len(unc_list) == 1:
                    ts1 = all_t[unc_list[0]]
                    new_bi_set = set(ts1)
                    new_bj_set = set(rng.sample(range(v), t))
                else:
                    new_bi_set = set(rng.sample(range(v), t))
                    new_bj_set = set(rng.sample(range(v), t))
                
                # Grow both to size k
                for _ in range(k - t):
                    for target in [new_bi_set, new_bj_set]:
                        if len(target) >= k:
                            continue
                        best_gain = -1
                        best_x = None
                        for x in range(v):
                            if x in target:
                                continue
                            gain = 0
                            for ts in itertools.combinations(sorted(target), t - 1):
                                ts_new = tuple(sorted(ts + (x,)))
                                if ts_new in t_index:
                                    ti = t_index[ts_new]
                                    if cover_count[ti] == 0:
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                            elif gain == best_gain and best_x is not None and rng.random() < 0.3:
                                best_x = x
                        if best_x is None:
                            remaining = [x for x in range(v) if x not in target]
                            if remaining:
                                best_x = rng.choice(remaining)
                            else:
                                break
                        target.add(best_x)
                
                new_bi = sorted(new_bi_set)
                new_bj = sorted(new_bj_set)
                
                if new_bi == old_bi and new_bj == old_bj:
                    T *= cooling
                    continue
                
                # Compute delta
                delta_unc = 0
                # Remove old blocks
                for ts in itertools.combinations(old_bi, t):
                    ti = t_index[ts]
                    cover_count[ti] -= 1
                    if cover_count[ti] == 0:
                        delta_unc += 1
                for ts in itertools.combinations(old_bj, t):
                    ti = t_index[ts]
                    cover_count[ti] -= 1
                    if cover_count[ti] == 0:
                        delta_unc += 1
                
                # Add new blocks
                for ts in itertools.combinations(new_bi, t):
                    ti = t_index[ts]
                    cover_count[ti] += 1
                    if cover_count[ti] == 1:
                        delta_unc -= 1
                for ts in itertools.combinations(new_bj, t):
                    ti = t_index[ts]
                    cover_count[ti] += 1
                    if cover_count[ti] == 1:
                        delta_unc -= 1
                
                uncovered_count += delta_unc
                
                if delta_unc <= 0 or rng.random() < math.exp(-delta_unc / T):
                    blocks[i] = new_bi
                    blocks[j] = new_bj
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best_blocks = [list(b) for b in blocks]
                        best_count = n_blocks
                else:
                    # Revert
                    for ts in itertools.combinations(new_bi, t):
                        ti = t_index[ts]
                        cover_count[ti] -= 1
                    for ts in itertools.combinations(new_bj, t):
                        ti = t_index[ts]
                        cover_count[ti] -= 1
                    for ts in itertools.combinations(old_bi, t):
                        ti = t_index[ts]
                        cover_count[ti] += 1
                    for ts in itertools.combinations(old_bj, t):
                        ti = t_index[ts]
                        cover_count[ti] += 1
                    uncovered_count -= delta_unc
                
                T *= cooling
                
            else:
                # Try to remove a redundant block (if n_blocks > lower bound)
                if n_blocks > 69:
                    i = rng.randrange(n_blocks)
                    old_block = blocks[i]
                    delta_unc = 0
                    for ts in itertools.combinations(old_block, t):
                        ti = t_index[ts]
                        cover_count[ti] -= 1
                        if cover_count[ti] == 0:
                            delta_unc += 1
                    uncovered_count += delta_unc
                    
                    if delta_unc <= 0:
                        # Block was redundant
                        blocks.pop(i)
                        n_blocks -= 1
                        if uncovered_count < best_uncovered:
                            best_uncovered = uncovered_count
                            best_blocks = [list(b) for b in blocks]
                            best_count = n_blocks
                    else:
                        # Revert
                        for ts in itertools.combinations(old_block, t):
                            ti = t_index[ts]
                            cover_count[ti] += 1
                        uncovered_count -= delta_unc
                    
                    T *= cooling
                else:
                    T *= cooling
        
        return best_blocks

    def remove_redundant(blocks, time_limit):
        """Iteratively remove blocks that are not needed."""
        n = len(blocks)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                candidate = blocks[:i] + blocks[i+1:]
                # Check validity
                covered = [0] * n_t
                for block in candidate:
                    for ts in itertools.combinations(block, t):
                        covered[t_index[ts]] = 1
                valid = all(covered)
                if valid:
                    blocks = candidate
                    n -= 1
                    changed = True
                    break
        return blocks

    # Main
    best_result = None
    best_len = float('inf')
    
    # Phase 1: Fast greedy to get initial blocks
    rng = random.Random(42)
    initial = fast_greedy(rng, deadline - 100)
    
    if not initial:
        # Fallback
        rng = random.Random(0)
        initial = fast_greedy(rng, deadline - 10)
    
    # Remove redundant
    initial = remove_redundant(initial, deadline - 90)
    best_result = initial
    best_len = len(initial)
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 20:
        sa_result = simulated_annealing(initial, deadline - 15, rng)
        sa_result = remove_redundant(sa_result, deadline - 5)
        if len(sa_result) < best_len:
            best_len = len(sa_result)
            best_result = sa_result
    
    # Phase 3: Another SA run with different seed if time
    if time.time() < deadline - 10:
        rng2 = random.Random(12345)
        initial2 = fast_greedy(rng2, deadline - 30)
        initial2 = remove_redundant(initial2, deadline - 25)
        sa_result2 = simulated_annealing(initial2, deadline - 8, rng2)
        sa_result2 = remove_redundant(sa_result2, deadline - 2)
        if len(sa_result2) < best_len:
            best_len = len(sa_result2)
            best_result = sa_result2
    
    if best_result is None:
        best_result = initial
    
    return best_result
