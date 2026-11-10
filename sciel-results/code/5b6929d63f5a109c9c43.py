# EVOLVE-BLOCK-START
import itertools
import time
import random
from math import comb
from collections import defaultdict

def _greedy_cover(v, k, t, rng, deadline):
    """
    Greedy set cover with hardness-based seed selection.
    Selects the uncovered t-subset that shares the fewest points with other uncovered t-subsets.
    """
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    uncovered = [True] * n_t
    n_uncovered = n_t
    
    # Precompute point to t-subset indices for hardness calculation
    point_to_ts = defaultdict(list)
    for i, ts in enumerate(all_t):
        for p in ts:
            point_to_ts[p].append(i)
            
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
            
        # 1. Select seed t-subset
        # We want to find an uncovered t-subset with minimal "interaction" with other uncovered ones.
        # Interaction = number of other uncovered t-subsets sharing at least one point.
        best_seed = None
        best_score = float('inf')
        
        # Sample candidates to keep it fast
        candidates = []
        if n_uncovered < 200:
            candidates = [i for i in range(n_t) if uncovered[i]]
        else:
            attempts = 0
            while len(candidates) < 100 and attempts < 500:
                idx = rng.randint(0, n_t - 1)
                if uncovered[idx] and idx not in candidates:
                    candidates.append(idx)
                attempts += 1
            if not candidates:
                for i in range(n_t):
                    if uncovered[i]:
                        candidates.append(i)
                        break

        for i in candidates:
            ts = all_t[i]
            neighbors = set()
            for p in ts:
                for j in point_to_ts[p]:
                    if j != i and uncovered[j]:
                        neighbors.add(j)
            
            score = len(neighbors)
            if score < best_score:
                best_score = score
                best_seed = i
            elif score == best_score and rng.random() < 0.2:
                best_seed = i
                
        if best_seed is None:
            break
            
        block = list(all_t[best_seed])
        
        # 2. Grow block to size k
        while len(block) < k:
            if time.time() > deadline:
                break
            in_block = set(block)
            best_x = None
            best_gain = -1
            best_tie_list = []
            
            rests = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in in_block:
                    continue
                gain = 0
                for rest in rests:
                    ts_new = tuple(sorted(rest + (x,)))
                    idx = t_index.get(ts_new)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_tie_list = [x]
                elif gain == best_gain and gain > 0:
                    best_tie_list.append(x)
            
            if best_x is None or best_gain <= 0:
                unused = [x for x in range(v) if x not in in_block]
                if not unused:
                    break
                best_x = rng.choice(unused)
            else:
                if len(best_tie_list) > 1:
                    best_x = rng.choice(best_tie_list)
            
            block.append(best_x)
            
        block.sort()
        blocks.append(block)
        
        # 3. Mark covered t-subsets
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1
                
    return blocks

def _is_valid(blocks, v, k, t):
    total = comb(v, t)
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    return len(covered) == total

def _local_search(blocks, v, k, t, rng, deadline):
    """
    Local search to reduce block count.
    1. Remove redundant blocks.
    2. Replace a block with another that covers its unique t-subsets and maximizes other coverage.
    """
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    blocks = [sorted(list(b)) for b in blocks]
    
    # Coverage count per t-subset
    cov = [0] * n_t
    for b in blocks:
        for ts in itertools.combinations(b, t):
            cov[t_index[ts]] += 1
            
    def recompute_cov():
        nonlocal cov
        cov = [0] * n_t
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov[t_index[ts]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        order = list(range(n))
        rng.shuffle(order)
        
        for bi in order:
            if time.time() > deadline:
                return blocks
            if bi >= len(blocks):
                continue
                
            b = blocks[bi]
            # Identify t-subsets uniquely covered by b
            unique_idxs = []
            for ts in itertools.combinations(b, t):
                idx = t_index[ts]
                if cov[idx] == 1:
                    unique_idxs.append(idx)
                    
            if not unique_idxs:
                # Block is fully redundant
                blocks.pop(bi)
                recompute_cov()
                improved = True
                break
                
            # Candidate points: union of points in unique t-subsets
            cand_set = set()
            for idx in unique_idxs:
                cand_set.update(all_t[idx])
            
            # If the union is too large, we can't fit it in a k-block.
            # However, we might be able to replace it if the unique sets are few.
            # Actually, the new block MUST contain the union of unique t-subsets?
            # No, the new block must COVER the unique t-subsets.
            # A t-subset is covered if ALL its points are in the block.
            # So the new block MUST contain the union of all unique t-subsets.
            # If len(union) > k, we cannot replace b with a single block.
            if len(cand_set) > k:
                continue
                
            cand_list = list(cand_set)
            
            # We must use all points in cand_set.
            # So the new block starts with cand_set.
            newblock = list(cand_set)
            
            # Check if we can fill the rest to maximize coverage
            # We have k - len(newblock) slots left.
            
            # Calculate current coverage of newblock (which contains all unique sets)
            # We want to pick remaining points to maximize the number of t-subsets
            # that are currently covered by b (to maintain similarity) or to cover
            # t-subsets that are "hard" (low coverage).
            
            # Let's just pick points that maximize the number of t-subsets in the new block
            # that have cov[idx] == 1 (unique to b) - but we already covered those.
            # So we should pick points that cover t-subsets that are NOT unique to b,
            # to ensure we don't lose coverage if b is removed?
            # Wait, if we replace b with newblock:
            # - Unique sets of b are covered by newblock (by construction).
            # - Non-unique sets of b are covered by other blocks.
            # - So we are safe.
            # - To help future reductions, we should pick points that cover t-subsets
            #   that are covered by few other blocks.
            
            # Let's greedily fill the remaining slots.
            while len(newblock) < k:
                in_block = set(newblock)
                best_x = None
                best_gain = -1
                best_tie_list = []
                
                rests = list(itertools.combinations(newblock, t - 1))
                
                for x in range(v):
                    if x in in_block:
                        continue
                    gain = 0
                    for rest in rests:
                        ts_new = tuple(sorted(rest + (x,)))
                        idx = t_index.get(ts_new)
                        if idx is not None:
                            # Prefer t-subsets with low coverage count
                            # This helps create "tighter" coverings
                            if cov[idx] <= 2:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_tie_list = [x]
                    elif gain == best_gain and gain > 0:
                        best_tie_list.append(x)
                        
                if best_x is None or best_gain <= 0:
                    # No good gain, pick random
                    unused = [x for x in range(v) if x not in in_block]
                    if not unused:
                        break
                    best_x = rng.choice(unused)
                else:
                    if len(best_tie_list) > 1:
                        best_x = rng.choice(best_tie_list)
                        
                newblock.append(best_x)
                
            newblock.sort()
            
            # Check if newblock is different from b and not duplicate
            if newblock != b and newblock not in blocks:
                blocks.pop(bi)
                blocks.append(newblock)
                recompute_cov()
                improved = True
                break
                
    return blocks

def construct(v, k, t):
    start = time.time()
    deadline = start + 112.0  # Safety margin
    best_blocks = None
    best_len = float('inf')
    
    # Iteration 1: Baseline
    rng = random.Random(12345)
    b = _greedy_cover(v, k, t, rng, deadline)
    if _is_valid(b, v, k, t):
        best_blocks = b
        best_len = len(b)
        
    if best_blocks is not None:
        # Use most of the remaining time for local search
        ls_deadline = deadline - 5.0
        ls = _local_search(best_blocks, v, k, t, rng, ls_deadline)
        if _is_valid(ls, v, k, t) and len(ls) < best_len:
            best_blocks = ls
            best_len = len(ls)
            
    # If we have time, try a few more restarts
    attempt = 2
    while time.time() < deadline - 20:
        rng = random.Random(1000 + attempt)
        # Give greedy less time, as local search is more effective
        g_deadline = time.time() + 10.0
        b = _greedy_cover(v, k, t, rng, g_deadline)
        if not _is_valid(b, v, k, t):
            attempt += 1
            continue
            
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = b
            
        # Local search on this new result
        if time.time() < deadline - 5:
            ls = _local_search(b, v, k, t, rng, deadline - 5)
            if _is_valid(ls, v, k, t) and len(ls) < best_len:
                best_len = len(ls)
                best_blocks = ls
                
        attempt += 1
        
    if best_blocks is None:
        return []
        
    return best_blocks
# EVOLVE-BLOCK-END
