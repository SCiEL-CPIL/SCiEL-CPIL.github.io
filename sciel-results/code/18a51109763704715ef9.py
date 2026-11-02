"""New approach: column-generation style set cover with greedy pricing."""
import itertools
import random
import sys


def construct(v, k, t):
    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    
    # Map each t-subset to an index
    t_index = {ts: i for i, ts in enumerate(all_t)}
    
    # uncovered: set of indices of t-subsets not yet covered
    uncovered = set(range(n_t))
    
    # For each point x, precompute which t-subsets contain x
    # Actually, for efficiency, we'll compute on the fly but cache per-block
    
    blocks = []
    random.seed(42)
    
    # Precompute: for each pair of points (a,b), which t-subsets contain both
    # This is too expensive for large v. Instead, we'll use a smarter greedy.
    
    # Greedy set cover with column generation pricing:
    # At each step, find the block B (k-subset) that maximizes |B's t-subsets ∩ uncovered|
    # Use a multi-stage greedy to construct candidate blocks.
    
    # Precompute for each point x, the list of t-subset indices containing x
    point_to_ts = [[] for _ in range(v)]
    for ts in all_t:
        for x in ts:
            point_to_ts[x].append(t_index[ts])
    
    # For efficiency, track coverage counts: cov_count[i] = how many blocks cover t-subset i
    # But since we're doing greedy, we just need uncovered set.
    
    # Heuristic to find good blocks:
    # 1. Start with the t-subset with the fewest candidate blocks (or just pick uncovered)
    # 2. Grow it to k elements by adding points that maximize new coverage
    
    # Precompute: for each (t-1)-subset, which points can extend it
    # Actually, let's just do a direct greedy.
    
    # For faster gain computation, precompute for each point x and each (t-1)-subset S
    # whether S ∪ {x} is a t-subset and its index.
    
    # Since t=4, k=7, v=19: C(19,3)=969 (t-1 subsets), points=19
    # Precompute: ext[(a,b,c)][x] = index of (a,b,c,x) if x not in {a,b,c}
    
    from functools import lru_cache
    
    # Precompute all (t-1)-subsets and their extensions
    t_minus_1 = t - 1
    ext_map = {}  # (a,b,c) -> {x: idx}
    for ts in all_t:
        ts_set = set(ts)
        for x in ts_set:
            rest = tuple(sorted(ts_set - {x}))
            if rest not in ext_map:
                ext_map[rest] = {}
            ext_map[rest][x] = t_index[ts]
    
    # Now greedy:
    # At each iteration, try multiple starting points and pick the best block.
    
    def block_coverage(block_set):
        """Return count of uncovered t-subsets in this block."""
        count = 0
        for ts in itertools.combinations(sorted(block_set), t):
            idx = t_index.get(ts, -1)
            if idx >= 0 and idx in uncovered:
                count += 1
        return count
    
    # For speed, we'll use the precomputed ext_map
    def block_coverage_fast(block_list):
        """Fast coverage count using precomputed ext_map for t=4 case."""
        if t != 4:
            return block_coverage(set(block_list))
        
        block_set = set(block_list)
        count = 0
        # For each triple in the block, check if the 4th point is in the block
        # Actually, iterate over all 4-subsets of the block
        bl = sorted(block_list)
        for i in range(len(bl)):
            for j in range(i+1, len(bl)):
                for l in range(j+1, len(bl)):
                    rest = (bl[i], bl[j], bl[l])
                    if rest in ext_map:
                        for x in ext_map[rest]:
                            if x in block_set:
                                idx = ext_map[rest][x]
                                if idx in uncovered:
                                    count += 1
        return count
    
    # Greedy construction of a block given a starting point
    def grow_block(start):
        block = {start}
        # Add points one by one, each time picking the point that maximizes new coverage
        # We need to track which t-subsets are covered so far in this block
        for _ in range(k - 1):
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: how many uncovered t-subsets would be added
                # by adding x to current block
                new_block = block | {x}
                # Only count t-subsets that include x (new ones)
                gain = 0
                if t == 4:
                    for i in block:
                        for j in block:
                            if i < j:
                                rest = (i, j, x)
                                # Check if rest is sorted
                                rest_sorted = tuple(sorted(rest))
                                if rest_sorted in ext_map:
                                    # The 4th point must be in block
                                    for y in block:
                                        if y != x and y in rest_sorted:
                                            idx = ext_map[rest_sorted][y]
                                            if idx in uncovered:
                                                gain += 1
                                # Also check other combinations
                                # Actually, any t-subset containing x and 3 points from block
                                pass
                    # Better: iterate over all (t-1)-subsets of current block
                    for sub in itertools.combinations(sorted(block), t - 1):
                        rest_sorted = tuple(sorted(sub))
                        if rest_sorted in ext_map:
                            idx = ext_map[rest_sorted].get(x, -1)
                            if idx >= 0 and idx in uncovered:
                                gain += 1
                else:
                    # General case
                    for sub in itertools.combinations(sorted(block), t - 1):
                        ts = tuple(sorted(sub + (x,)))
                        idx = t_index.get(ts, -1)
                        if idx >= 0 and idx in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                break
            block.add(best_x)
        
        return sorted(block)
    
    # Main loop
    while uncovered:
        # Try multiple starting points and pick the best resulting block
        best_block = None
        best_cov = -1
        
        # Pick a few candidate starting points
        # 1. Points in uncovered t-subsets with high frequency
        # For simplicity, try the first uncovered t-subset's elements and a few random ones
        first_uncovered = min(uncovered)
        start_candidates = set(all_t[first_uncovered])
        
        # Add some random candidates
        for _ in range(5):
            if uncovered:
                idx = random.choice(list(uncovered))
                start_candidates.add(all_t[idx][0])
        
        for start in start_candidates:
            if len(block := grow_block(start)) == k:
                cov = block_coverage_fast(block)
                if cov > best_cov:
                    best_cov = cov
                    best_block = block
        
        if best_block is None or best_cov <= 0:
            # Fallback: just take first uncovered t-subset and extend arbitrarily
            first_uncovered = min(uncovered)
            base = all_t[first_uncovered]
            block = list(base)
            for x in range(v):
                if x not in block and len(block) < k:
                    block.append(x)
            best_block = sorted(block)
            best_cov = block_coverage_fast(best_block)
        
        blocks.append(best_block)
        
        # Update uncovered
        bl = sorted(best_block)
        if t == 4:
            for i in range(len(bl)):
                for j in range(i+1, len(bl)):
                    for l in range(j+1, len(bl)):
                        rest = (bl[i], bl[j], bl[l])
                        if rest in ext_map:
                            for x in ext_map[rest]:
                                if x in set(bl):
                                    idx = ext_map[rest][x]
                                    uncovered.discard(idx)
        else:
            for ts in itertools.combinations(bl, t):
                idx = t_index.get(ts, -1)
                if idx >= 0:
                    uncovered.discard(idx)
    
    return blocks
