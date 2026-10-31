"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Exact Branch-and-Bound solver for Minimum Set Cover.
- Precomputes coverage masks for all k-subsets.
- Uses a greedy heuristic for an initial upper bound.
- B&B uses a fractional packing lower bound and dynamic variable ordering
  (choosing the uncovered t-subset with the fewest covering blocks).
"""
import time
import itertools
import math

_START = None
_DEADLINE = 118.0  # Safety margin

def _elapsed():
    return time.time() - _START

def construct(v, k, t):
    global _START
    _START = time.time()
    
    # 1. Generate all blocks (k-subsets) and all t-subsets
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_t_subs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_blocks)
    n_t = len(all_t_subs)
    
    # Map t-subsets to indices
    t_index = {ts: i for i, ts in enumerate(all_t_subs)}
    full_mask = (1 << n_t) - 1
    
    # 2. Precompute coverage masks for each block
    # mask[i] is an integer where bit j is 1 if block i covers t-subset j
    masks = [0] * n_blocks
    # Also precompute for each t-subset j, the list of blocks covering it (for LB and variable ordering)
    coverers = [[] for _ in range(n_t)]
    
    for i, blk in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(blk, t):
            j = t_index[ts]
            mask |= (1 << j)
            coverers[j].append(i)
        masks[i] = mask
        
    # Precompute the "frequency" of each t-subset (number of blocks covering it)
    # This is used for the lower bound: LB = sum(1/freq[j] for j in uncovered)
    freq = [len(c) for c in coverers]
    
    # 3. Greedy Heuristic to get an initial upper bound (incumbent)
    def greedy_solve():
        uncovered = full_mask
        selected = []
        while uncovered:
            if _elapsed() > _DEADLINE - 5:
                return None
            best_block = -1
            best_gain = 0
            # Iterate all blocks
            for i in range(n_blocks):
                m = masks[i]
                if m & uncovered:
                    gain = (m & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
            if best_block == -1:
                break # Should not happen if full_mask is valid
            selected.append(best_block)
            uncovered &= ~masks[best_block]
        return selected

    incumbent = greedy_solve()
    if incumbent is None:
        # Fallback if greedy timed out or failed
        # Return empty list if we can't even get a greedy solution, 
        # but greedy is very fast for this size.
        return []
        
    best_len = len(incumbent)
    best_solution = incumbent[:]
    
    # 4. Branch and Bound
    # State: (uncovered_mask, current_depth, current_solution)
    # We use a stack for iterative DFS.
    
    def calc_lb(uncovered):
        if uncovered == 0:
            return 0
        s = 0.0
        u = uncovered
        while u:
            lsb = u & -u
            j = lsb.bit_length() - 1
            s += 1.0 / freq[j]
            u ^= lsb
        return math.ceil(s)
    
    stack = [(full_mask, 0, [])]
    
    while stack:
        if _elapsed() > _DEADLINE:
            break
            
        uncovered, depth, sol = stack.pop()
        
        # Prune if depth + LB > best_len
        lb = calc_lb(uncovered)
        if depth + lb > best_len:
            continue
            
        if uncovered == 0:
            if depth < best_len:
                best_len = depth
                best_solution = sol[:]
            continue
            
        # Select variable (t-subset)
        # Find j in uncovered with min freq[j]
        u = uncovered
        min_freq = float('inf')
        target_j = -1
        while u:
            lsb = u & -u
            j = lsb.bit_length() - 1
            f = freq[j]
            if f < min_freq:
                min_freq = f
                target_j = j
            u ^= lsb
        
        if target_j == -1:
            continue
            
        # Branch on blocks covering target_j
        candidates = coverers[target_j]
        
        # Calculate gain for each candidate (coverage of other uncovered subsets)
        scored_candidates = []
        for i in candidates:
            m = masks[i]
            gain = (m & uncovered).bit_count()
            scored_candidates.append((gain, i))
            
        # Sort by gain descending to find good solutions early
        scored_candidates.sort(reverse=True)
        
        for gain, i in scored_candidates:
            new_uncovered = uncovered & ~masks[i]
            new_depth = depth + 1
            
            # Prune
            if new_depth + calc_lb(new_uncovered) >= best_len:
                continue
                
            if new_uncovered == 0:
                if new_depth < best_len:
                    best_len = new_depth
                    best_solution = sol + [i]
                continue
            else:
                stack.append((new_uncovered, new_depth, sol + [i]))

    # 5. Convert result
    if best_solution:
        result = [list(all_blocks[i]) for i in best_solution]
        return result
    else:
        # Fallback: return greedy result
        return [list(all_blocks[i]) for i in incumbent]
