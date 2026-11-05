import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple random trials
    trial = 0
    while time.time() < deadline:
        trial += 1
        
        # 1. Construct initial cover
        blocks = _greedy_cover(v, k, t, all_tsubsets_set, deadline, trial)
        if blocks is None:
            continue
        
        # 2. Remove redundant blocks (fast pass)
        blocks = _remove_redundant(blocks, v, k, t, deadline)
        
        # 3. Structural Refinement: 2-Swap Local Search
        #    This attempts to reduce block count by finding a block B and a new block B'
        #    such that B' covers all "critical" t-subsets of B (those covered only by B)
        #    and B' allows the removal of at least one other block (or B itself becomes
        #    redundant relative to the new configuration, though usually we remove B).
        #    Actually, the standard move is:
        #    - Pick a block B.
        #    - Identify its "unique" coverage (t-subsets only in B).
        #    - Try to construct a new block B' that covers all unique subsets of B.
        #    - If B' exists, replace B with B'.
        #    - Then check if any *other* block becomes redundant.
        #    - If a block is removed, count decreases.
        
        improved_blocks = _two_swap_search(blocks, v, k, t, deadline)
        
        # 4. Final cleanup
        improved_blocks = _remove_redundant(improved_blocks, v, k, t, deadline)
        
        if len(improved_blocks) < best_len:
            best_len = len(improved_blocks)
            best_blocks = improved_blocks
            
        # Early termination if we hit the known best
        if best_len <= 76:
            break
            
    if best_blocks is None:
        # Fallback: run a deterministic greedy to ensure we return *something* valid
        blocks = _greedy_cover(v, k, t, all_tsubsets_set, deadline, -1)
        if blocks is not None:
            best_blocks = _remove_redundant(blocks, v, k, t, deadline)
        else:
            best_blocks = []
            
    return [list(b) for b in best_blocks]

def _greedy_cover(v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Greedy covering with randomized tie-breaking.
    """
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    
    uncovered = set(all_tsubsets_set)
    blocks = []
    points = list(range(v))
    
    while uncovered:
        if time.time() > deadline:
            return None
        
        # Pick a random uncovered t-subset as seed
        seed = tuple(rng.choice(list(uncovered)))
        block_set = set(seed)
        
        while len(block_set) < k:
            if time.time() > deadline:
                return None
            
            best_x = -1
            best_gain = -1
            best_tiebreakers = []
            
            t_minus_1_subsets = list(itertools.combinations(sorted(block_set), t-1))
            
            for x in points:
                if x in block_set:
                    continue
                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_tiebreakers = [x]
                elif gain == best_gain and gain > 0:
                    best_tiebreakers.append(x)
            
            if best_x == -1:
                # If no gain, pick random to fill size
                available = [x for x in points if x not in block_set]
                if not available:
                    return None
                best_x = rng.choice(available)
                
            block_set.add(best_x)
            
        block = tuple(sorted(block_set))
        blocks.append(block)
        
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            
    return blocks

def _remove_redundant(blocks, v, k, t, deadline):
    """
    Iteratively removes blocks that are redundant.
    """
    if not blocks:
        return blocks
    
    cover_count = Counter()
    for block in blocks:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1
            
    improved = list(blocks)
    changed = True
    
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = improved[i]
            redundant = True
            for s in itertools.combinations(block, t):
                if cover_count[s] <= 1:
                    redundant = False
                    break
            
            if redundant:
                for s in itertools.combinations(block, t):
                    cover_count[s] -= 1
                improved.pop(i)
                changed = True
                
    return improved

def _two_swap_search(blocks, v, k, t, deadline):
    """
    Structural improvement:
    For each block B, identify its 'critical' t-subsets (covered only by B).
    Try to find a new block B' that covers all critical subsets of B.
    If found, replace B with B' and check if any other block becomes redundant.
    This is a '2-swap' in the sense of replacing one block and potentially removing another.
    """
    if not blocks:
        return blocks
    
    improved = [list(b) for b in blocks]
    n_blocks = len(improved)
    
    # Precompute all possible blocks? No, C(24,7) is ~346k, feasible to precompute t-subsets for each?
    # Or just generate candidates on the fly.
    # Let's precompute the t-subsets for all possible blocks to speed up coverage checks?
    # C(24,7) = 346104. Each has C(7,3)=35 t-subsets.
    # Total t-subsets entries: 346104 * 35 ~ 12M. This might be heavy for memory/time.
    # Better: on-the-fly check.
    
    points = list(range(v))
    
    # Initial cover count
    cover_count = Counter()
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1
    
    any_change = True
    while any_change and time.time() < deadline:
        any_change = False
        
        # Iterate blocks in random order to avoid bias
        order = list(range(len(improved)))
        random.shuffle(order)
        
        for i in order:
            if time.time() > deadline:
                return improved
            
            block_i = improved[i]
            if not block_i: continue
            
            # 1. Identify critical subsets of block_i
            critical_subs = []
            for s in itertools.combinations(block_i, t):
                if cover_count[s] == 1:
                    critical_subs.append(s)
            
            if not critical_subs:
                # Block i is already redundant? _remove_redundant should have caught it.
                # But maybe it became redundant due to previous swaps in this loop.
                # If so, remove it.
                # Check redundancy again
                is_red = True
                for s in itertools.combinations(block_i, t):
                    if cover_count[s] <= 1:
                        is_red = False
                        break
                if is_red:
                    for s in itertools.combinations(block_i, t):
                        cover_count[s] -= 1
                    improved.pop(i)
                    any_change = True
                    # Re-index? We are iterating over a copy of indices.
                    # It's safer to break and restart if list size changes.
                    break
                continue
            
            # 2. Try to find a replacement block B'
            # B' must cover all critical_subs.
            # Also, B' should ideally cover many other subsets to make other blocks redundant.
            # Strategy: Pick a random subset of critical_subs to seed B'?
            # Or just try to construct a block containing all critical points?
            # Critical subsets are t-subsets. Their union is a set of points.
            # Let P_crit = union of all points in critical_subs.
            # B' must contain P_crit? No, B' must *cover* each critical subset.
            # Since critical subsets are t-subsets, and B' is a k-set.
            # If a critical subset S is in B', then S is covered.
            # So B' must contain at least one point from... no, B' must *contain* S as a subset?
            # Yes, "lies in some block" means S is a subset of the block.
            # So for every critical subset S, S must be a subset of B'.
            # This implies B' must contain the union of all critical subsets?
            # If S1 = {1,2,3} and S2 = {3,4,5}, B' must contain {1,2,3} AND {3,4,5}.
            # So B' must contain {1,2,3,4,5}.
            # So B' must contain P_union = union(critical_subs).
            
            p_union = set()
            for s in critical_subs:
                p_union.update(s)
            
            if len(p_union) > k:
                # Impossible to cover all critical subsets with a single block of size k
                continue
                
            # B' must contain p_union.
            # We need to choose (k - len(p_union)) more points from the remaining points.
            # Remaining points: points not in p_union.
            # We want to choose these extra points to maximize the chance of making
            # other blocks redundant.
            # Heuristic: Choose points that appear in many other blocks' critical sets?
            # Or just try a few random completions.
            
            extra_needed = k - len(p_union)
            available_points = [x for x in points if x not in p_union]
            
            if extra_needed == 0:
                candidate_blocks = [tuple(sorted(p_union))]
            else:
                # Generate a few random candidates
                # If extra_needed is small, we can enumerate?
                # C(24-|union|, extra_needed).
                # |union| >= t = 3. Max |union| is k=7.
                # If |union|=3, extra=4. C(21,4) = 5985. Feasible to enumerate all?
                # If |union|=4, extra=3. C(20,3) = 1140.
                # If |union|=5, extra=2. C(19,2) = 171.
                # If |union|=6, extra=1. C(18,1) = 18.
                # If |union|=7, extra=0.
                # So enumeration is feasible.
                
                candidate_blocks = []
                for extra_combo in itertools.combinations(available_points, extra_needed):
                    cand = tuple(sorted(p_union | set(extra_combo)))
                    candidate_blocks.append(cand)
            
            # Evaluate candidates
            best_cand = None
            best_score = -1
            
            # For each candidate, check if it covers all criticals (it does by construction)
            # and count how many *other* blocks become redundant if we replace block_i with cand.
            # Actually, replacing block_i with cand changes cover_count.
            # We need to simulate:
            # 1. Remove block_i from cover_count.
            # 2. Add cand to cover_count.
            # 3. Check if any block j != i is now redundant.
            
            # To speed up, we can pre-compute the "redundancy potential" of other blocks.
            # A block j is redundant if all its t-subsets have count >= 2 (in the new config).
            # In the current config, let c_j(s) be the count of s in block j (1 if s in j, 0 else).
            # This is complex to simulate quickly for all j.
            
            # Simpler heuristic:
            # Score a candidate by how many t-subsets it covers that are currently covered
            # by exactly one other block (besides block_i).
            # If cand covers a subset s, and cover_count[s] == 2 (one is block_i, one is block_j),
            # then after replacing block_i with cand, cover_count[s] becomes 1 (from cand) + 1 (from block_j) = 2?
            # Wait.
            # Current: block_i covers s, block_j covers s. Count = 2.
            # New: block_i removed. Count = 1 (from block_j).
            # Then cand added. If cand covers s, Count = 2.
            # So s is still covered by 2 blocks. No redundancy created for s.
            
            # To create redundancy for block_j, we need block_j to have a subset s_j
            # such that after swap, cover_count[s_j] becomes 1.
            # This happens if:
            # - s_j was covered by block_i and block_j (count 2).
            # - We remove block_i (count becomes 1).
            # - We add cand. If cand does NOT cover s_j, count stays 1.
            # - If s_j is the *only* subset of block_j with count 1 in the new config, block_j is redundant.
            
            # This is getting complicated. Let's just do a few random trials or a simple heuristic.
            # Heuristic: Prefer candidates that share many points with other blocks?
            # Or just pick the first valid candidate?
            
            # Let's try a simple scoring:
            # Score = number of t-subsets in cand that are NOT in block_i.
            # (New coverage).
            
            for cand in candidate_blocks:
                # Check validity: cand must be distinct from existing blocks?
                # If cand is already in improved (and not block_i), it's a duplicate.
                # We should avoid duplicates.
                if cand in improved and cand != tuple(block_i):
                    continue
                
                # Calculate gain: how many new t-subsets does cand cover compared to block_i?
                # And does it help remove others?
                # Let's just pick a random candidate from the list to add diversity.
                # Or the one with max "new" coverage.
                
                # For now, let's just try the first few candidates and see if we can remove a block.
                # To keep it fast, we only test up to 10 candidates per block.
                
                if len(candidate_blocks) > 10:
                    candidate_blocks = random.sample(candidate_blocks, 10)
                
                for cand in candidate_blocks:
                    # Simulate swap
                    # 1. Update cover_count: remove block_i, add cand
                    # We need to be careful with counts.
                    
                    # Temporarily decrement counts for block_i
                    for s in itertools.combinations(block_i, t):
                        cover_count[s] -= 1
                    
                    # Increment counts for cand
                    for s in itertools.combinations(cand, t):
                        cover_count[s] = cover_count.get(s, 0) + 1
                    
                    # Check for redundancy in other blocks
                    removed_any = False
                    for j in range(len(improved)):
                        if j == i:
                            continue
                        block_j = improved[j]
                        is_red = True
                        for s in itertools.combinations(block_j, t):
                            if cover_count[s] <= 1:
                                is_red = False
                                break
                        if is_red:
                            # Remove block_j
                            for s in itertools.combinations(block_j, t):
                                cover_count[s] -= 1
                            improved.pop(j)
                            removed_any = True
                            # Adjust index i if j < i
                            if j < i:
                                i -= 1
                            any_change = True
                            break # Only remove one per swap to keep it simple? Or loop?
                    
                    if not removed_any:
                        # Undo swap if no block was removed
                        # Decrement cand
                        for s in itertools.combinations(cand, t):
                            cover_count[s] -= 1
                        # Increment block_i
                        for s in itertools.combinations(block_i, t):
                            cover_count[s] = cover_count.get(s, 0) + 1
                        # Continue to next candidate
                        continue
                    else:
                        # Swap successful and block removed
                        # Replace block_i with cand
                        improved[i] = list(cand)
                        # We already updated cover_count for the swap and removal.
                        # We need to ensure block_i's old counts are removed (done)
                        # and cand's counts are added (done).
                        # And removed block's counts are decremented (done).
                        # So cover_count is correct.
                        # Break out of candidate loop
                        break
            
            if any_change:
                # List size changed, restart iteration
                break
                
    return improved
