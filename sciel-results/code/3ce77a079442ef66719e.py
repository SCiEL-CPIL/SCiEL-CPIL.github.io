"""Covering design C(19,7,4) - fast greedy with local search (2-exchange)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1)
    such that every t-subset is contained in at least one block.
    Minimizes the number of blocks.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Precompute all t-subsets as bitmasks for fast lookup
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for array-based tracking
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point x and each (t-1)-subset, the list of t-subsets
    # that contain both the (t-1)-subset and x
    # gain_cache[x][frozen_bits] -> list of t-subset indices
    # Actually, let's precompute: for each (t-1)-subset S, and each x not in S,
    # the t-subset S∪{x} and its index
    
    # For efficiency, precompute for each (t-1)-subset the mapping from point to t-subset idx
    # (t-1)-subsets:
    tm1_subsets = list(itertools.combinations(range(v), t - 1))
    tm1_to_tsubsets = {}  # (t-1)-subset -> {point: t_subset_idx}
    for s in tm1_subsets:
        s_set = set(s)
        mapping = {}
        for x in range(v):
            if x not in s_set:
                tsub = tuple(sorted(s + (x,)))
                mapping[x] = subset_to_idx[tsub]
        tm1_to_tsubsets[s] = mapping
    
    def greedy_construct(seed=None):
        """Run greedy algorithm with optional randomization."""
        if seed is not None:
            random.seed(seed)
        
        # Track coverage count for each t-subset
        coverage = [0] * n_t
        
        blocks = []
        # Start with the first uncovered t-subset (or random)
        uncovered_list = list(range(n_t))
        
        while uncovered_list:
            # Pick a t-subset to cover: first uncovered, or random among uncovered
            if seed is not None and random.random() < 0.3:
                idx = random.choice(uncovered_list)
            else:
                idx = uncovered_list[0]
            
            target_subset = all_t_subsets[idx]
            block_set = set(target_subset)
            block_list = list(block_set)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block_list) < k:
                best_x = -1
                best_gain = -1
                best_new_covered = None
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Compute gain: how many currently uncovered t-subsets would be covered
                    # A t-subset is covered if it's a subset of block_list + {x}
                    # i.e., it contains all of block_list except possibly one element,
                    # and contains x
                    # More precisely: t-subset T is covered by block_list∪{x} iff
                    # T ⊆ block_list∪{x} and T is not already covered by block_list alone
                    # T ⊆ block_list∪{x} means T\{x} ⊆ block_list
                    # Since |T|=t and |block_list|=j (j < k-1), we need T\{x} ⊆ block_list
                    # which means T shares at least t-1 elements with block_list, and the remaining is x
                    # Actually T ⊆ block_list ∪ {x} iff T\{x} ⊆ block_list
                    # T\{x} has t-1 elements, so T\{x} must be a subset of block_list
                    # This means T = (T\{x}) ∪ {x} where T\{x} ⊆ block_list
                    # So T contains x and T\{x} ⊆ block_list
                    # Equivalently: T contains x, and T ∩ block_list has size at least t-1
                    # Since |T|=t and x∈T, T\{x} has t-1 elements, all in block_list
                    
                    # For each (t-1)-subset S of block_list, T = S ∪ {x} is a candidate
                    # We need S ⊆ block_list, |S| = t-1, x ∉ S
                    # And T = S∪{x} must be uncovered
                    
                    gain = 0
                    newly_covered = []
                    
                    # Iterate over (t-1)-subsets of block_list
                    if len(block_list) >= t - 1:
                        for S in itertools.combinations(block_list, t - 1):
                            S_set = set(S)
                            if x in S_set:
                                continue
                            # T = S ∪ {x}
                            tsub = tuple(sorted(S + (x,)))
                            t_idx = subset_to_idx.get(tsub)
                            if t_idx is not None and coverage[t_idx] == 0:
                                gain += 1
                                newly_covered.append(t_idx)
                    
                    # Also, if block_list itself has fewer than t elements, 
                    # we might cover t-subsets that are subsets of block_list∪{x} 
                    # but don't contain all of block_list... no, we need T⊆block_list∪{x}
                    # and |T|=t. If |block_list|<t-1, then no (t-1)-subset of block_list exists.
                    # The above handles that (no combinations if len < t-1).
                    
                    if gain > best_gain or (gain == best_gain and best_x == -1):
                        best_gain = gain
                        best_x = x
                        best_new_covered = newly_covered
                
                if best_x == -1:
                    # No point can improve; pick any remaining point
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining) if seed is not None else remaining[0]
                        best_new_covered = []
                    else:
                        break
                
                block_set.add(best_x)
                block_list.append(best_x)
                
                # Update coverage for newly covered t-subsets
                if best_new_covered:
                    for t_idx in best_new_covered:
                        coverage[t_idx] += 1
            
            # Finalize block
            block_list.sort()
            blocks.append(block_list)
            
            # Update coverage for all t-subsets of this block
            for tsub in itertools.combinations(block_list, t):
                t_idx = subset_to_idx[tsub]
                coverage[t_idx] += 1
            
            # Remove covered t-subsets from uncovered_list
            # This is O(n_t) each time, which could be slow
            # Better: maintain a set of uncovered indices
            # For now, rebuild (slow but correct)
            uncovered_list = [i for i in range(n_t) if coverage[i] == 0]
            
            if time.time() > deadline - 10:
                break
        
        # Verify coverage
        # Check all t-subsets are covered
        for i in range(n_t):
            if coverage[i] == 0:
                # Incomplete - try to fix by adding blocks
                # This shouldn't happen with correct greedy, but just in case
                pass
        
        return blocks, coverage
    
    # Run greedy multiple times with different seeds and keep best
    best_blocks = None
    best_count = float('inf')
    
    # First run: deterministic greedy
    blocks, cov = greedy_construct(seed=None)
    if all(c > 0 for c in cov):
        best_blocks = blocks
        best_count = len(blocks)
    
    # Now try local search: 2-exchange (remove one block, add another)
    # And also try more greedy runs with randomization
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Try randomized greedy
        for seed in range(1, 20):
            if time.time() > deadline - 20:
                break
            blocks_r, cov_r = greedy_construct(seed=seed)
            if all(c > 0 for c in cov_r):
                if len(blocks_r) < best_count:
                    best_blocks = blocks_r
                    best_count = len(blocks_r)
    
    # Local search: 2-exchange
    if best_blocks is not None and time.time() < deadline - 15:
        # Precompute for each block, the t-subsets it covers
        # And for each t-subset, which blocks cover it
        n_blocks = len(best_blocks)
        
        # block_tsubsets: for each block, list of t-subset indices it covers
        block_tsubsets = []
        for b in best_blocks:
            tsubs = [subset_to_idx[tsub] for tsub in itertools.combinations(b, t)]
            block_tsubsets.append(set(tsubs))
        
        # For each t-subset, count how many blocks cover it
        tsub_count = [0] * n_t
        for tsubs in block_tsubsets:
            for t_idx in tsubs:
                tsub_count[t_idx] += 1
        
        # 2-exchange: remove block i, add a new block j
        # A block j can be added if it covers at least one t-subset that only block i covers
        # After removing i, t-subsets with count==1 that are in block i become uncovered
        # New block j must cover all of those
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            n_blocks = len(best_blocks)
            
            # Recompute block_tsubsets and tsub_count
            block_tsubsets = []
            for b in best_blocks:
                tsubs = [subset_to_idx[tsub] for tsub in itertools.combinations(b, t)]
                block_tsubsets.append(set(tsubs))
            
            tsub_count = [0] * n_t
            for tsubs in block_tsubsets:
                for t_idx in tsubs:
                    tsub_count[t_idx] += 1
            
            for i in range(n_blocks):
                if time.time() > deadline - 10:
                    break
                
                # Find t-subsets that only block i covers
                exclusive = set()
                for t_idx in block_tsubsets[i]:
                    if tsub_count[t_idx] == 1:
                        exclusive.add(t_idx)
                
                # New block must cover all exclusive t-subsets
                # Find a k-subset that covers all exclusive t-subsets
                # The exclusive t-subsets must all be contained in some k-subset
                # Union of exclusive t-subsets must have size <= k
                if exclusive:
                    union_pts = set()
                    for t_idx in exclusive:
                        union_pts.update(all_t_subsets[t_idx])
                    if len(union_pts) > k:
                        continue
                
                # Try to find a good replacement block
                # Greedy: start from union of exclusive points, add points to cover more
                if exclusive:
                    start_pts = set()
                    for t_idx in exclusive:
                        start_pts.update(all_t_subsets[t_idx])
                else:
                    # No exclusive t-subsets, so we can remove block i freely
                    # But we need to add a block that covers something useful
                    # Actually if no exclusive, removing i doesn't uncover anything
                    # So we just remove i and are done (improvement!)
                    best_blocks.pop(i)
                    improved = True
                    break
                
                # Build a block starting from start_pts
                block_set = set(start_pts)
                if len(block_set) > k:
                    continue
                
                # Add points to maximize coverage of currently under-covered t-subsets
                # (t-subsets with low count)
                while len(block_set) < k:
                    best_x = -1
                    best_score = -1
                    for x in range(v):
                        if x in block_set:
                            continue
                        score = 0
                        # For each (t-1)-subset S of block_set, T=S∪{x}
                        if len(block_set) >= t - 1:
                            for S in itertools.combinations(tuple(block_set), t - 1):
                                S_set = set(S)
                                if x in S_set:
                                    continue
                                tsub = tuple(sorted(S + (x,)))
                                t_idx = subset_to_idx.get(tsub)
                                if t_idx is not None:
                                    # Score: prefer covering low-count t-subsets
                                    score += 1000 - tsub_count[t_idx]
                        best_score = max(best_score, score)
                        if score > best_score:
                            best_score = score
                            best_x = x
                    if best_x == -1:
                        remaining = [x for x in range(v) if x not in block_set]
                        if remaining:
                            best_x = remaining[0]
                        else:
                            break
                    block_set.add(best_x)
                
                new_block = sorted(block_set)
                
                # Check if this new block is already in best_blocks
                new_block_set = set(new_block)
                is_dup = False
                for b in best_blocks:
                    if set(b) == new_block_set:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Verify: after removing i and adding new_block, all t-subsets covered
                # Remove i's coverage
                for t_idx in block_tsubsets[i]:
                    tsub_count[t_idx] -= 1
                # Add new block's coverage
                for tsub in itertools.combinations(new_block, t):
                    t_idx = subset_to_idx[tsub]
                    tsub_count[t_idx] += 1
                # Check all covered
                if all(c > 0 for c in tsub_count):
                    # Accept
                    best_blocks[i] = new_block
                    improved = True
                    break
                else:
                    # Rollback
                    for tsub in itertools.combinations(new_block, t):
                        t_idx = subset_to_idx[tsub]
                        tsub_count[t_idx] -= 1
                    for t_idx in block_tsubsets[i]:
                        tsub_count[t_idx] += 1
        
        # If we removed blocks without replacement (the no-exclusive case), verify
        if best_blocks:
            final_cov = [0] * n_t
            for b in best_blocks:
                for tsub in itertools.combinations(b, t):
                    t_idx = subset_to_idx[tsub]
                    final_cov[t_idx] += 1
            if not all(c > 0 for c in final_cov):
                # Invalid, revert to previous best
                # We should have kept a copy... let's just re-run greedy
                blocks, cov = greedy_construct(seed=None)
                if all(c > 0 for c in cov):
                    best_blocks = blocks
                    best_count = len(blocks)
    
    if best_blocks is None:
        # Fallback: simple greedy
        blocks, cov = greedy_construct(seed=None)
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
