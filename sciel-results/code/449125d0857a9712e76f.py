"""Covering design C(24,7,3) via bitset-accelerated greedy + intensive local search."""
import itertools
import time
import random
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 118  # Safety margin before 150s kill
    
    # Precompute all t-subsets and assign indices
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Bitset for uncovered: use Python int as bitset
    full_mask = (1 << n_tsets) - 1
    
    # Precompute for each point, the set of t-sets it belongs to (as bitset)
    # point_tsets[p] = bitset of t-sets containing point p
    point_tsets = [0] * v
    for ts_idx, ts in enumerate(all_tsets):
        bit = 1 << ts_idx
        for p in ts:
            point_tsets[p] |= bit
    
    # For a block (set of k points), compute its coverage bitset
    # block_mask(block) = OR of point_tsets[p] for p in block
    # But we need to be careful: a t-set is covered by a block iff all t points are in the block
    # Actually, the t-sets covered by a block are exactly the t-subsets of the block
    # Let's precompute: for each possible k-subset, its coverage mask is expensive
    # Instead, compute on the fly using the point_tsets approach:
    # A t-set ts is in block B iff all elements of ts are in B
    # mask(B) = AND over p in B of (all t-sets containing p) ... no that's wrong
    # mask(B) = set of t-sets that are subsets of B
    # = intersection over p NOT in B of (complement of point_tsets[p])? No.
    # Better: mask(B) = {ts : ts ⊆ B} = AND over p in B of point_tsets[p] ∩ ... no
    # Actually: ts ⊆ B iff for all p in ts, p in B. 
    # The set of t-sets contained in B = {ts : ts ⊆ B}
    # We can compute this as: start with full_mask, and for each p NOT in B, remove all t-sets containing p
    # mask(B) = full_mask AND (AND over p NOT in B of (~point_tsets[p]))
    # That's v-k ANDs which is 17 for v=24, k=7. Still expensive.
    
    # Alternative: precompute for each (t-1)-subset, the t-sets extending it
    # For gain calculation when adding point x to current block S:
    # gain(x) = number of uncovered t-sets of form {x} ∪ R where R ⊆ S, |R| = t-1
    # Precompute: for each (t-1)-subset R, list of t-sets containing R
    # For t=3, (t-1)=2, so for each pair R, list of triples containing R
    
    if t == 3:
        # For each pair (a,b), the triples containing it are (a,b,c) for c > b
        # Precompute pair_to_triples: pair -> list of triple indices
        pair_to_triples = defaultdict(list)
        for ts_idx, ts in enumerate(all_tsets):
            a, b, c = ts
            pair_to_triples[(a, b)].append(ts_idx)
            pair_to_triples[(a, c)].append(ts_idx)
            pair_to_triples[(b, c)].append(ts_idx)
    else:
        # General case
        tm1_to_tsets = defaultdict(list)
        for ts_idx, ts in enumerate(all_tsets):
            for i in range(t):
                tm1 = tuple(sorted(ts[:i] + ts[i+1:]))
                tm1_to_tsets[tm1].append(ts_idx)
    
    def _compute_block_mask(block):
        """Compute bitset of t-sets covered by this block."""
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << tset_index[ts])
        return mask
    
    def _greedy_build(uncovered_mask, deadline_inner):
        """Build a covering using greedy with bitset operations."""
        blocks = []
        block_masks = []
        uncovered = uncovered_mask
        n_uncovered = uncovered.bit_count()
        
        while uncovered and time.time() < deadline_inner:
            # Pick a random uncovered t-set to seed
            # Find an uncovered bit
            # Use a random position among uncovered bits
            # For speed, pick a random t-set and check if uncovered
            attempts = 0
            while attempts < 200:
                idx = random.randint(0, n_tsets - 1)
                if (uncovered >> idx) & 1:
                    seed_ts = all_tsets[idx]
                    break
                attempts += 1
            else:
                # Fallback: find first uncovered
                idx = 0
                while idx < n_tsets and not ((uncovered >> idx) & 1):
                    idx += 1
                if idx >= n_tsets:
                    break
                seed_ts = all_tsets[idx]
            
            block = list(seed_ts)
            block_set = set(block)
            
            while len(block) < k and time.time() < deadline_inner:
                best_x = -1
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    if t == 3:
                        # For each pair R in block, check triples (R, x)
                        for i in range(len(block)):
                            for j in range(i+1, len(block)):
                                a, b = block[i], block[j]
                                if a > b:
                                    a, b = b, a
                                # Triples containing (a,b) and x
                                # The triple is (a,b,x) sorted
                                # Check if (a,b,x) is in pair_to_triples[(min(a,b),max(a,b))]
                                # Actually we need: is (min(a,b), max(a,b), x) a valid triple?
                                # The triple must be sorted. Let's just check all pairs in block
                                # For pair (a,b) from block, the triples are (a,b,c) for c not in {a,b}
                                # We want c=x, so check if (a,b,x) is a triple, i.e., x is the third element
                                # The triple index: we need (min(a,b), max(a,b), x) to be sorted
                                # Let's compute the sorted triple and check
                                triple = tuple(sorted([a, b, x]))
                                # Check if this triple is uncovered
                                ts_idx = tset_index.get(triple, -1)
                                if ts_idx >= 0 and (uncovered >> ts_idx) & 1:
                                    gain += 1
                    else:
                        for R in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(list(R) + [x]))
                            ts_idx = tset_index.get(ts, -1)
                            if ts_idx >= 0 and (uncovered >> ts_idx) & 1:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    if best_gain == 0:
                        remaining = [x for x in range(v) if x not in block_set]
                        if remaining:
                            best_x = random.choice(remaining)
                        else:
                            break
                    else:
                        break
                else:
                    best_x = random.choice(best_candidates)
                
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) == k:
                block.sort()
                bm = _compute_block_mask(block)
                if bm & uncovered:  # Only add if it covers something new
                    uncovered &= ~bm
                    blocks.append(block)
                    block_masks.append(bm)
        
        return blocks
    
    def _prune_redundant(blocks):
        """Remove blocks that don't cover any unique t-set."""
        if not blocks:
            return blocks
        
        # Count coverage for each t-set
        coverage_count = defaultdict(int)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                coverage_count[ts] += 1
        
        to_remove = set()
        for i, b in enumerate(blocks):
            is_redundant = True
            for ts in itertools.combinations(b, t):
                if coverage_count[ts] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                to_remove.add(i)
        
        if to_remove:
            return [b for i, b in enumerate(blocks) if i not in to_remove]
        return blocks
    
    def _local_search_remove(blocks, deadline_inner):
        """Try to remove blocks one by one, replacing with better blocks if needed."""
        current = list(blocks)
        
        # Compute coverage counts
        def _get_coverage(blks):
            cov = defaultdict(int)
            for b in blks:
                for ts in itertools.combinations(b, t):
                    cov[ts] += 1
            return cov
        
        cov = _get_coverage(current)
        
        # Try removing each block
        for i in range(len(current) - 1, -1, -1):
            if time.time() >= deadline_inner:
                break
            
            # Check if removing block i leaves any t-set uncovered
            unique_to_i = []
            for ts in itertools.combinations(current[i], t):
                if cov[ts] == 1:
                    unique_to_i.append(ts)
            
            if not unique_to_i:
                # Can remove directly
                current.pop(i)
                # Update coverage
                for ts in itertools.combinations(current[i] if i < len(current) else [], t):
                    pass
                # Recompute coverage
                cov = _get_coverage(current)
                continue
            
            # Try to find a replacement block that covers all unique_to_i
            # and allows further pruning
            best_replacement = None
            best_score = -1
            
            for attempt in range(50):
                if time.time() >= deadline_inner:
                    break
                
                # Start from block i, modify it
                cand = list(current[i])
                random.shuffle(cand)
                
                # Try to swap some elements
                for _ in range(random.randint(1, 3)):
                    pos = random.randint(0, len(cand) - 1)
                    removed = cand[pos]
                    new_p = random.randint(0, v - 1)
                    if new_p not in cand:
                        cand[pos] = new_p
                    else:
                        cand = [p for p in cand if p != removed]
                        cand.append(removed)
                
                if len(set(cand)) != k:
                    continue
                cand.sort()
                
                if cand == current[i]:
                    continue
                
                # Check if cand covers all unique_to_i
                cand_set = set(cand)
                valid = True
                for ts in unique_to_i:
                    if not set(ts).issubset(cand_set):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Score: how many other blocks' unique t-sets does cand cover?
                score = 0
                cand_tsets = set(itertools.combinations(cand, t))
                for j in range(len(current)):
                    if j == i:
                        continue
                    for ts in itertools.combinations(current[j], t):
                        if cov[ts] == 1 and ts in cand_tsets:
                            score += 1
                
                if score > best_score:
                    best_score = score
                    best_replacement = cand
            
            if best_replacement is not None:
                test_blocks = list(current)
                test_blocks[i] = best_replacement
                pruned = _prune_redundant(test_blocks)
                
                if len(pruned) < len(current):
                    current = pruned
                    cov = _get_coverage(current)
        
        return current
    
    def _pairwise_swap_search(blocks, deadline_inner):
        """Try swapping elements between pairs of blocks to enable pruning."""
        current = list(blocks)
        
        def _get_coverage(blks):
            cov = defaultdict(int)
            for b in blks:
                for ts in itertools.combinations(b, t):
                    cov[ts] += 1
            return cov
        
        for iteration in range(20):
            if time.time() >= deadline_inner:
                break
            
            improved = False
            cov = _get_coverage(current)
            
            for i in range(len(current)):
                if time.time() >= deadline_inner:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() >= deadline_inner:
                        break
                    
                    # Try swapping one element from block i with one from block j
                    for pi in range(k):
                        for pj in range(k):
                            if current[i][pi] == current[j][pj]:
                                continue
                            
                            new_i = list(current[i])
                            new_j = list(current[j])
                            new_i[pi], new_j[pj] = new_j[pj], new_i[pi]
                            
                            if len(set(new_i)) != k or len(set(new_j)) != k:
                                continue
                            new_i.sort()
                            new_j.sort()
                            
                            test = list(current)
                            test[i] = new_i
                            test[j] = new_j
                            
                            pruned = _prune_redundant(test)
                            if len(pruned) < len(current):
                                current = pruned
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current
    
    # Main search loop
    best_blocks = None
    best_count = float('inf')
    
    n_restarts = 30  # More restarts with faster greedy
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        random.seed(restart * 7919 + 13)
        
        # 1. Greedy build
        deadline_greedy = min(time.time() + 15, deadline - 20)
        blocks = _greedy_build(full_mask, deadline_greedy)
        
        if not blocks:
            continue
        
        # Verify coverage
        covered_mask = 0
        for b in blocks:
            covered_mask |= _compute_block_mask(b)
        
        if covered_mask != full_mask:
            # Force cover remaining
            uncovered = full_mask & ~covered_mask
            while uncovered and time.time() < deadline - 10:
                # Find an uncovered t-set
                idx = 0
                while idx < n_tsets and not ((uncovered >> idx) & 1):
                    idx += 1
                if idx >= n_tsets:
                    break
                ts = all_tsets[idx]
                block = list(ts)
                block_set = set(block)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                covered_mask |= _compute_block_mask(block)
                uncovered &= ~_compute_block_mask(block)
        
        if covered_mask != full_mask:
            continue
        
        # 2. Prune
        pruned = _prune_redundant(blocks)
        prev_len = len(pruned)
        while len(pruned) < prev_len and time.time() < deadline - 15:
            prev_len = len(pruned)
            pruned = _prune_redundant(pruned)
        
        # 3. Local search
        if time.time() < deadline - 10:
            ls_blocks = _local_search_remove(pruned, deadline - 8)
            if len(ls_blocks) < len(pruned):
                pruned = ls_blocks
        
        # 4. Pairwise swap search
        if time.time() < deadline - 5:
            ps_blocks = _pairwise_swap_search(pruned, deadline - 3)
            if len(ps_blocks) < len(pruned):
                pruned = ps_blocks
        
        # 5. Final prune
        prev_len = len(pruned)
        while len(pruned) < prev_len and time.time() < deadline - 2:
            prev_len = len(pruned)
            pruned = _prune_redundant(pruned)
        
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_blocks = list(pruned)
        
        # Early stop if we hit the known best
        if best_count <= 76:
            break
    
    if best_blocks is None or len(best_blocks) == 0:
        # Fallback
        all_tsets_set = set(all_tsets)
        uncovered = set(all_tsets)
        blocks = []
        
        while uncovered and time.time() < deadline - 2:
            seed_ts = random.choice(list(uncovered))
            block = list(seed_ts)
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for R in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(list(R) + [x]))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) == k:
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                covered = set(itertools.combinations(block, t))
                uncovered -= covered
        
        best_blocks = blocks
    
    return best_blocks
