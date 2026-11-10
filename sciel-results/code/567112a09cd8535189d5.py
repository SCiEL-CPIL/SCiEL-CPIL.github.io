import itertools
import time
import random
import math
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    # Precompute all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_index = {s: i for i, s in enumerate(all_t_subs)}
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # Coverage masks: Python integers as bitsets
    coverage_masks = []
    for ks in all_k_subs:
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_sub_index[ts]
            mask |= (1 << idx)
        coverage_masks.append(mask)
    
    # Build reverse index: for each t-subset index, which k-subsets cover it
    # This helps with faster redundancy checks
    t_to_k = [[] for _ in range(n_t)]
    for ki, mask in enumerate(coverage_masks):
        m = mask
        while m:
            low = m & (-m)
            idx = low.bit_length() - 1
            t_to_k[idx].append(ki)
            m ^= low
    
    full_mask = (1 << n_t) - 1
    
    # Greedy set cover
    def greedy_cover(uncovered_mask, existing_blocks, deadline):
        result = list(existing_blocks)
        unc = uncovered_mask
        used = set()
        for b in existing_blocks:
            used.add(b)
        
        while unc != 0 and time.time() < deadline:
            best_gain = 0
            best_ks = None
            for ki in range(n_k):
                if ki in used:
                    continue
                gain = (coverage_masks[ki] & unc).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_ks = ki
            
            if best_ks is None or best_gain == 0:
                break
            
            used.add(best_ks)
            result.append(all_k_subs[best_ks])
            unc &= ~coverage_masks[best_ks]
        
        return result, unc
    
    # Run greedy
    blocks, remaining_uncovered = greedy_cover(full_mask, [], deadline)
    
    if time.time() >= deadline:
        return [list(b) for b in blocks]
    
    def compute_uncovered_mask(block_list):
        covered = 0
        for b in block_list:
            for ts in itertools.combinations(b, t):
                idx = t_sub_index[ts]
                covered |= (1 << idx)
        return full_mask & ~covered
    
    def block_coverage(b):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = t_sub_index[ts]
            mask |= (1 << idx)
        return mask
    
    # Local search: remove redundant blocks, then try swaps
    def remove_redundant(blocks, deadline):
        """Remove blocks whose coverage is fully contained in others."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n = len(blocks)
            if n == 0:
                break
            # Compute union of all other blocks' coverage for each block
            # Use prefix/suffix for efficiency
            prefix = [0] * (n + 1)
            suffix = [0] * (n + 1)
            for i in range(n):
                prefix[i+1] = prefix[i] | block_coverage(blocks[i])
            for i in range(n-1, -1, -1):
                suffix[i] = suffix[i+1] | block_coverage(blocks[i])
            
            keep = []
            for i in range(n):
                b_cov = block_coverage(blocks[i])
                other_cov = prefix[i] | suffix[i+1]
                if (b_cov & ~other_cov) == 0:
                    changed = True
                else:
                    keep.append(blocks[i])
            if changed:
                blocks = keep
        return blocks
    
    def simulated_annealing(blocks, deadline):
        """Try block swaps with simulated annealing to find better configurations."""
        blocks = [list(b) for b in blocks]
        n_blocks = len(blocks)
        if n_blocks < 2:
            return blocks
        
        best_blocks = [list(b) for b in blocks]
        best_n = n_blocks
        
        # Precompute coverages
        coverages = [block_coverage(b) for b in blocks]
        total_cov = 0
        for c in coverages:
            total_cov |= c
        
        # Temperature schedule
        t0 = 1.0
        t_end = 0.01
        # Determine number of iterations based on time budget
        max_iter = 50000
        temp = t0
        
        for it in range(max_iter):
            if time.time() >= deadline:
                break
            # Decay temperature
            progress = it / max_iter
            temp = t0 * (t_end / t0) ** progress
            
            # Pick two random blocks
            i, j = random.sample(range(n_blocks), 2)
            bi = blocks[i]
            bj = blocks[j]
            
            # Pick random points to swap
            pi = random.randint(0, k-1)
            pj = random.randint(0, k-1)
            x = bi[pi]
            y = bj[pj]
            if x == y:
                continue
            
            new_bi = list(bi)
            new_bj = list(bj)
            new_bi[pi] = y
            new_bj[pj] = x
            
            # Check for duplicates
            if len(set(new_bi)) < k or len(set(new_bj)) < k:
                continue
            
            new_bi.sort()
            new_bj.sort()
            
            # Compute new coverages
            old_cov_i = coverages[i]
            old_cov_j = coverages[j]
            new_cov_i = block_coverage(tuple(new_bi))
            new_cov_j = block_coverage(tuple(new_bj))
            
            # Check if this creates a valid covering (no uncovered t-subsets)
            # Old total coverage without i and j
            rest_cov = total_cov & ~old_cov_i & ~old_cov_j
            # Actually we need: is every t-subset still covered?
            # New total coverage
            new_total = rest_cov | new_cov_i | new_cov_j
            # We need new_total == total_cov (or at least covers everything that was covered)
            # But actually we just need full_mask to be covered
            # Check: is there any t-subset in total_cov that's not in new_total?
            lost = total_cov & ~new_total
            gained = new_total & ~total_cov
            
            if lost:
                # This swap loses coverage - only accept with probability
                delta = lost.bit_count() - gained.bit_count()
                if delta > 0 and random.random() < math.exp(-delta / temp):
                    # Accept
                    blocks[i] = new_bi
                    blocks[j] = new_bj
                    coverages[i] = new_cov_i
                    coverages[j] = new_cov_j
                    total_cov = new_total
                continue
            else:
                # No coverage lost
                blocks[i] = new_bi
                blocks[j] = new_bj
                coverages[i] = new_cov_i
                coverages[j] = new_cov_j
                total_cov = new_total
                
                # Try to remove redundant blocks
                if it % 100 == 0 and time.time() < deadline:
                    blocks = remove_redundant(blocks, deadline)
                    coverages = [block_coverage(b) for b in blocks]
                    total_cov = 0
                    for c in coverages:
                        total_cov |= c
                    n_blocks = len(blocks)
                    if n_blocks < best_n:
                        best_n = n_blocks
                        best_blocks = [list(b) for b in blocks]
        
        return best_blocks
    
    if time.time() < deadline:
        blocks = remove_redundant(blocks, deadline)
    
    if time.time() < deadline:
        blocks = simulated_annealing(blocks, deadline)
    
    # Final check: ensure all t-subsets are covered
    if time.time() < deadline:
        final_unc = compute_uncovered_mask(blocks)
        while final_unc != 0 and time.time() < deadline:
            best_gain = 0
            best_ks = None
            for ki in range(n_k):
                gain = (coverage_masks[ki] & final_unc).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_ks = ki
            if best_ks is None or best_gain == 0:
                break
            blocks.append(list(all_k_subs[best_ks]))
            final_unc &= ~coverage_masks[best_ks]
    
    # Deduplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        tb = tuple(sorted(b))
        if tb not in seen:
            seen.add(tb)
            unique_blocks.append(list(tb))
    
    return unique_blocks
