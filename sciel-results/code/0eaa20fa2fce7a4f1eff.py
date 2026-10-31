import time
import itertools
import random
import sys

_DEADLINE = 145.0
_START = None

def _elapsed():
    return time.time() - _START

def _precompute(v, k, t):
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    coverage = []
    rev_coverage = [[] for _ in range(n_t)]
    
    for ki, ks in enumerate(k_subsets):
        idxs = [t_index[ts] for ts in itertools.combinations(ks, t)]
        coverage.append(idxs)
        for ti in idxs:
            rev_coverage[ti].append(ki)
            
    return all_t, t_index, k_subsets, coverage, n_t, rev_coverage

def _greedy_construct(coverage, n_t, rng, n_k, deadline, bias=None):
    cnt = [0] * n_t
    blocks = []
    uncovered = n_t
    
    # Precompute gain for each block? No, dynamic.
    # To speed up, we can maintain a list of "active" blocks? 
    # For n_k ~ 50k, n_t ~ 3876, scanning all n_k is 50k * 35 (avg t per k) = 1.75M ops per step.
    # 150 blocks * 1.75M = 262M ops. Might be slow in Python.
    # Optimization: Only consider blocks that cover at least one uncovered t-subset.
    
    while uncovered > 0 and _elapsed() < deadline:
        best_i = -1
        best_gain = 0
        best_bias_val = -1
        
        # Optimization: If we had a list of uncovered t's, we could iterate them and find blocks covering them.
        # But maintaining that is complex. Let's stick to scanning but break early if possible?
        # Actually, for v=19, k=7, t=4, n_k = 50388.
        # Scanning 50k blocks is fast if the inner loop is tight.
        
        for i in range(n_k):
            g = 0
            for idx in coverage[i]:
                if cnt[idx] == 0:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_i = i
                best_bias_val = bias.get(i, 0) if bias else 0
            elif g == best_gain and g > 0:
                b_val = bias.get(i, 0) if bias else 0
                if b_val > best_bias_val:
                    best_bias_val = b_val
                    best_i = i
        
        if best_i < 0:
            break
        
        blocks.append(best_i)
        for idx in coverage[best_i]:
            if cnt[idx] == 0:
                uncovered -= 1
            cnt[idx] += 1
            
    return blocks, cnt, uncovered

def _local_search(blocks, coverage, rev_coverage, n_t, n_k, rng, deadline, bias=None):
    blocks = list(blocks)
    cnt = [0] * n_t
    for b in blocks:
        for idx in coverage[b]:
            cnt[idx] += 1
            
    if any(c == 0 for c in cnt):
        return blocks, cnt
        
    in_blocks = set(blocks)
    
    # Precompute "redundancy" of each block in current solution?
    # redundant[b] = number of t-subsets in b that have cnt > 1.
    
    improved = True
    while improved and _elapsed() < deadline:
        improved = False
        
        # 1. Removal Pass
        i = 0
        while i < len(blocks) and _elapsed() < deadline:
            b = blocks[i]
            removable = True
            for idx in coverage[b]:
                if cnt[idx] == 1:
                    removable = False
                    break
            if removable:
                for idx in coverage[b]:
                    cnt[idx] -= 1
                blocks.pop(i)
                in_blocks.discard(b)
                improved = True
                continue
            i += 1
            
        if improved:
            continue # Restart loop to check for more removals
        
        # 2. Targeted Swap Pass
        # Identify blocks with low "vulnerability" (high redundancy) to be candidates for removal (B_out)
        # Identify blocks with high "coverage of vulnerable" to be candidates for addition (B_in)
        
        # Calculate vulnerability score for blocks in solution
        # vuln[b] = number of t-subsets in b with cnt == 1
        # redun[b] = number of t-subsets in b with cnt > 1
        
        vuln_scores = [0] * len(blocks)
        for i, b in enumerate(blocks):
            v = 0
            for idx in coverage[b]:
                if cnt[idx] == 1:
                    v += 1
            vuln_scores[i] = v
            
        # Sort block indices by vuln score ascending (most redundant first)
        # We want to remove blocks with fewest cnt==1 t-subsets.
        # But we need to map back to block ID.
        
        # Candidates for B_in:
        # Pick a random vulnerable t-subset (cnt==1)
        vulnerable_ts = [i for i in range(n_t) if cnt[i] == 1]
        
        if not vulnerable_ts:
            break # No vulnerable t-subsets, likely optimal or stuck in a way that requires 2-for-1? 
            # If no cnt==1, all cnt>=2. Then any block is removable? 
            # If all cnt>=2, then removing any block leaves cnt>=1. So Removal Pass should have caught this.
            # So if we are here, there must be cnt==1.
            
        vt = vulnerable_ts[rng.randrange(len(vulnerable_ts))]
        
        # Find blocks covering vt that are NOT in solution
        candidates_in = [c for c in rev_coverage[vt] if c not in in_blocks]
        
        if not candidates_in:
            # Try another vulnerable t
            # Or just break?
            # Let's try a few random vulnerable t's
            found = False
            for _ in range(5):
                vt = vulnerable_ts[rng.randrange(len(vulnerable_ts))]
                candidates_in = [c for c in rev_coverage[vt] if c not in in_blocks]
                if candidates_in:
                    found = True
                    break
            if not found:
                break

        # Score candidates_in by how many vulnerable t's they cover
        # We want B_in to cover many cnt==1 t's.
        best_in = -1
        best_in_score = -1
        for c in candidates_in:
            score = 0
            for idx in coverage[c]:
                if cnt[idx] == 1:
                    score += 1
            if score > best_in_score:
                best_in_score = score
                best_in = c
                
        if best_in < 0:
            continue

        # Now choose B_out.
        # We want to remove a block that is "redundant".
        # Sort current blocks by vuln_scores (ascending).
        # Pick one of the top few (most redundant).
        
        # To avoid O(N log N) every time, we can just pick a random block with low vuln score.
        # Or, since we computed vuln_scores, let's just pick the best one.
        # But maybe the "best" one doesn't lead to a removal.
        # Let's try the top 5 most redundant blocks.
        
        # Get indices of blocks in solution
        # vuln_scores is aligned with blocks list.
        
        # Find indices with lowest vuln scores
        # Use a simple selection for top 5
        # Or just random sample from the lower half?
        
        # Let's try the block with the absolute lowest vuln score.
        # If it fails, try the next, etc.
        
        # Sort indices by vuln score
        # This is O(N log N). N is ~150. Fast enough.
        indices_sorted = sorted(range(len(blocks)), key=lambda i: vuln_scores[i])
        
        swap_success = False
        for out_idx in indices_sorted[:10]: # Try top 10 most redundant
            b_out = blocks[out_idx]
            
            # Tentative Swap
            # Remove b_out
            for idx in coverage[b_out]:
                cnt[idx] -= 1
            
            # Add b_in
            for idx in coverage[best_in]:
                cnt[idx] += 1
            
            # Check validity: all cnt > 0?
            valid = True
            for idx in range(n_t):
                if cnt[idx] == 0:
                    valid = False
                    break
            
            if not valid:
                # Revert
                for idx in coverage[best_in]:
                    cnt[idx] -= 1
                for idx in coverage[b_out]:
                    cnt[idx] += 1
                continue
            
            # Check if any block is now removable
            # We only need to check blocks that might have become removable.
            # Actually, counts changed. Any block could be removable.
            # But specifically, blocks that share t-subsets with b_in or b_out are most likely.
            # For simplicity, scan all blocks.
            
            removable_found = False
            for i in range(len(blocks)):
                b = blocks[i]
                rem = True
                for idx in coverage[b]:
                    if cnt[idx] == 1:
                        rem = False
                        break
                if rem:
                    # Remove it
                    for idx in coverage[b]:
                        cnt[idx] -= 1
                    blocks.pop(i)
                    in_blocks.discard(b)
                    # Update the swap: we removed b, but we also swapped b_out -> b_in.
                    # Wait, the swap was tentative. 
                    # If we remove b, the state is:
                    # Original blocks - {b_out, b} + {b_in}
                    # We need to update the blocks list correctly.
                    
                    # The blocks list currently has b_out removed? No, we didn't pop it yet.
                    # We just modified cnt.
                    # We need to:
                    # 1. Pop b (the removable one).
                    # 2. Replace b_out with b_in.
                    
                    # Find b_out in blocks
                    # It might have shifted if b was before it?
                    # We are iterating i.
                    
                    # Let's just rebuild the state cleanly.
                    
                    # Revert tentative swap
                    for idx in coverage[best_in]:
                        cnt[idx] -= 1
                    for idx in coverage[b_out]:
                        cnt[idx] += 1
                    
                    # Perform actual swap: b_out -> b_in
                    blocks[out_idx] = best_in
                    in_blocks.discard(b_out)
                    in_blocks.add(best_in)
                    
                    for idx in coverage[b_out]:
                        cnt[idx] -= 1
                    for idx in coverage[best_in]:
                        cnt[idx] += 1
                    
                    # Now remove b
                    # Find b in blocks
                    b_idx = blocks.index(b)
                    for idx in coverage[b]:
                        cnt[idx] -= 1
                    blocks.pop(b_idx)
                    in_blocks.discard(b)
                    
                    improved = True
                    swap_success = True
                    break
            
            if swap_success:
                break
            
            # If no removal found, revert tentative swap
            for idx in coverage[best_in]:
                cnt[idx] -= 1
            for idx in coverage[b_out]:
                cnt[idx] += 1
        
        if not swap_success:
            # Try a few random swaps if targeted failed?
            # Or just break to avoid infinite loop if no progress.
            # To ensure progress, we can allow a "random" swap that doesn't require immediate removal 
            # if it improves a metric, but that's risky.
            # Let's just break.
            break
            
    return blocks, cnt

def construct(v, k, t):
    global _START
    _START = time.time()

    all_t, t_index, k_subsets, coverage, n_t, rev_coverage = _precompute(v, k, t)
    n_k = len(k_subsets)

    best_blocks = None
    best_size = float('inf')
    global_freq = {}

    seeds = [12345, 67890, 24680, 13579, 99999, 55555, 11111, 77777, 135791, 246802]
    
    for seed in seeds:
        if _elapsed() > _DEADLINE - 40:
            break
        rng = random.Random(seed)
        
        bias = dict(global_freq)
        
        blocks, cnt, uncovered = _greedy_construct(coverage, n_t, rng, n_k, _DEADLINE - 30, bias)
        
        if uncovered > 0:
            # Repair
            while uncovered > 0 and _elapsed() < _DEADLINE - 10:
                best_i = -1
                best_gain = 0
                for i in range(n_k):
                    g = 0
                    for idx in coverage[i]:
                        if cnt[idx] == 0:
                            g += 1
                    if g > best_gain:
                        best_gain = g
                        best_i = i
                if best_i < 0:
                    break
                blocks.append(best_i)
                for idx in coverage[best_i]:
                    if cnt[idx] == 0:
                        uncovered -= 1
                    cnt[idx] += 1
        
        if uncovered > 0:
            continue

        ls_blocks, ls_cnt = _local_search(blocks, coverage, rev_coverage, n_t, n_k, rng, _DEADLINE - 10, bias)
        
        if any(c == 0 for c in ls_cnt):
            ls_blocks = blocks
        
        if len(ls_blocks) < best_size:
            best_size = len(ls_blocks)
            best_blocks = list(ls_blocks)
            for b in ls_blocks:
                global_freq[b] = global_freq.get(b, 0) + 1

    if best_blocks is None:
        rng = random.Random(12345)
        blocks, cnt, uncovered = _greedy_construct(coverage, n_t, rng, n_k, _DEADLINE - 5)
        best_blocks = blocks
        best_size = len(blocks)

    # Final Polish
    if _elapsed() < _DEADLINE - 10 and best_blocks is not None:
        cnt_best = [0] * n_t
        for b in best_blocks:
            for idx in coverage[b]:
                cnt_best[idx] += 1
        
        if all(c > 0 for c in cnt_best):
            # Run local search again on the best solution
            # Use a different seed for rng in LS
            rng_polish = random.Random(42)
            final_blocks, final_cnt = _local_search(best_blocks, coverage, rev_coverage, n_t, n_k, rng_polish, _DEADLINE - 2)
            
            if all(c > 0 for c in final_cnt) and len(final_blocks) < len(best_blocks):
                best_blocks = final_blocks
                best_size = len(final_blocks)

    # Ensure validity
    cnt = [0] * n_t
    for b in best_blocks:
        for idx in coverage[b]:
            cnt[idx] += 1
    if any(c == 0 for c in cnt):
        # Repair
        while any(c == 0 for c in cnt) and _elapsed() < _DEADLINE - 2:
            best_i = -1
            best_gain = 0
            for i in range(n_k):
                g = 0
                for idx in coverage[i]:
                    if cnt[idx] == 0:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_i = i
            if best_i < 0:
                break
            best_blocks.append(best_i)
            for idx in coverage[best_i]:
                cnt[idx] += 1

    # Deduplicate and return
    seen = set()
    dedup = []
    for b in best_blocks:
        key = k_subsets[b]
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    start = time.time()
    result = construct(v, k, t)
    end = time.time()
    print(f"Result size: {len(result)}, Time: {end-start:.2f}s")
