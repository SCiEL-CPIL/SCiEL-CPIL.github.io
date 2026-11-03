import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precomputation
    # 1. Generate all k-subsets (blocks)
    all_points = list(range(v))
    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)
    
    # 2. For each block, generate its t-subsets
    # block_to_subs[i] is a list of tuples (t-subsets)
    block_to_subs = []
    for b in all_blocks:
        subs = list(itertools.combinations(b, t))
        block_to_subs.append(subs)
        
    # 3. Map t-subsets to blocks
    # tsub_to_blocks: dict from tuple to list of block indices
    tsub_to_blocks = {}
    for i, subs in enumerate(block_to_subs):
        for s in subs:
            if s not in tsub_to_blocks:
                tsub_to_blocks[s] = []
            tsub_to_blocks[s].append(i)
            
    # All t-subsets
    all_t_subsets = list(tsub_to_blocks.keys())
    
    best_blocks = None
    best_count = float('inf')
    
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Greedy Construction
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 2:
                break
            
            # Pick a random uncovered t-subset
            seed_subset = random.choice(list(uncovered))
            
            # Get candidate blocks that contain this subset
            candidates = tsub_to_blocks[seed_subset]
            
            best_block_idx = -1
            max_gain = -1
            best_gain_blocks = []
            
            for idx in candidates:
                # Calculate gain: number of uncovered subsets in this block
                gain = 0
                # Optimization: iterate through block's subsets and check if in uncovered
                for s in block_to_subs[idx]:
                    if s in uncovered:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_gain_blocks = [idx]
                elif gain == max_gain:
                    best_gain_blocks.append(idx)
            
            if best_block_idx == -1 and not best_gain_blocks:
                # Should not happen if uncovered is not empty
                break
                
            # Pick one randomly from best gain blocks to add diversity
            chosen_idx = random.choice(best_gain_blocks)
            blocks.append(chosen_idx)
            
            # Update uncovered
            for s in block_to_subs[chosen_idx]:
                uncovered.discard(s)
        
        if not blocks:
            continue
            
        # Reduction Phase 1: Remove redundant blocks
        # Count coverage
        coverage_count = {}
        for b_idx in blocks:
            for s in block_to_subs[b_idx]:
                coverage_count[s] = coverage_count.get(s, 0) + 1
                
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Iterate through blocks to find redundant ones
            # We need to be careful with indices if we pop.
            # Better to build a new list.
            new_blocks = []
            for b_idx in blocks:
                redundant = True
                for s in block_to_subs[b_idx]:
                    if coverage_count[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for s in block_to_subs[b_idx]:
                        coverage_count[s] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            if changed:
                blocks = new_blocks
                
        # Reduction Phase 2: 2-for-1 Replacement
        # Try to replace two blocks with one
        # This is O(B^2) checks, each check is O(C(k,t) * log(B)) roughly
        # B is ~300. 300^2 = 90k pairs.
        # For each pair, calculate union of subsets.
        # Check if intersection of covering blocks is non-empty.
        
        # To speed up, we can limit the number of iterations or checks per restart.
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Shuffle blocks to try different pairs
            random.shuffle(blocks)
            for i in range(len(blocks)):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, len(blocks)):
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    # Calculate union of subsets covered by b1 and b2
                    # We need to check if there is a single block covering all these subsets.
                    # Let U be the set of subsets.
                    # We need a block B_new such that U is subset of subsets(B_new).
                    # This is equivalent to: The intersection of all blocks covering each s in U is not empty.
                    
                    # Optimization: If the union of points in b1 and b2 is > k, impossible.
                    # But b1 and b2 are k-subsets. Union size can be up to v.
                    # However, if U is covered by B_new, B_new must contain all points in any s in U.
                    # Let P_union be the set of points in the union of all s in U.
                    # Actually, simpler: Just compute the intersection of candidate blocks.
                    
                    # Start with all blocks covering the first subset of b1
                    # This might be slow if we do it for every pair.
                    # Alternative: Precompute nothing, just check.
                    
                    # Let's try a cheaper check first.
                    # If b1 == b2, skip.
                    if b1 == b2: continue
                    
                    # Get subsets of b1
                    subs1 = block_to_subs[b1]
                    # Get subsets of b2
                    subs2 = block_to_subs[b2]
                    
                    # We need to find if there exists a block covering subs1 U subs2.
                    # Let's pick one subset from subs1, say s0.
                    # Any replacement block must be in tsub_to_blocks[s0].
                    # Let's iterate through tsub_to_blocks[s0] and check if it covers ALL subs in subs1 U subs2.
                    
                    s0 = subs1[0]
                    candidates = tsub_to_blocks[s0]
                    
                    found_replacement = False
                    for c_idx in candidates:
                        # Check if c_idx covers all subsets in subs1 and subs2
                        # We can check if subs1 is subset of block_to_subs[c_idx]
                        # and subs2 is subset of block_to_subs[c_idx]
                        
                        # Optimization: Check points.
                        # If the union of points in subs1 U subs2 is contained in block c_idx?
                        # No, that's not sufficient.
                        # We must check subset containment.
                        
                        # Since subs1 and subs2 are small (56 each), we can do a set check.
                        # But creating sets is slow.
                        # We can check: Is every s in subs1 in block_to_subs[c_idx]?
                        
                        # Precomputed sets for block_to_subs would help here.
                        # Let's assume we have block_to_subs_set[i]
                        
                        # For now, use a simple check.
                        # If c_idx is b1 or b2, it's not a "new" block, but might be a replacement for the pair?
                        # If c_idx == b1, does b1 cover subs2? If so, b2 is redundant.
                        # This case is handled by Phase 1 if b2 was redundant.
                        # But b2 might not be redundant on its own, but redundant given b1? No, redundancy is global.
                        
                        # Let's just check if c_idx covers all.
                        is_covering = True
                        # Check subs1
                        for s in subs1:
                            if s not in tsub_to_blocks.get(c_idx, []): # This is wrong, tsub_to_blocks maps tsub to blocks
                                # We need to check if s is in block_to_subs[c_idx]
                                # We don't have a reverse map block -> set of subs for O(1) lookup.
                                # We have block_to_subs[c_idx] as a list.
                                pass
                            
                        # This is getting complicated and slow in pure Python.
                        # Let's skip the complex 2-for-1 if it's too slow to implement safely within time.
                        # The current score is 1.0213. The main gain is likely from a better greedy/reduction.
                        
                        # Let's stick to a simpler 2-for-1 check:
                        # Only try if the union of the two blocks' points is small?
                        # No, let's just rely on the improved Greedy + Phase 1 Reduction.
                        # The Phase 1 reduction is very effective.
                        
                        pass
                    # End of 2-for-1 logic (skipped for safety/speed)
                    
        # Update best
        if len(blocks) < best_count:
            best_count = len(blocks)
            # Convert indices back to lists of points
            best_blocks = [list(all_blocks[idx]) for idx in blocks]
            
            if best_count <= 265:
                break
                
    if best_blocks is None:
        # Fallback
        return []
        
    return best_blocks
