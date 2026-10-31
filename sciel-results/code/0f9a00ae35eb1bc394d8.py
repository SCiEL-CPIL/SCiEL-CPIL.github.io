import itertools
import random
import time

def construct(v, k, t):
    # Pre-compute all t-subsets and their indices for fast counting
    # v=19, k=7, t=4
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Pre-compute for all possible k-blocks, the list of t-subset indices they cover
    # There are C(19,7) = 50388 blocks. 
    # Pre-computing this might take some time but is worth it for speed.
    all_kblocks = list(itertools.combinations(range(v), k))
    block_to_tsubs = {}
    for b in all_kblocks:
        tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_to_tsubs[b] = tsubs

    # 1. Greedy Initialization
    # uncovered_mask or just a set of indices? 
    # Using a list of counts is faster for updates.
    # Start with 0 counts.
    cover_count = [0] * num_tsubs
    uncovered = set(range(num_tsubs))
    
    blocks = []
    
    # Greedy: pick block covering most uncovered
    while uncovered:
        best_block = None
        max_cov = -1
        # Optimization: We can't iterate all 50388 blocks every step if it's slow.
        # But 50388 * 35 (tsubs per block) is ~1.7M ops. 
        # Number of blocks in solution ~150. 150 * 1.7M = 255M ops. Might be too slow in Python.
        
        # Alternative Greedy: 
        # Pick an uncovered t-subset. Find all k-blocks containing it.
        # Among those, pick the one that covers the most *other* uncovered t-subsets.
        
        # Let's pre-compute: for each t-subset, which k-blocks contain it?
        # This is the inverse map.
        # tsub -> list of k-block indices
        tsub_to_blocks = [[] for _ in range(num_tsubs)]
        for i, b in enumerate(all_kblocks):
            for idx in block_to_tsubs[b]:
                tsub_to_blocks[idx].append(i)
                
        # Greedy Loop
        while uncovered:
            # Pick a random uncovered t-subset to anchor
            target_tsub = random.choice(list(uncovered))
            
            best_block_idx = -1
            best_gain = -1
            
            # Candidates are blocks containing target_tsub
            candidates = tsub_to_blocks[target_tsub]
            
            for b_idx in candidates:
                b = all_kblocks[b_idx]
                # Count how many of its t-subsets are currently uncovered
                gain = 0
                for idx in block_to_tsubs[b]:
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                b = all_kblocks[best_block_idx]
                blocks.append(b)
                # Update counts and uncovered
                for idx in block_to_tsubs[b]:
                    cover_count[idx] += 1
                    if cover_count[idx] == 1:
                        uncovered.discard(idx)
            else:
                break # Should not happen

    # Now we have an initial valid cover.
    # 2. Local Search with Simulated Annealing / Iterated Local Search
    # We will try to reduce the number of blocks.
    
    # Helper: check if a block is redundant
    def is_redundant(block):
        for idx in block_to_tsubs[block]:
            if cover_count[idx] <= 1:
                return False
        return True

    def remove_block(block):
        for idx in block_to_tsubs[block]:
            cover_count[idx] -= 1
            if cover_count[idx] == 0:
                uncovered.add(idx)
        blocks.remove(block)

    def add_block(block):
        for idx in block_to_tsubs[block]:
            cover_count[idx] += 1
            if cover_count[idx] == 1:
                uncovered.discard(idx)
        blocks.append(block)

    def clean_redundants():
        changed = True
        while changed:
            changed = False
            # Iterate backwards to remove safely
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(blocks[i]):
                    b = blocks[i]
                    # Remove manually to avoid list lookup cost in remove()
                    blocks.pop(i)
                    for idx in block_to_tsubs[b]:
                        cover_count[idx] -= 1
                        if cover_count[idx] == 0:
                            uncovered.add(idx)
                    changed = True
                    break # Restart scan? Or continue? 
                    # Restarting is safer for finding all, but slower.
                    # Let's just break and re-loop.

    # Initial cleanup
    clean_redundants()
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Moves:
    # 1. Try to remove a random block. If valid, accept. If invalid, try to repair.
    # 2. Repair: Find block that covers most uncovered. Add it.
    # 3. Then clean redundants.
    
    while time.time() - start_time < timeout:
        if len(blocks) == 0: break
        
        # Pick random block to remove
        i = random.randint(0, len(blocks) - 1)
        b_old = blocks[i]
        
        # Remove it
        blocks.pop(i)
        for idx in block_to_tsubs[b_old]:
            cover_count[idx] -= 1
            if cover_count[idx] == 0:
                uncovered.add(idx)
                
        if not uncovered:
            # Valid and smaller (or same size if we didn't add anything yet)
            # Actually size decreased by 1.
            clean_redundants()
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
        else:
            # Invalid. Need to repair.
            # Find best block to add
            # Strategy: Pick a random uncovered t-subset, find best block covering it.
            # Or: Pick block that maximizes coverage of all uncovered.
            # The latter is expensive. Let's use the anchor strategy again.
            
            # To make it stochastic (SA-like), we might accept a "worse" repair?
            # No, let's just do greedy repair for now.
            
            target_tsub = random.choice(list(uncovered))
            best_b_idx = -1
            best_gain = -1
            candidates = tsub_to_blocks[target_tsub]
            
            for b_idx in candidates:
                # Skip if we just removed it? No, we can re-add it.
                b = all_kblocks[b_idx]
                gain = 0
                for idx in block_to_tsubs[b]:
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
                    
            if best_b_idx != -1:
                b_new = all_kblocks[best_b_idx]
                blocks.append(b_new)
                for idx in block_to_tsubs[b_new]:
                    cover_count[idx] += 1
                    if cover_count[idx] == 1:
                        uncovered.discard(idx)
                        
                # Now try to clean up
                clean_redundants()
                
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)
                elif len(blocks) == best_len:
                    # Tie, maybe keep if we want diversity? 
                    # For now, just keep current if not worse.
                    pass
                else:
                    # Worse. 
                    # In pure ILS, we revert. In SA, we might accept.
                    # Let's revert to keep it simple and stable, but add a bit of randomness.
                    # If we revert, we are back to previous state.
                    # To escape local minima, we need to accept worse states sometimes.
                    # But "worse" here means more blocks.
                    # If we accept a state with more blocks, we risk growing.
                    # Let's stick to "Accept if better, or with small probability if equal/worse".
                    # Actually, the "clean_redundants" step is powerful. 
                    # If we add a block and then remove 2 redundants, we improved by 1.
                    # So even if len(blocks) increased temporarily, the final clean might be good.
                    
                    # The logic above already does clean_redundants.
                    # So len(blocks) is the final size after repair+clean.
                    # If it's worse than best, we should probably revert to best?
                    # No, ILS usually keeps the current state and continues, hoping to find a way out.
                    # But if we just reverted, we are stuck.
                    # Let's accept the current state (which might be worse than best) to explore.
                    # We track best_blocks separately.
                    pass
                    
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            # Or restart with higher T?
            # Let's just keep it low.
            
    return best_blocks
