import itertools
import time
import numpy as np

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsub_idxs.append(idxs)
        
    # Convert to numpy arrays for faster operations if needed, but lists are fine for iteration
    # We will use a greedy approach with multiple restarts and local search (ILS)
    
    def greedy_construct():
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        # We can speed up greedy by precomputing gains or using a priority queue,
        # but for v=19, k=7, num_kblocks is 50388. 
        # A simple loop is ~50k * (some ops). 
        # Let's use a vectorized approach for gains if possible, or just efficient python.
        
        # Actually, computing gains A.T @ u every step is expensive if done naively.
        # But num_tsubs is C(19,4) = 3876. num_kblocks is 50388.
        # A is sparse.
        
        # Let's try a simple greedy first to get a baseline solution
        # To speed it up, we can maintain a list of uncovered tsubs.
        
        uncovered_count = num_tsubs
        uncovered_tsubs = set(range(num_tsubs))
        
        while uncovered_count > 0 and time.monotonic() < deadline - 10.0:
            best_gain = -1
            best_block = -1
            
            # Iterate through all blocks? That's 50k iterations.
            # For each block, count how many new tsubs it covers.
            # This is slow in Python if done naively.
            
            # Alternative: Randomized Greedy
            # Pick a random uncovered tsub, then pick a random block covering it?
            # Or pick a block that covers the most uncovered tsubs.
            
            # Let's optimize the gain calculation.
            # We can sample a few blocks or use a heuristic.
            # However, standard greedy is O(N * M) where N is steps, M is blocks.
            # N is roughly 150. M is 50k. 150 * 50k = 7.5M operations. This is fine.
            
            for j in range(num_kblocks):
                # Check if this block adds any new coverage
                gain = 0
                # We need to know which tsubs in this block are currently uncovered.
                # We can check against the 'uncovered' boolean array.
                for idx in block_tsub_idxs[j]:
                    if not covered[idx]:
                        gain += 1
                        if gain > best_gain:
                            # Early termination if we can't beat best?
                            # Max possible gain for a block is C(7,4)=35.
                            # If best_gain is already 35, we can't beat it.
                            if gain == 35: 
                                break
                if gain > best_gain:
                    best_gain = gain
                    best_block = j
                    
            if best_block == -1 or best_gain == 0:
                break # No block covers new tsubs (should not happen if not covered)
                
            selected.append(best_block)
            # Update covered
            for idx in block_tsub_idxs[best_block]:
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
                    # Optional: remove from uncovered_tsubs set if we used it
            
        return selected

    def polish(selected):
        # Remove redundant blocks
        if not selected:
            return selected
            
        counts = np.zeros(num_tsubs, dtype=int)
        for b_idx in selected:
            for idx in block_tsub_idxs[b_idx]:
                counts[idx] += 1
                
        selected_set = set(selected)
        changed = True
        while changed and time.monotonic() < deadline - 5.0:
            changed = False
            # Iterate in reverse to remove from end, or copy list
            # We want to remove blocks that are not critical.
            # A block is critical if it is the only one covering some tsub.
            # So if for all tsubs in block, counts[tsub] > 1, we can remove it.
            
            # To make it faster, we can just iterate through selected.
            new_selected = []
            for b_idx in selected:
                # Check if we can remove b_idx
                can_remove = True
                for idx in block_tsub_idxs[b_idx]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove b_idx
                    for idx in block_tsub_idxs[b_idx]:
                        counts[idx] -= 1
                    changed = True
                else:
                    new_selected.append(b_idx)
                    
            selected = new_selected
            if not changed:
                break
                
        return selected

    def local_search(selected):
        # Try to swap blocks to improve (reduce count or maintain count with better structure for future swaps)
        # For covering designs, we usually just want to minimize count.
        # A move: remove one block, add one block.
        # If removing a block leaves some tsubs uncovered, we need to add a block that covers ALL of them?
        # No, adding one block might not cover all newly uncovered tsubs if they are spread out.
        # But if we remove block B1, the uncovered tsubs are those that were ONLY covered by B1.
        # Let U be the set of tsubs only covered by B1.
        # We need to find a block B2 that covers all of U.
        # If such a B2 exists and B2 is not already in the set (or is, but we are swapping), we can swap.
        # Actually, if B2 is already in the set, we just removed B1 and B2 stays. That's a pure deletion.
        # If B2 is new, we swap B1 for B2. Count stays same.
        # This doesn't reduce count directly unless we can delete.
        
        # Better: 1-for-1 swap to find a configuration where a deletion is possible?
        # Or just try to delete. The polish step handles deletion.
        # So local search should try to enable deletion.
        # Strategy: Pick a block B1. Calculate U (uncovered if B1 removed).
        # Find a block B2 that covers U.
        # If B2 exists, swap B1 -> B2.
        # Then try to delete again.
        
        # This is expensive. Let's do a simpler ILS:
        # 1. Polish
        # 2. Perturb: Randomly remove 1-2 blocks, then greedily fill gaps.
        # 3. Polish
        # 4. Keep best
        
        best_selected = polish(selected[:])
        best_len = len(best_selected)
        
        current_selected = best_selected[:]
        
        no_improve = 0
        max_no_improve = 50
        
        while time.monotonic() < deadline - 2.0 and no_improve < max_no_improve:
            # Perturb
            # Randomly remove 1 or 2 blocks
            num_remove = 1 if len(current_selected) > 20 else 1 # Be careful not to break coverage too badly
            # Actually, removing blocks breaks coverage. We must repair.
            
            # Let's try a "kick": Remove a random block, then add a random block that helps?
            # Or: Remove a random block, then run greedy to repair.
            
            candidate = current_selected[:]
            
            # Remove 1 random block
            if candidate:
                rem_idx = np.random.randint(len(candidate))
                rem_block = candidate.pop(rem_idx)
                
                # Now some tsubs might be uncovered.
                # Run a fast greedy repair
                covered = np.zeros(num_tsubs, dtype=bool)
                for b_idx in candidate:
                    for idx in block_tsub_idxs[b_idx]:
                        covered[idx] = True
                        
                # Greedily add blocks to cover uncovered
                # We can reuse the greedy logic but only add until covered
                # To speed up, we can just pick random blocks that cover uncovered?
                # Or standard greedy.
                
                # Let's do a few steps of greedy
                for _ in range(5): # Limit repair steps to keep it fast
                    if np.all(covered):
                        break
                    best_gain = -1
                    best_block = -1
                    # Only consider blocks not in candidate? Or any block?
                    # Any block is fine, but we should avoid adding blocks already in candidate (no gain).
                    for j in range(num_kblocks):
                        if j in candidate: # This check is slow (list search). Use set?
                            continue
                        gain = 0
                        for idx in block_tsub_idxs[j]:
                            if not covered[idx]:
                                gain += 1
                                if gain > best_gain:
                                    if gain == 35: break
                        if gain > best_gain:
                            best_gain = gain
                            best_block = j
                            
                    if best_block == -1 or best_gain == 0:
                        break
                    candidate.append(best_block)
                    for idx in block_tsub_idxs[best_block]:
                        covered[idx] = True

            # Polish the candidate
            candidate = polish(candidate)
            
            if len(candidate) < best_len:
                best_selected = candidate[:]
                best_len = len(candidate)
                current_selected = candidate[:]
                no_improve = 0
            else:
                no_improve += 1
                # Accept with some probability? Or just reject?
                # For covering, we only care about size. So reject if not smaller.
                # To escape local minima, we might need to accept worse solutions?
                # But "worse" means larger, which is bad.
                # However, a larger solution might allow a better swap later?
                # Unlikely to help if we always polish.
                # Let's stick to hill climbing on size.
                
                # If no improvement, perturb again from current_selected (which is best so far? or last?)
                # If we reject, we should probably restart from best_selected to avoid drifting.
                # Or keep current_selected as the one we perturbed?
                # Let's restart from best_selected for stability.
                current_selected = best_selected[:]
                
        return best_selected

    # Main Loop
    # 1. Initial Greedy
    initial = greedy_construct()
    initial = polish(initial)
    
    # 2. Local Search / ILS
    best = local_search(initial)
    
    # 3. Try a few more restarts if time permits
    # Greedy is deterministic? No, if we add randomness.
    # Let's add randomness to the greedy selection to allow restarts.
    
    def greedy_construct_randomized():
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        
        while time.monotonic() < deadline - 10.0:
            if np.all(covered):
                break
                
            # Calculate gains
            gains = np.zeros(num_kblocks)
            # This is the bottleneck.
            # Let's try to speed it up.
            # For each tsub not covered, find blocks containing it?
            # No, we need gain per block.
            
            # Alternative: Randomized Greedy
            # Pick a random uncovered tsub.
            # Find all blocks covering it.
            # Pick one at random (or weighted by gain).
            
            # This is much faster! O(1) per step roughly.
            
            # Build reverse index: tsub -> list of blocks
            # This takes time to build, but we can do it once.
            
        # Let's build reverse index outside
        return selected

    # Let's implement the reverse index approach for faster greedy restarts.
    # tsub_to_blocks = [list of block indices for each tsub]
    
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for j in range(num_kblocks):
        for idx in block_tsub_idxs[j]:
            tsub_to_blocks[idx].append(j)
            
    def greedy_construct_fast():
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        selected_set = set()
        
        while time.monotonic() < deadline - 10.0:
            # Find uncovered tsubs
            # We can keep a list of uncovered tsubs
            # Initially all
            # When we add a block, we remove its tsubs from uncovered list.
            
            # To avoid scanning all tsubs every time, maintain a list.
            pass
        
        # Let's just use the standard greedy but with the reverse index to speed up gain calculation?
        # No, the standard greedy scans all blocks.
        # The "random uncovered tsub" method is a heuristic (Set Cover Greedy variant).
        # It is often good enough.
        
        uncovered = set(range(num_tsubs))
        
        while uncovered and time.monotonic() < deadline - 10.0:
            # Pick a random uncovered tsub
            # To be faster, pick from a list
            # Converting set to list is O(N).
            # Let's maintain a list of uncovered tsubs.
            
            # This is getting complicated to maintain dynamically.
            # Let's stick to the previous greedy but maybe limit the number of restarts.
            
            break # Placeholder
            
        return selected

    # Let's try a different strategy for restarts:
    # Use the "Randomized Greedy" where at each step we pick a block from the top K% of gains.
    
    def greedy_construct_topk(K=10):
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        
        while time.monotonic() < deadline - 10.0:
            if np.all(covered):
                break
                
            # Calculate gains
            gains = np.zeros(num_kblocks)
            for j in range(num_kblocks):
                gain = 0
                for idx in block_tsub_idxs[j]:
                    if not covered[idx]:
                        gain += 1
                gains[j] = gain
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find blocks with gain close to max
            # Threshold: max_gain - 2 ? Or just top K?
            # Let's pick randomly from blocks with gain >= max_gain - 1
            candidates = np.where(gains >= max_gain - 1)[0]
            
            if len(candidates) == 0:
                candidates = np.where(gains == max_gain)[0]
                
            best_block = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_block)
            for idx in block_tsub_idxs[best_block]:
                covered[idx] = True
                
        return selected

    # Run a few restarts
    best = local_search(best)
    
    num_restarts = 5
    for i in range(num_restarts):
        if time.monotonic() > deadline - 20.0:
            break
        candidate = greedy_construct_topk(10)
        candidate = polish(candidate)
        candidate = local_search(candidate)
        
        if len(candidate) < len(best):
            best = candidate
            
    # Final polish
    best = polish(best)
    
    # Convert indices to lists
    result_blocks = [list(all_kblocks[i]) for i in best]
    
    return result_blocks
