import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Precomputation
    points = list(range(v))
    all_4subs = list(itertools.combinations(points, t))
    # Map 4-sub to integer mask for fast set operations
    def to_mask(sub):
        m = 0
        for x in sub:
            m |= (1 << x)
        return m
        
    all_4sub_masks = [to_mask(s) for s in all_4subs]
    all_4sub_set = set(all_4sub_masks)
    
    # Generate all blocks (k-subsets)
    all_blocks = list(itertools.combinations(points, k))
    all_block_masks = [to_mask(b) for b in all_blocks]
    
    # Map block mask to list of 4-sub masks it contains
    block_4subs = {}
    # Map 4-sub mask to list of block masks containing it
    sub_to_blocks = defaultdict(list)
    
    for bm in all_block_masks:
        # Extract 4-subs from block
        block_points = [i for i in range(v) if (bm >> i) & 1]
        block_4s = [to_mask(s) for s in itertools.combinations(block_points, t)]
        block_4subs[bm] = block_4s
        for s4 in block_4s:
            sub_to_blocks[s4].append(bm)

    def greedy_cover(seed):
        random.seed(seed)
        uncovered = set(all_4sub_set)
        blocks = []
        cover_count = defaultdict(int) # 4sub mask -> count
        
        while uncovered:
            # Pick a random uncovered 4-sub to anchor the choice
            anchor = random.choice(list(uncovered))
            
            # Get candidate blocks containing the anchor
            candidates = sub_to_blocks[anchor]
            
            best_block = None
            best_gain = -1
            
            # Evaluate candidates
            for bm in candidates:
                # Gain is number of currently uncovered 4-subs this block covers
                # We can approximate or calculate exactly.
                # Exact: len(uncovered & set(block_4subs[bm]))
                # Since uncovered is a set, and block_4subs[bm] is a list, 
                # we can iterate block_4subs[bm] and check membership in uncovered.
                gain = 0
                for s4 in block_4subs[bm]:
                    if s4 in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early termination if we already cover everything? 
                            # Or just keep max.
                            pass
                # Optimization: If gain is 0, skip.
                if gain > best_gain:
                    best_gain = gain
                    best_block = bm
                    # If we cover all uncovered, we can break early? 
                    # No, there might be another block covering more? 
                    # No, max possible is len(uncovered). 
                    if best_gain == len(uncovered):
                        break
            
            if best_block is None:
                # Should not happen if uncovered is not empty
                continue
                
            # Add block
            blocks.append(best_block)
            for s4 in block_4subs[best_block]:
                if s4 in uncovered:
                    uncovered.remove(s4)
                cover_count[s4] += 1
            
            # Check time
            if time.time() > deadline - 10:
                break
        
        return blocks, cover_count, uncovered

    def local_search(blocks, cover_count, uncovered):
        # blocks is list of masks
        # cover_count is dict 4sub->count
        # uncovered should be empty if valid. If not, we can't remove/swap safely without breaking coverage.
        # The greedy loop above stops when uncovered is empty OR time runs out.
        # If time runs out, uncovered might not be empty. 
        # If uncovered is not empty, the solution is INVALID. 
        # We must ensure we return a valid solution.
        # So, if time runs out during greedy, we must finish covering? 
        # Or we can just return the best *valid* solution found in previous iterations.
        # The prompt says: "Valid when every t-subset ... lies in at least one block".
        # So we must only return valid covers.
        
        if uncovered:
            # Try to quickly finish? Or just mark as invalid?
            # Better to just not use this result if it's invalid.
            return None
            
        improved = True
        current_blocks = list(blocks)
        current_cover_count = dict(cover_count)
        
        while improved and (time.time() < deadline - 5):
            improved = False
            
            # Pass 1: Remove redundant blocks
            for i in range(len(current_blocks)):
                if time.time() > deadline - 5:
                    break
                bm = current_blocks[i]
                is_redundant = True
                for s4 in block_4subs[bm]:
                    if current_cover_count[s4] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block i
                    for s4 in block_4subs[bm]:
                        current_cover_count[s4] -= 1
                    current_blocks.pop(i)
                    improved = True
                    # Restart loop as indices changed
                    break
            
            if improved:
                continue
                
            # Pass 2: Swap blocks to potentially enable removals
            # For each block, find essential 4-subs
            for i in range(len(current_blocks)):
                if time.time() > deadline - 5:
                    break
                bm = current_blocks[i]
                essential_4subs = []
                for s4 in block_4subs[bm]:
                    if current_cover_count[s4] == 1:
                        essential_4subs.append(s4)
                
                if not essential_4subs:
                    # Should have been removed in Pass 1
                    continue
                
                # Find candidate blocks that cover all essential_4subs
                # Intersect sub_to_blocks for each essential 4-sub
                candidates = set(sub_to_blocks[essential_4subs[0]])
                for s4 in essential_4subs[1:]:
                    candidates.intersection_update(sub_to_blocks[s4])
                
                # Remove current block from candidates to avoid no-op swap
                candidates.discard(bm)
                
                if candidates:
                    # Pick a random candidate to diversify
                    new_bm = random.choice(list(candidates))
                    
                    # Perform swap
                    # Remove old block
                    for s4 in block_4subs[bm]:
                        current_cover_count[s4] -= 1
                    # Add new block
                    for s4 in block_4subs[new_bm]:
                        current_cover_count[s4] += 1
                    current_blocks[i] = new_bm
                    improved = True
                    # Restart to check for new redundancies
                    break
        
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    seed = 0
    
    while time.time() < deadline - 10:
        blocks, cover_count, uncovered = greedy_cover(seed)
        seed += 1
        
        if not uncovered:
            # Valid cover
            result = local_search(blocks, cover_count, uncovered)
            if result is not None:
                if len(result) < best_len:
                    best_len = len(result)
                    best_blocks = result
                    # If we hit a very good number, we can stop early? 
                    # Lower bound is 131. Best known 151.
                    # If we find < 151, it's a new record!
                    if best_len <= 150:
                         # Keep searching for a bit to see if we can go lower?
                         # Or just stop if we have plenty of time?
                         # Let's continue to try to beat 150.
                         pass
        else:
            # Invalid, skip
            pass

    # If no valid block found (should not happen), fallback to a simple valid one
    if best_blocks is None:
        # Fallback: Greedy without time cut-off (might be slow but valid)
        # Or just return a trivially valid but large set?
        # The greedy above should find a valid one quickly.
        # If it didn't, it means time ran out before completing coverage.
        # We need to ensure at least one valid solution is returned.
        # Let's run a final greedy that MUST complete.
        uncovered = set(all_4sub_set)
        blocks = []
        while uncovered:
            anchor = random.choice(list(uncovered))
            candidates = sub_to_blocks[anchor]
            best_block = None
            best_gain = -1
            for bm in candidates:
                gain = sum(1 for s4 in block_4subs[bm] if s4 in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = bm
            if best_block:
                blocks.append(best_block)
                for s4 in block_4subs[best_block]:
                    if s4 in uncovered:
                        uncovered.remove(s4)
        best_blocks = blocks

    # Convert masks back to lists of ints
    final_blocks = []
    for bm in best_blocks:
        final_blocks.append([i for i in range(v) if (bm >> i) & 1])
        
    return final_blocks

if __name__ == "__main__":
    # Test with task parameters
    blocks = construct(19, 7, 4)
    print(f"Found {len(blocks)} blocks")
