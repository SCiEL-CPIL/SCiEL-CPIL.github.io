import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    # Precompute t-subsets for all possible blocks containing a specific target
    # This is expensive, so we'll compute on the fly but optimize the gain calculation
    
    def greedy_cover():
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to break symmetry
            target = random.choice(list(uncovered))
            
            # Generate candidates containing target
            remaining = [x for x in all_points if x not in target]
            need = k - t
            
            best_gain = -1
            best_block = None
            
            # Optimization: If we just want a valid cover, we can pick any.
            # But we want min blocks, so we pick max gain.
            for combo in itertools.combinations(remaining, need):
                block = tuple(sorted(target + combo))
                # Calculate gain
                gain = 0
                for sub in itertools.combinations(block, t):
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if best_gain == 56: # Max possible gain for k=8, t=5 is C(8,5)=56
                        break
            
            if best_block is None:
                # Should not happen
                break
                
            blocks.append(best_block)
            
            # Update uncovered
            for sub in itertools.combinations(best_block, t):
                if sub in uncovered:
                    uncovered.remove(sub)
                    
            if time.time() > deadline - 10:
                break
                
        return blocks

    def delete_blocks(blocks):
        # Count coverage
        counts = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[sub] = counts.get(sub, 0) + 1
        
        removed_any = True
        while removed_any:
            removed_any = False
            # Iterate in reverse to potentially remove later added (often more redundant) blocks first?
            # Or random order. Let's use random order to avoid bias.
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                b = blocks[i]
                # Check if any subset in b has count 1
                can_remove = True
                for sub in itertools.combinations(b, t):
                    if counts[sub] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    for sub in itertools.combinations(b, t):
                        counts[sub] -= 1
                    blocks[i] = None # Mark for removal
                    removed_any = True
                    if time.time() > deadline - 5:
                        return [b for b in blocks if b is not None]
            
            # Compact list
            if removed_any:
                blocks = [b for b in blocks if b is not None]
        
        return blocks

    def swap_blocks(blocks, max_swaps=50):
        """Try to swap blocks to improve coverage efficiency."""
        if time.time() > deadline - 15:
            return blocks
            
        counts = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[sub] = counts.get(sub, 0) + 1
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 15:
                    break
                b = blocks[i]
                # Try to remove b and see if we can add a better block
                for sub in itertools.combinations(b, t):
                    if counts[sub] == 1:
                        # b is critical, can't remove
                        break
                else:
                    # b is not critical, try to remove it
                    for sub in itertools.combinations(b, t):
                        counts[sub] -= 1
                    
                    # Find a block that covers the most uncovered subsets
                    # Actually, we just need to ensure coverage is maintained
                    # Since b was not critical, coverage is still complete
                    # We can remove b
                    blocks[i] = None
                    improved = True
                    break
            
            # Compact list
            blocks = [b for b in blocks if b is not None]
        
        return blocks

    def local_search(blocks):
        """Perform local search by trying to replace blocks with better ones."""
        if time.time() > deadline - 20:
            return blocks
            
        counts = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[sub] = counts.get(sub, 0) + 1
        
        # Try to find blocks that can be replaced with blocks covering more unique subsets
        for i in range(len(blocks)):
            if time.time() > deadline - 20:
                break
            b = blocks[i]
            # Calculate how many subsets are only covered by b
            unique_count = 0
            for sub in itertools.combinations(b, t):
                if counts[sub] == 1:
                    unique_count += 1
            
            if unique_count > 0:
                # b has unique coverage, try to find a replacement
                # This is complex, so skip for now
                continue
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy instances
    while time.time() < deadline - 30:
        blocks = greedy_cover()
        if not blocks:
            continue
            
        # Try to delete
        blocks = delete_blocks(blocks)
        
        # Try local search improvements
        blocks = swap_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we hit a very good score, we can stop or continue trying to improve
        # The lower bound is 264. Best known is 335.
        # If we get < 335, it's a new record (unlikely but possible).
        # If we get close to 335, keep trying.
        
        # Optional: Local search?
        # For now, just restart greedy with different random seeds.
        
    if best_blocks is None:
        # Fallback to a single deterministic run if something went wrong
        random.seed(42)
        blocks = greedy_cover()
        blocks = delete_blocks(blocks)
        return blocks
        
    return best_blocks
