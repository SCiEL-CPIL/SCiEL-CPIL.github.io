import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using a randomized greedy
    set cover approach with bitmask-based acceleration and local search.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin for 150s limit
    
    # Precompute all t-subsets (the universe to be covered)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all blocks and their coverage
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, store the bitmask of covered t-subsets
    # Using Python integers as bitmasks (2024 bits fits in a Python int)
    block_mask = [0] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
            t_to_blocks[idx].append(i)
        block_mask[i] = mask
    
    # Full mask representing all t-subsets
    full_mask = (1 << num_t_subsets) - 1
    
    def count_bits(mask):
        """Count set bits in an integer mask."""
        return bin(mask).count('1')
    
    def run_greedy(seed):
        """Run randomized greedy set cover using bitmask operations."""
        random.seed(seed)
        uncovered = full_mask
        selected = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to focus on
            # Find set bits in uncovered, pick one randomly
            # For efficiency, we can use the lowest set bit or random
            # Let's use random: get list of set bit positions
            # Actually, for speed, let's just iterate through set bits
            # We need a random one. Let's get all positions.
            
            # Fast way to get random set bit position:
            # Convert to list of positions
            positions = []
            m = uncovered
            while m:
                # Find lowest set bit
                lb = m & (-m)
                pos = lb.bit_length() - 1
                positions.append(pos)
                m ^= lb
            
            if not positions:
                break
                
            target_idx = random.choice(positions)
            candidates = t_to_blocks[target_idx]
            
            best_block = -1
            best_gain = -1
            
            for b_idx in candidates:
                # Gain = number of uncovered t-subsets covered by this block
                gain_mask = block_mask[b_idx] & uncovered
                gain = count_bits(gain_mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected.append(best_block)
            uncovered &= ~block_mask[best_block]
        
        return selected
    
    def try_remove_blocks(blocks):
        """Remove redundant blocks using bitmask operations."""
        if not blocks:
            return blocks
        
        # Compute combined coverage of all blocks
        # A block is redundant if its mask is a subset of the combined mask of others
        
        # Build coverage count using bitmasks
        # For each t-subset, count how many blocks cover it
        cover_count = [0] * num_t_subsets
        for b_idx in blocks:
            m = block_mask[b_idx]
            # Count bits: for each set bit in m, increment cover_count
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                cover_count[pos] += 1
                m ^= lb
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                
                b_idx = blocks[i]
                m = block_mask[b_idx]
                is_redundant = True
                # Check if any t-subset covered by this block has count <= 1
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    if cover_count[pos] <= 1:
                        is_redundant = False
                        break
                    m ^= lb
                
                if is_redundant:
                    # Remove this block
                    m = block_mask[b_idx]
                    while m:
                        lb = m & (-m)
                        pos = lb.bit_length() - 1
                        cover_count[pos] -= 1
                        m ^= lb
                    blocks.pop(i)
                    changed = True
                    break
        
        return blocks
    
    def local_search(blocks, max_swaps=50):
        """Try to improve by swapping one block for another to reduce size."""
        if not blocks or len(blocks) <= 1:
            return blocks
        
        current = list(blocks)
        
        for _ in range(max_swaps):
            if time.time() > deadline - 10:
                break
            
            improved = False
            n = len(current)
            
            # Try removing each block and see if we can re-cover the lost subsets
            # with fewer blocks or same count but different blocks
            for i in range(n):
                if time.time() > deadline - 10:
                    break
                
                # Compute coverage without block i
                combined = 0
                for j in range(n):
                    if j != i:
                        combined |= block_mask[current[j]]
                
                lost = block_mask[current[i]] & ~combined
                
                if not lost:
                    # Block i is redundant
                    current.pop(i)
                    improved = True
                    break
                
                # Try to find a single block that covers all lost subsets
                # AND is not already in the solution
                # This would be a swap: remove i, add new block
                # The new block must cover 'lost' and we need to check if it's worth it
                # Since we're removing one and adding one, size stays the same
                # So we only do this if it enables future removals
                # For now, let's just do redundancy removal
                # Swap logic would be more complex; stick with removal for reliability
        
        return current
    
    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    # Try multiple seeds
    seeds = list(range(30))
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        
        greedy_res = run_greedy(seed)
        if not greedy_res:
            continue
        
        # Post-process: remove redundant blocks
        processed_res = try_remove_blocks(greedy_res)
        
        # Additional local search
        processed_res = local_search(processed_res, max_swaps=20)
        
        if len(processed_res) < best_len:
            best_len = len(processed_res)
            best_blocks = list(processed_res)
        
        if best_len <= 70:
            break
    
    # If we have time, try more random seeds
    if best_blocks is not None and time.time() < deadline - 10:
        while time.time() < deadline - 5:
            seed = random.randint(0, 1000000)
            greedy_res = run_greedy(seed)
            if not greedy_res:
                continue
            
            processed_res = try_remove_blocks(greedy_res)
            processed_res = local_search(processed_res, max_swaps=10)
            
            if len(processed_res) < best_len:
                best_len = len(processed_res)
                best_blocks = list(processed_res)
                
            if best_len <= 69:
                break
    
    if best_blocks is None:
        # Fallback: run one greedy and return
        greedy_res = run_greedy(0)
        if greedy_res:
            best_blocks = greedy_res
        else:
            return []
    
    # Convert block indices back to actual blocks
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    for b in final_blocks:
        b.sort()
        
    return final_blocks
