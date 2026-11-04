import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all t-subsets and their bitmasks
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    mask_to_idx = {}
    for i, sub in enumerate(all_t_subs):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        mask_to_idx[mask] = i

    # Precompute masks for all (t-1) subsets to speed up gain calculation
    # We need to quickly compute the bitmask of t-subsets covered by adding x to a set B.
    # Covered t-subsets are of the form S U {x} where S is a (t-1)-subset of B.
    
    def get_gain_mask(block, x):
        """
        Returns a bitmask of t-subset indices that would be covered by adding x to block.
        block: list of points (size < k)
        x: point to add
        """
        x_bit = 1 << x
        gain_mask = 0
        # Iterate over (t-1) subsets of block
        for rest in itertools.combinations(block, t - 1):
            m_rest = 0
            for p in rest:
                m_rest |= (1 << p)
            full_mask = m_rest | x_bit
            idx = mask_to_idx.get(full_mask)
            if idx is not None:
                gain_mask |= (1 << idx)
        return gain_mask

    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        # Use an integer bitmask for uncovered t-subsets
        # Bit i is 1 if t-subset i is uncovered
        uncovered_mask = (1 << n_t_subs) - 1
        blocks = []
        
        while uncovered_mask:
            # Pick a random uncovered t-subset to start
            # Get index of a random set bit
            # Convert to string? No, just iterate or use a trick.
            # For n_t_subs ~ 11k, we can just find a random index and check?
            # Or extract a random bit.
            # Let's just pick a random index from the list of uncovered ones?
            # Finding all uncovered indices is O(n).
            # We can maintain a list of uncovered indices and remove from it?
            # That's complex.
            
            # Simple approach: pick random index, if not uncovered, try again?
            # Or just pick the first uncovered one for speed, and randomize block growth.
            # Let's pick a random index and check if it's uncovered.
            # If n_t_subs is 11628, this is fast.
            
            # Optimization: Find an uncovered index quickly.
            # We can use the fact that we just need *one* uncovered subset.
            # Let's just pick a random index.
            start_idx = random.randint(0, n_t_subs - 1)
            # Ensure it's actually uncovered? 
            # If it is covered, we wasted a step? 
            # No, we just start a block that might not cover it immediately?
            # But we want to cover uncovered subsets.
            # Better: Find an uncovered subset.
            # Since we start with all uncovered, and remove them, the density drops.
            # Let's just iterate through indices until we find an uncovered one.
            # Worst case O(n). Average O(1) if dense.
            
            found = False
            while not found:
                start_idx = random.randint(0, n_t_subs - 1)
                if (uncovered_mask >> start_idx) & 1:
                    found = True
            
            block = list(all_t_subs[start_idx])
            block_set = set(block)
            
            # Grow the block
            while len(block) < k:
                best_x = -1
                best_gain = -1
                candidates = []
                
                # Calculate gain for each candidate x
                for x in range(v):
                    if x in block_set:
                        continue
                    gain_mask = get_gain_mask(block, x)
                    # Count bits in intersection
                    gain = (uncovered_mask & gain_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        candidates = [x]
                    elif gain == best_gain:
                        candidates.append(x)
                
                if best_x == -1:
                    # No gain, just pick a random unused point
                    available = [x for x in range(v) if x not in block_set]
                    if not available:
                        break
                    best_x = random.choice(available)
                
                # Random tie-breaking
                if len(candidates) > 1:
                    best_x = random.choice(candidates)
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            # For each t-subset in block, clear its bit in uncovered_mask
            for t_sub in itertools.combinations(block, t):
                mask = 0
                for p in t_sub:
                    mask |= (1 << p)
                idx = mask_to_idx.get(mask)
                if idx is not None:
                    uncovered_mask &= ~(1 << idx)
                    
        return blocks

    def reduce_covering(blocks):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return blocks
        
        # Precompute the list of t-subset indices for each block
        block_t_indices = []
        for b in blocks:
            idxs = []
            for t_sub in itertools.combinations(b, t):
                mask = 0
                for p in t_sub:
                    mask |= (1 << p)
                idx = mask_to_idx.get(mask)
                if idx is not None:
                    idxs.append(idx)
            block_t_indices.append(idxs)
        
        # Coverage count for each t-subset
        cov = np.zeros(n_t_subs, dtype=np.int32)
        for idxs in block_t_indices:
            for idx in idxs:
                cov[idx] += 1
        
        removed = np.zeros(len(blocks), dtype=bool)
        changed = True
        
        while changed:
            changed = False
            for i in range(len(blocks)):
                if removed[i]:
                    continue
                idxs = block_t_indices[i]
                if not idxs:
                    removed[i] = True
                    changed = True
                    continue
                    
                is_essential = False
                for idx in idxs:
                    if cov[idx] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    removed[i] = True
                    changed = True
                    for idx in idxs:
                        cov[idx] -= 1
        
        result = [blocks[i] for i in range(len(blocks)) if not removed[i]]
        return result

    best_blocks = None
    best_size = float('inf')
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        seed = attempt * 12345 + 67890
        blocks = greedy_construct(seed)
        
        if not blocks:
            continue
        
        reduced_blocks = reduce_covering(blocks)
        
        if len(reduced_blocks) < best_size:
            best_size = len(reduced_blocks)
            best_blocks = reduced_blocks
            
        if time.time() > deadline - 10.0:
            break

    if best_blocks is None:
        blocks = greedy_construct(None)
        return reduce_covering(blocks)
        
    return best_blocks
