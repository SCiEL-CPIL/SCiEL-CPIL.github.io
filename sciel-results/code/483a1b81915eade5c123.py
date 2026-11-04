import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 145s limit, 5s margin for safety
    
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_subsets = len(t_subsets)
    
    # Precompute masks for all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = {}
    for b in all_blocks:
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= 1 << subset_to_idx[sub]
        block_masks[b] = mask
    
    # Precompute pair masks: for each pair (i,j) with i<j, the mask of t-subsets containing both i and j
    # This allows O(1) gain computation when adding a point to a partial block
    pair_masks = {}
    for i in range(v):
        for j in range(i+1, v):
            mask = 0
            # t-subsets containing both i and j: choose t-2 more from remaining v-2 points
            for rest in itertools.combinations([x for x in range(v) if x != i and x != j], t-2):
                sub = tuple(sorted([i, j] + list(rest)))
                mask |= 1 << subset_to_idx[sub]
            pair_masks[(i,j)] = mask
    
    # Precompute for a partial block S (as frozenset), the mask of t-subsets within S
    # We'll compute this on the fly using pair_masks for efficiency
    def partial_block_mask(block_list):
        """Compute mask of t-subsets contained in block_list (which has size < k)."""
        if len(block_list) < t:
            return 0
        mask = 0
        for sub in itertools.combinations(block_list, t):
            mask |= 1 << subset_to_idx[sub]
        return mask
    
    # Faster: compute gain of adding point x to current block
    # The new t-subsets formed are those containing x and t-1 points from current block
    def gain_of_adding(block, x, uncovered):
        """Compute how many currently uncovered t-subsets would be covered by adding x to block."""
        # New t-subsets: {x} ∪ S where S is a (t-1)-subset of block
        gain = 0
        for s in itertools.combinations(block, t-1):
            sub = tuple(sorted(s + (x,)))
            idx = subset_to_idx[sub]
            if (uncovered >> idx) & 1:
                gain += 1
        return gain
    
    best_blocks = None
    best_size = float('inf')
    
    # Number of attempts based on available time
    # Estimate: each attempt takes roughly T seconds
    # We'll run as many as fit before deadline
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        strategy = attempt % 4
        
        uncovered = (1 << n_subsets) - 1
        blocks = []
        counts = [0] * n_subsets
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick which uncovered subset to start with
            if strategy == 0:
                # Lowest set bit
                idx = (uncovered & -uncovered).bit_length() - 1
            elif strategy == 1:
                # Random set bit - faster method
                pop = uncovered.bit_count()
                if pop == 0:
                    break
                rank = random.randrange(pop)
                # Find rank-th set bit using bit manipulation
                temp = uncovered
                found = -1
                # Use a faster approach: iterate through bits
                for i in range(n_subsets):
                    if temp & (1 << i):
                        rank -= 1
                        if rank < 0:
                            found = i
                            break
                idx = found if found != -1 else (uncovered & -uncovered).bit_length() - 1
            elif strategy == 2:
                # Pick uncovered subset with some randomization
                pop = uncovered.bit_count()
                if pop <= 20:
                    bits = []
                    temp = uncovered
                    for i in range(n_subsets):
                        if temp & (1 << i):
                            bits.append(i)
                    idx = random.choice(bits) if bits else (uncovered & -uncovered).bit_length() - 1
                else:
                    idx = (uncovered & -uncovered).bit_length() - 1
            else:
                # Strategy 3: Random among first N uncovered subsets
                sample_size = min(30, uncovered.bit_count())
                candidates = []
                i = 0
                while len(candidates) < sample_size and i < n_subsets:
                    if (uncovered >> i) & 1:
                        candidates.append(i)
                    i += 1
                idx = random.choice(candidates) if candidates else (uncovered & -uncovered).bit_length() - 1
            
            first_subset = t_subsets[idx]
            block = list(first_subset)
            
            # Add k-t more points greedily
            while len(block) < k:
                best_gain = -1
                best_point = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Compute gain: how many new uncovered t-subsets would be covered
                    # New t-subsets contain x and t-1 elements from current block
                    gain = 0
                    for s in itertools.combinations(block, t-1):
                        sub = tuple(sorted(s + (x,)))
                        idx2 = subset_to_idx[sub]
                        if (uncovered >> idx2) & 1:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and gain > 0 and random.random() < 0.05:
                        best_point = x
                        
                if best_point == -1:
                    break
                block.append(best_point)
                block.sort()
                
            # Finalize block
            final_block = tuple(block)
            blocks.append(final_block)
            
            # Update uncovered and counts
            mask = block_masks[final_block]
            uncovered &= ~mask
            for sub in itertools.combinations(final_block, t):
                counts[subset_to_idx[sub]] += 1
        
        if time.time() > deadline:
            break
            
        # Pruning Phase - remove redundant blocks
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for i in range(len(blocks)):
                redundant = True
                for sub in itertools.combinations(blocks[i], t):
                    if counts[subset_to_idx[sub]] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(i)
            
            # Remove from end to preserve indices
            for i in reversed(to_remove):
                b = blocks[i]
                blocks.pop(i)
                for sub in itertools.combinations(b, t):
                    counts[subset_to_idx[sub]] -= 1
                changed = True
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
            
    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]
