import itertools
import random

def construct(v, k, t):
    """
    Bitset-based Greedy Set Cover with Random Sampling and Restarts.
    """
    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    # Generate all k-subsets
    all_blocks = list(itertools.combinations(all_points, k))
    num_blocks = len(all_blocks)
    
    # Precompute bitsets
    # Using Python integers for bitsets
    block_bitsets = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        bits = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_index[t_sub]
            bits |= (1 << idx)
        block_bitsets[i] = bits
        
    best_blocks = None
    best_len = float('inf')
    
    # Parameters for search
    num_restarts = 200
    base_sample_size = 1000
    max_steps = 150
    
    for r in range(num_restarts):
        rng = random.Random(1000 + r)
        
        uncovered = (1 << num_t_subs) - 1
        selected_blocks = []
        
        steps = 0
        while uncovered and steps < max_steps:
            steps += 1
            # Adaptive sample size: start small, increase if progress is slow?
            # For now, fixed sample size for simplicity and speed.
            sample_size = base_sample_size
            
            if num_blocks <= sample_size:
                candidates = range(num_blocks)
            else:
                candidates = rng.sample(range(num_blocks), sample_size)
            
            best_idx = -1
            best_score = -1
            best_bits = 0
            
            for idx in candidates:
                bits = block_bitsets[idx]
                score = (bits & uncovered).bit_count()
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_bits = bits
            
            if best_idx == -1 or best_score == 0:
                # Fallback: random pick from all
                best_idx = rng.randint(0, num_blocks - 1)
                best_bits = block_bitsets[best_idx]
            
            selected_blocks.append(all_blocks[best_idx])
            uncovered &= ~best_bits
        
        if uncovered == 0:
            if len(selected_blocks) < best_len:
                best_len = len(selected_blocks)
                best_blocks = [list(b) for b in selected_blocks]
                
        if best_len <= 76:
            break
            
    return best_blocks if best_blocks is not None else []
