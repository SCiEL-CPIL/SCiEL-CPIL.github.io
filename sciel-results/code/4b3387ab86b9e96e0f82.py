"""Covering design C(19,7,4) using bitset-accelerated greedy and local search."""
import random
import itertools

def construct(v, k, t):
    """
    Find a covering design with as few blocks as possible.
    Uses bitset-accelerated greedy initialization and local search (2-for-1 swaps).
    """
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # Precompute bitsets for all possible k-subsets
    # There are C(19,7) = 50388 blocks
    all_blocks = list(itertools.combinations(range(v), k))
    block_bitset = {}
    # Map t-subset index to list of blocks containing it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    
    for b in all_blocks:
        bs = 0
        for ts in itertools.combinations(b, t):
            idx = t_sub_index[ts]
            bs |= (1 << idx)
            tsub_to_blocks[idx].append(b)
        block_bitset[b] = bs
        
    # 2. Greedy Initialization
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
            
        uncovered = (1 << n_t_subs) - 1
        cover = []
        
        while uncovered:
            best_block = None
            best_gain = -1
            # To add randomness, we can shuffle the block list or just pick randomly from top gains
            # For speed, we iterate all blocks. 50k * 150 iterations = 7.5M operations. Acceptable.
            # To make it random, we can add a small random noise to the gain?
            # Or just shuffle the blocks once?
            # Let's just iterate and keep track of best. If ties, pick random.
            
            candidates = []
            for b in all_blocks:
                gain = bin(uncovered & block_bitset[b]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    candidates = [b]
                elif gain == best_gain:
                    candidates.append(b)
            
            if best_block is None:
                break
                
            # Random selection from candidates
            b = random.choice(candidates)
            cover.append(b)
            uncovered &= ~block_bitset[b]
            
        return cover

    # 3. Local Search
    def local_search(cover):
        # Convert to set for fast lookup
        cover_set = set(cover)
        # Precompute union of all block bitsets? 
        # We need to calculate "unique to pair" which requires OR of others.
        # This is expensive if done from scratch every time.
        # But v=19, cover size ~150. 
        # Let's just do it naively but efficiently.
        
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant blocks
            # For each block, check if it covers any unique subsets.
            # unique(b) = block_bitset[b] & ~(OR of all other blocks)
            # If unique(b) == 0, remove b.
            
            # To speed up, we can maintain a global OR?
            # But removing a block changes the OR.
            # Let's just iterate.
            
            to_remove = []
            # Calculate total union
            total_union = 0
            for b in cover:
                total_union |= block_bitset[b]
                
            for b in cover:
                others_union = total_union & ~block_bitset[b]
                # Wait, if b is the only one covering some subset, then that subset is in block_bitset[b] but not in others_union.
                # unique = block_bitset[b] & ~others_union
                # If unique == 0, b is redundant.
                if (block_bitset[b] & ~others_union) == 0:
                    to_remove.append(b)
                    
            if to_remove:
                for b in to_remove:
                    cover.remove(b)
                improved = True
                continue # Restart loop as cover changed
                
            # Step B: 2-for-1 Swap
            # Try a random subset of pairs to limit time
            # Number of pairs is N*(N-1)/2. N~150, pairs ~11000.
            # We can try all pairs, or a random sample.
            # Let's try all pairs, but break early if we find a swap.
            
            # We need to be careful with indices if we modify list.
            # We'll collect swaps and apply them after.
            
            found_swap = False
            # Shuffle pairs for randomization?
            # Let's just iterate.
            for i in range(len(cover)):
                for j in range(i+1, len(cover)):
                    b1 = cover[i]
                    b2 = cover[j]
                    
                    # Calculate unique subsets to this pair
                    # union_pair = block_bitset[b1] | block_bitset[b2]
                    # others_union = total_union & ~(block_bitset[b1] | block_bitset[b2])
                    # unique_to_pair = union_pair & ~others_union
                    # Note: total_union is the OR of ALL blocks in cover.
                    # So others_union is OR of all blocks EXCEPT b1 and b2.
                    
                    pair_union = block_bitset[b1] | block_bitset[b2]
                    others_union = total_union & ~pair_union
                    unique_to_pair = pair_union & ~others_union
                    
                    if unique_to_pair == 0:
                        # One of them is redundant? 
                        # If unique_to_pair is 0, it means every subset covered by b1 or b2 is also covered by some other block.
                        # So both can be removed? No, maybe one is redundant and the other is not?
                        # If unique_to_pair is 0, then b1 is redundant AND b2 is redundant?
                        # Not necessarily. 
                        # Example: b1 covers {1}, b2 covers {2}, b3 covers {1,2}. 
                        # unique_to_pair(b1,b2) = {1,2} & ~{1,2} = 0.
                        # Both b1 and b2 are redundant.
                        # We should have caught this in Step A?
                        # Step A checks if b is redundant w.r.t ALL others.
                        # If b1 is redundant, Step A removes it.
                        # So if we are here, neither is individually redundant.
                        # But together they might be?
                        # If unique_to_pair is 0, it means the union of their unique contributions is empty.
                        # This implies they are both redundant?
                        # Let's just remove one? Or both?
                        # If we remove both, we lose nothing.
                        # But Step A should have removed them one by one.
                        # Why didn't it?
                        # Maybe the order matters?
                        # Anyway, if unique_to_pair is 0, we can remove one of them.
                        # Let's remove b2.
                        cover.remove(b2)
                        improved = True
                        found_swap = True
                        break
                        
                    # Find a block B_new that covers unique_to_pair
                    # Pick a set bit from unique_to_pair
                    # Find the index of the first set bit
                    # unique_to_pair is an int.
                    # idx = (unique_to_pair & -unique_to_pair).bit_length() - 1
                    # Or just iterate bits?
                    # Let's get one arbitrary t-subset index.
                    
                    # Get a representative t-subset index
                    # We can just pick the lowest set bit.
                    lowbit = unique_to_pair & -unique_to_pair
                    rep_idx = lowbit.bit_length() - 1
                    
                    # Candidates are blocks containing this t-subset
                    candidates = tsub_to_blocks[rep_idx]
                    
                    for b_new in candidates:
                        # Check if b_new covers all of unique_to_pair
                        # i.e., unique_to_pair is a subset of block_bitset[b_new]
                        if (unique_to_pair & ~block_bitset[b_new]) == 0:
                            # Valid swap found
                            # Check if b_new is already in cover
                            if b_new in cover:
                                # If b_new is b1 or b2, we just remove the other
                                if b_new == b1:
                                    cover.remove(b2)
                                else: # b_new == b2
                                    cover.remove(b1)
                            else:
                                # Replace b1, b2 with b_new
                                cover.remove(b1)
                                cover.remove(b2)
                                cover.append(b_new)
                                
                            improved = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        break
                        
                if found_swap:
                    break
            
            if found_swap:
                # Recalculate total_union for next iteration
                total_union = 0
                for b in cover:
                    total_union |= block_bitset[b]
                continue
                
            # If no swap found, break
            break
            
        return cover

    # Main Loop: Try multiple greedy initializations
    best_cover = None
    best_len = float('inf')
    
    for seed in range(10):
        cover = greedy_init(seed)
        cover = local_search(cover)
        
        if len(cover) < best_len:
            best_len = len(cover)
            best_cover = cover
            
        # Early stop if we hit a very good number?
        # 151 is best known.
        if best_len <= 151:
            break
            
    return [list(b) for b in best_cover]
