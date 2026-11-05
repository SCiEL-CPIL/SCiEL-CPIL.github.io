import time
import random
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Build S_{n-1} using the standard recursive method
    # This is fast and provides the base blocks.
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    # Now word is S_{n-1}
    # Extract the (n-1)! permutations of length n-1
    perms_n_minus_1 = []
    seen = set()
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_n_minus_1.append(window)
            
    # Step 2: Create blocks for n
    # Block for permutation p is p + [n] + p
    blocks = []
    for p in perms_n_minus_1:
        blocks.append(list(p) + [n] + list(p))
        
    num_blocks = len(blocks)
    
    # Step 3: Find a good ordering of blocks to maximize overlaps
    # We use a greedy nearest-neighbor approach with some randomization/restarts
    # and then a 2-opt improvement.
    
    def get_overlap(block1, block2):
        # Max overlap between suffix of block1 and prefix of block2
        max_len = min(len(block1), len(block2))
        for o in range(max_len, -1, -1):
            if o == 0:
                return 0
            if block1[len(block1)-o:] == block2[:o]:
                return o
        return 0

    best_order = None
    best_length = float('inf')
    
    start_time = time.time()
    time_limit = 110  # Leave 10s for construction and return
    
    # Pre-compute overlaps? 720x720 = 518400. Might be slow in Python if not optimized.
    # Let's try a few random starts and greedy builds.
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Random start
        current_idx = random.randint(0, num_blocks - 1)
        current_block = blocks[current_idx]
        current_length = len(current_block)
        order = [current_idx]
        remaining = set(range(num_blocks))
        remaining.remove(current_idx)
        
        # Greedy nearest neighbor
        while remaining:
            best_next = -1
            best_overlap = -1
            # To speed up, we can limit the search or use a heuristic
            # But 720 is small enough to check all remaining in each step?
            # 720 * 720 operations ~ 500k. In Python, this might be slow.
            # Let's optimize the overlap check.
            
            # Optimization: Only check overlaps of size >= 1?
            # Actually, we want max overlap.
            
            for next_idx in remaining:
                next_block = blocks[next_idx]
                # Quick check: last char of current must match first char of next for overlap > 0?
                # Not necessarily, overlap can be 0.
                # But max overlap is what we want.
                
                # We can pre-calculate the suffixes/prefixes?
                # Let's just do the check.
                max_len = min(len(current_block), len(next_block))
                # Most overlaps are small. Check from max_len down?
                # Or check from 1 up to max_len?
                # The standard recursive construction has large overlaps.
                # Let's check from max_len down to 0.
                
                # To speed up, we can break early if we find a "good" overlap?
                # No, we want the best.
                
                # Optimization: 
                # If current_block[-1] != next_block[0], overlap is 0.
                if current_block[-1] != next_block[0]:
                    overlap = 0
                else:
                    # Check larger overlaps
                    # We can use a simple loop
                    found = 0
                    for o in range(1, max_len + 1):
                        if current_block[-o:] == next_block[:o]:
                            found = o
                    overlap = found
                    
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_next = next_idx
                    
            if best_next == -1:
                # Should not happen
                break
                
            order.append(best_next)
            remaining.remove(best_next)
            current_block = blocks[best_next]
            current_length += len(current_block) - best_overlap
            
        # Calculate total length
        if len(order) == num_blocks:
            if current_length < best_length:
                best_length = current_length
                best_order = order[:]
                
    # If we found a good order, use it. Otherwise, fall back to original order.
    if best_order is None:
        best_order = list(range(num_blocks))
        
    # Construct the final word
    final_word = []
    for i, idx in enumerate(best_order):
        block = blocks[idx]
        if i == 0:
            final_word.extend(block)
        else:
            prev_block = blocks[best_order[i-1]]
            overlap = get_overlap(prev_block, block)
            final_word.extend(block[overlap:])
            
    return "".join(map(str, final_word))
