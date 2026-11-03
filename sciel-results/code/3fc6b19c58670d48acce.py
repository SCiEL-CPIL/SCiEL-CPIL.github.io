import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Build up to n-1
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
            # Find max overlap
            max_overlap = 0
            # Optimization: only check up to m-1
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
        
    # Now for m = n, we have a list of (n-1)! permutations from the previous step
    # We want to order them to minimize the total length of the final word
    
    perms, seen = [], set()
    for i in range(len(word) - n + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    # perms contains (n-1)! tuples
    # We create blocks for each
    blocks = []
    for p in perms:
        blocks.append(list(p) + [n] + list(p))
        
    # Initial order: same as perms
    current_order = list(range(len(perms)))
    
    # Helper to calculate length given an order of block indices
    def calc_length(order):
        if not order:
            return len(word)
        # Start with word
        # We simulate appending blocks
        # To be fast, we can just sum the lengths minus overlaps
        # But overlaps depend on the previous block's end and current block's start
        # We need the actual end of the previous block.
        # The end of the previous block is the last (n-1) symbols of the previous block?
        # No, the block is P + [n] + P.
        # The overlap is between the end of the accumulated word and the start of the new block.
        
        # Let's just build the string for the last part?
        # That might be slow if we do it for every swap.
        # 720 blocks. String length ~ 5000.
        # 720 * 5000 = 3.6M ops per evaluation.
        # If we do 1000 swaps, 3.6B ops. Too slow in Python.
        
        # We need a faster overlap check.
        # The overlap between block A and block B only depends on the suffix of A and prefix of B.
        # The suffix of the accumulated word is the suffix of the last appended block.
        # So we can maintain the current suffix.
        
        # Let's optimize calc_length
        cur_suffix = tuple(word) # Initial word
        total_len = len(word)
        for idx in order:
            b = blocks[idx]
            # Find max overlap between cur_suffix and b
            # Max possible overlap is len(b)-1? No, limited by len(cur_suffix)
            # Actually, overlap k means cur_suffix[-k:] == b[:k]
            # We can check k from min(len(cur_suffix), len(b)-1) down to 0
            # But we can precompute? No, cur_suffix changes.
            
            # Heuristic: The overlap is usually small (1 to n-1)
            # Let's check k from min(len(cur_suffix), n-1) down to 0
            max_k = min(len(cur_suffix), n-1)
            found = 0
            for k in range(max_k, -1, -1):
                if cur_suffix[-k:] == tuple(b[:k]):
                    found = k
                    break
            total_len += len(b) - found
            # Update suffix
            # The new suffix is the last n-1 symbols of the new accumulated word
            # The new accumulated word ends with b[found:]
            # So the new suffix is the last n-1 of b[found:]
            # If len(b[found:]) >= n-1, suffix = b[found:][-n-1:]
            # Else suffix = cur_suffix[-(n-1 - len(b[found:])):] + b[found:]
            
            new_part = b[found:]
            if len(new_part) >= n - 1:
                cur_suffix = tuple(new_part[-(n-1):])
            else:
                # Need to take from old suffix
                needed = (n - 1) - len(new_part)
                cur_suffix = cur_suffix[-needed:] + tuple(new_part)
                
        return total_len

    best_order = current_order
    best_len = calc_length(best_order)
    
    # Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize start to avoid local minima? 
        # Or just standard swaps.
        # Try all adjacent swaps
        for i in range(len(best_order) - 1):
            # Swap i and i+1
            new_order = best_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_len = calc_length(new_order)
            if new_len < best_len:
                best_len = new_len
                best_order = new_order
                improved = True
                # Don't break, keep looking for more improvements in this pass
                # Or break to restart? Standard hill climbing usually takes the first improvement or best improvement.
                # Let's take the best improvement in the pass.
                # For speed, let's just take the first improvement and restart the loop?
                # That's safer.
                break
                
        if not improved:
            # Try random swaps to escape local minima
            if random.random() < 0.1:
                i = random.randint(0, len(best_order)-2)
                j = random.randint(0, len(best_order)-2)
                if i != j:
                    new_order = best_order[:]
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_len = calc_length(new_order)
                    if new_len < best_len:
                        best_len = new_len
                        best_order = new_order
                        improved = True

    # Construct final word
    final_word = list(word)
    for idx in best_order:
        b = blocks[idx]
        # Find overlap
        max_k = min(len(final_word), n-1)
        found = 0
        for k in range(max_k, -1, -1):
            if final_word[-k:] == b[:k]:
                found = k
                break
        final_word += b[found:]
        
    return "".join(map(str, final_word))
