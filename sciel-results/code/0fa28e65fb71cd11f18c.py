import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
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
    
    # If n is small, return immediately
    if n <= 5:
        return "".join(map(str, word))
    
    # For n >= 6, try to improve the ordering of blocks
    # Extract the blocks used in the last step
    prev_word = word[:-2]  # Remove the last 'n' and the last block? No, let's re-extract.
    
    # Re-extract blocks from the word BEFORE the last step (n-1 superpermutation)
    # Actually, it's easier to just re-run the extraction on the word of length n-1.
    # But we don't have it stored. Let's re-generate it.
    
    # Regenerate superpermutation for n-1
    prev_word_list = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(prev_word_list) - m + 2):
            window = tuple(prev_word_list[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        prev_word_list = new
        
    # Extract blocks from prev_word_list
    blocks = []
    seen = set()
    for i in range(len(prev_word_list) - (n - 2)):
        window = tuple(prev_word_list[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            blocks.append(window)
            
    # The standard order of blocks is the order they appear in prev_word_list
    # Let's verify this matches the order used in the standard construction for n
    # The standard construction for n uses these blocks in this order.
    
    # Define a function to calculate the length of the superpermutation given an order of blocks
    # This is the same logic as before, but we need to be careful about the reconstruction.
    # The word is constructed as:
    # b1 n b1 b2 n b2 ... bk n bk
    # Overlap between b_i and b_{i+1} is used to merge them.
    # Specifically, if we have accumulated word W ending with b_i, and we want to add b_{i+1} n b_{i+1},
    # we find the max overlap k such that W[-k:] == b_{i+1}[:k].
    # Then we append b_{i+1}[k:] + [n] + b_{i+1}.
    # Wait, the structure is ... b_i, n, b_i, b_{i+1}, n, b_{i+1} ...
    # So after b_i, we have n, then b_i again? No.
    # Let's trace the standard construction again.
    # S_n = b1 n b1 b2 n b2 ...
    # Step 1: Start with b1.
    # Step 2: Append n, then b1? No.
    # The recursive step says: for each permutation p in S_{n-1}, append p, n, p.
    # But with overlaps.
    # So if current word is W, and we process p:
    # We want to append p, n, p.
    # We check overlap between W suffix and p prefix.
    # Append p[overlap:]
    # Then append n
    # Then append p (checking overlap with n? No, n is a new symbol, so no overlap with p unless p starts with n, which it doesn't).
    # So we just append p.
    # So the segment added for p is: p[overlap:] + [n] + p.
    
    def calc_length(block_order):
        if not block_order:
            return 0
        # Start with first block
        # The word starts with b1.
        # Then we add n, b1? No.
        # Let's simulate.
        # W = []
        # for p in block_order:
        #   overlap = find_overlap(W, p)
        #   W += p[overlap:]
        #   W += [n]
        #   W += p
        #   # Note: the second p is appended fully because n != p[0]
        #   # Wait, is there overlap between n and p? No.
        #   # Is there overlap between the first p[overlap:] and the second p?
        #   # The sequence is ... p[overlap:], n, p.
        #   # n is distinct from all symbols in p. So no overlap across n.
        
        # So the length is:
        # len(b1) + (1 + len(b1)) + sum_{i=2}^k ( (len(b_i) - overlap(W_{i-1}, b_i)) + 1 + len(b_i) )
        # Where W_{i-1} is the word built so far.
        # But W_{i-1} ends with b_{i-1} (the second p of the previous iteration).
        # So overlap is between b_{i-1} and b_i.
        
        total_len = len(block_order[0])
        for i in range(len(block_order)):
            # Add n and p
            total_len += 1 + len(block_order[i])
            if i < len(block_order) - 1:
                curr = block_order[i]
                next_b = block_order[i+1]
                # Find overlap between curr and next_b
                max_k = min(len(curr), len(next_b))
                overlap = 0
                for k in range(max_k, 0, -1):
                    if curr[-k:] == next_b[:k]:
                        overlap = k
                        break
                # The next block b_{i+1} will be appended with overlap
                # So the cost for b_{i+1} is (len(b_{i+1}) - overlap) + 1 + len(b_{i+1})
                # But we already added 1 + len(b_{i+1}) in the next loop iteration?
                # Let's restructure.
                pass
                
        # Let's just simulate the length directly.
        total_len = 0
        current_suffix = []
        for i, p in enumerate(block_order):
            # Find overlap between current_suffix and p
            max_k = min(len(current_suffix), len(p))
            overlap = 0
            for k in range(max_k, 0, -1):
                if current_suffix[-k:] == list(p[:k]):
                    overlap = k
                    break
            total_len += len(p) - overlap
            current_suffix = list(p)
            # Add n
            total_len += 1
            current_suffix.append(n)
            # Add p
            total_len += len(p)
            current_suffix.extend(p)
            
        return total_len

    # This is O(N^2 * L) where N is num blocks (720) and L is block length (6).
    # 720 * 720 * 6 is ~ 3M ops per full pass.
    # We can do a few passes.
    
    best_order = blocks[:]
    best_len = calc_length(best_order)
    
    start_time = time.time()
    time_limit = 110
    
    # 2-opt local search
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(len(best_order) - 1):
            for j in range(i + 1, len(best_order)):
                # Try reversing segment i..j
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                new_len = calc_length(new_order)
                if new_len < best_len:
                    best_len = new_len
                    best_order = new_order
                    improved = True
                    if time.time() - start_time > time_limit:
                        break
            if time.time() - start_time > time_limit:
                break
    
    # Reconstruct the word from best_order
    new_word = []
    for p in best_order:
        block = list(p) + [n] + list(p)
        overlap = next(o for o in range(min(len(new_word), len(block)), -1, -1) if new_word[len(new_word) - o:] == block[:o])
        new_word += block[overlap:]
        
    # Verify validity? (Optional, but good for debugging)
    # For now, return the new word
    return "".join(map(str, new_word))
