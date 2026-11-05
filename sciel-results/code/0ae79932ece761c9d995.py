import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate the standard superpermutation for n-1 to get the block order
    # We need the sequence of (n-1)-permutations as they appear in the standard S_{n-1}
    
    def get_standard_blocks(m):
        if m == 1:
            return [[1]]
        
        # Recursively get the word for m-1
        # But we just need the order of blocks for m-1 to generate blocks for m?
        # No, the blocks for m are the permutations of 1..m-1.
        # The *order* of these blocks in the standard construction of S_m
        # is determined by the structure of S_{m-1}.
        
        # Let's just generate the standard word for m-1 and extract the blocks.
        # This is O(n!) which is fine for n=7 (5040).
        
        # Standard construction of word
        word = [1]
        for k in range(2, m + 1):
            # Extract (k-1)-permutations from current word
            perms = []
            seen = set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1:
                    if window not in seen:
                        seen.add(window)
                        perms.append(window)
            
            # Build new word
            new_word = []
            for p in perms:
                # Insert k
                # Standard logic:
                # The block is p. We add k then p.
                # Overlap with previous end.
                block = list(p) + [k] + list(p)
                # Find overlap with new_word
                overlap = 0
                max_ov = min(len(new_word), len(block))
                for ov in range(max_ov, 0, -1):
                    if new_word[len(new_word)-ov:] == block[:ov]:
                        overlap = ov
                        break
                new_word.extend(block[overlap:])
            word = new_word
            
        # Now extract the m-permutations? No, we want the (m-1)-permutations that form the blocks for the next step.
        # Actually, for construct(n), we need the blocks of size n-1.
        # These are the permutations of 1..n-1.
        # Their order in the standard S_n is the order they appear in the construction.
        
        # Let's just return the list of blocks (tuples) in the order they were added to the word.
        # Wait, the blocks for S_n are the perms of 1..n-1.
        # In the loop above for k=m, we iterated over `perms` which are the (m-1)-perms.
        # So `perms` is exactly the list of blocks we need for the next level.
        
        # So, to get blocks for n, we call get_standard_blocks(n-1) which returns the word?
        # No, let's refactor.
        
        # We want a function that returns the list of blocks (perms of 1..m-1) in the standard order.
        
    # Let's implement a helper to get the blocks for a given size m (perms of 1..m-1)
    def get_blocks_for(m):
        """Returns list of tuples, each a permutation of 1..m-1, in standard order."""
        if m == 2:
            return [(1,), (2,)] # Wait, perms of 1..1? No.
            # For S_2, blocks are perms of 1..1? No.
            # S_1 = "1".
            # Blocks for S_2 are perms of 1..1? No, S_2 is built from S_1.
            # S_1 has "1".
            # Perms of 1..1 is just (1,).
            # So blocks for S_2 are [(1,)].
            # Let's trace:
            # S_1 = [1].
            # Blocks for S_2: perms of 1..1. Just (1,).
            # S_2 = (1,) + [2] + (1,) -> 121.
            # Correct.
            
        # General case:
        # We need the blocks for S_m. These are perms of 1..m-1.
        # The order is determined by S_{m-1}.
        
        if m == 2:
            return [(1,)]
            
        # Get blocks for S_{m-1} (perms of 1..m-2)
        prev_blocks = get_blocks_for(m-1)
        
        # We need to reconstruct S_{m-1} to find the order of perms of 1..m-1?
        # No, the order of perms of 1..m-1 in S_m is NOT simply related to perms of 1..m-2 in a trivial way without building the word.
        
        # Actually, the standard construction defines the order.
        # Let's just build the word for m-1 and extract the windows.
        
        # Build S_{m-1}
        word = [1]
        for k in range(2, m):
            perms = []
            seen = set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1:
                    if window not in seen:
                        seen.add(window)
                        perms.append(window)
                        
            new_word = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = 0
                max_ov = min(len(new_word), len(block))
                for ov in range(max_ov, 0, -1):
                    if new_word[len(new_word)-ov:] == block[:ov]:
                        overlap = ov
                        break
                new_word.extend(block[overlap:])
            word = new_word
            
        # Now word is S_{m-1}.
        # Extract the (m-1)-permutations from S_{m-1}.
        blocks = []
        seen = set()
        for i in range(len(word) - (m-2)):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    blocks.append(window)
                    
        return blocks

    # Get blocks for n (perms of 1..n-1)
    blocks = get_blocks_for(n)
    num_blocks = len(blocks)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Note: k can be 0 to n-2.
    
    # To speed up, we can use a dictionary or just a 2D list.
    # 720x720 is small.
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        b1 = blocks[i]
        for j in range(num_blocks):
            b2 = blocks[j]
            max_k = n - 2 # Max possible overlap is n-2? 
                         # Blocks are length n-1.
                         # Overlap is suffix of b1 and prefix of b2.
                         # Max length is n-1.
                         # But if b1 == b2, overlap is n-1.
                         # Can b1 == b2? Yes, if the order has duplicates?
                         # No, blocks are distinct permutations.
                         # So b1 != b2.
                         # Max overlap is n-2?
                         # Example: 123 and 231. Suffix 23, Prefix 23. Overlap 2.
                         # Example: 123 and 124? No, distinct perms of same set.
                         # If b1 = 1234, b2 = 2341. Overlap 3.
                         # So max overlap is n-2?
                         # If b1 = 123, b2 = 234? No, same symbols.
                         # b1 = 123, b2 = 231. Suffix 23, Prefix 23.
                         # b1 = 123, b2 = 312? Suffix 3, Prefix 3? No.
                         # b1 = 123, b2 = 123? Impossible.
                         # So max overlap is n-2.
                         
            # Actually, just check up to n-1.
            max_ov = n - 1
            for k in range(max_ov, 0, -1):
                if b1[-k:] == b2[:k]:
                    overlap[i][j] = k
                    break
    
    # Initial order: standard order (index 0 to num_blocks-1)
    current_order = list(range(num_blocks))
    
    # Calculate initial total overlap
    def get_total_overlap(order):
        total = 0
        for i in range(len(order) - 1):
            total += overlap[order[i]][order[i+1]]
        return total
        
    current_overlap = get_total_overlap(current_order)
    best_overlap = current_overlap
    best_order = current_order[:]
    
    start_time = time.time()
    time_limit = 110
    
    # VNS: 2-opt
    # We try to reverse segments [i, j]
    # Improvement in overlap:
    # Old: O(i-1, i) + O(j, j+1)
    # New: O(i-1, j) + O(i, j+1)
    # (Assuming i > 0 and j < N-1)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        N = len(current_order)
        for i in range(N - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, N):
                # Calculate delta
                old_ov = 0
                new_ov = 0
                
                if i > 0:
                    old_ov += overlap[current_order[i-1]][current_order[i]]
                    new_ov += overlap[current_order[i-1]][current_order[j]]
                else:
                    # Start of sequence, no left neighbor
                    pass
                    
                if j < N - 1:
                    old_ov += overlap[current_order[j]][current_order[j+1]]
                    new_ov += overlap[current_order[j+1]][current_order[i]]
                else:
                    # End of sequence, no right neighbor
                    pass
                
                # Note: The internal overlaps within [i, j] are reversed.
                # Overlap(a, b) is not necessarily symmetric.
                # So reversing a segment changes ALL internal overlaps.
                # This makes O(1) update impossible unless we precompute prefix sums of overlaps?
                # But the overlaps change because the direction is reversed.
                # Overlap(A, B) != Overlap(B, A).
                
                # So we must calculate the new total overlap for the segment.
                # This is O(j-i).
                # Total complexity O(N^3) per pass?
                # N=720. N^3 is 373 million. Too slow for Python.
                
                # Optimization:
                # We can precompute the "cost" of a segment?
                # Or just use a smaller neighborhood.
                # Or accept that we do fewer iterations.
                
                # Let's try a different move: Swap adjacent blocks?
                # Or just do a few random 2-opts?
                
                # Given the time limit, let's try a simpler heuristic:
                # Greedy construction?
                # Start with a random block.
                # Always pick the next block with max overlap.
                # This is O(N^2). 720^2 = 500k. Fast.
                # We can do this multiple times with different starting points or random tie-breaking.
                
                # Let's abandon 2-opt for now and use Greedy + Local Search on the result.
                # Or just multiple Greedy runs.
                
                pass # Skip this 2-opt loop for now, implement Greedy below
                
        # Break out of while loop if we decide to use Greedy
        break
        
    # Implement Greedy Search
    # Run multiple times with different random seeds or starting nodes
    
    def greedy_search(start_idx, random_seed=0):
        random.seed(random_seed)
        unvisited = set(range(num_blocks))
        unvisited.remove(start_idx)
        current = start_idx
        order = [start_idx]
        
        while unvisited:
            # Find best next
            best_next = -1
            best_ov = -1
            # To add randomness, we can pick from top K
            candidates = []
            for next_idx in unvisited:
                ov = overlap[current][next_idx]
                candidates.append((ov, next_idx))
            
            # Sort by overlap desc
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick the best, or randomly from top few?
            # Let's pick the best to be deterministic first, then vary start.
            # Or pick randomly from top 5 to escape local optima?
            # Let's try picking the best.
            
            if candidates:
                best_next = candidates[0][1]
                order.append(best_next)
                unvisited.remove(best_next)
                current = best_next
            else:
                break
                
        return order
        
    # Run greedy from all possible starts? 720 runs.
    # Each run O(N^2) = 500k ops.
    # 720 * 500k = 360 million ops.
    # In Python, this might take 30-60 seconds.
    # We have 110s.
    
    best_order_greedy = None
    best_ov_greedy = -1
    
    # We can optimize by precomputing the "best next" for each node?
    # No, it depends on visited set.
    
    # Let's just run it.
    
    for start in range(num_blocks):
        if time.time() - start_time > time_limit - 10:
            break
            
        order = greedy_search(start)
        ov = get_total_overlap(order)
        if ov > best_ov_greedy:
            best_ov_greedy = ov
            best_order_greedy = order
            
    # Use the best greedy result
    if best_order_greedy:
        current_order = best_order_greedy
        current_overlap = best_ov_greedy
        
    # Now do a few 2-opt improvements on this result?
    # Since we have a good starting point, 2-opt might help.
    # But 2-opt is slow.
    # Let's try a few random 2-opt moves.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        N = len(current_order)
        # Try random 2-opt moves
        for _ in range(1000): # Fixed number of attempts
            if time.time() - start_time > time_limit:
                break
                
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            # Calculate new overlap
            # We need to sum overlaps in the reversed segment
            # Old segment overlaps: sum(overlap[current_order[k]][current_order[k+1]] for k in i..j-1)
            # New segment overlaps: sum(overlap[current_order[k+1]][current_order[k]] for k in i..j-1)
            # Plus boundary changes.
            
            old_seg_ov = 0
            new_seg_ov = 0
            for k in range(i, j):
                old_seg_ov += overlap[current_order[k]][current_order[k+1]]
                new_seg_ov += overlap[current_order[k+1]][current_order[k]]
                
            old_bound = 0
            new_bound = 0
            
            if i > 0:
                old_bound += overlap[current_order[i-1]][current_order[i]]
                new_bound += overlap[current_order[i-1]][current_order[j]]
            if j < N - 1:
                old_bound += overlap[current_order[j]][current_order[j+1]]
                new_bound += overlap[current_order[j+1]][current_order[i]]
                
            old_total_change = old_seg_ov + old_bound
            new_total_change = new_seg_ov + new_bound
            
            if new_total_change > old_total_change:
                # Apply move
                current_order[i:j+1] = current_order[i:j+1][::-1]
                current_overlap += (new_total_change - old_total_change)
                improved = True
                
    # Reconstruct word
    word = []
    for i, block_idx in enumerate(current_order):
        block = blocks[block_idx]
        # Add n
        word.append(n)
        # Add block
        word.extend(block)
        
    # Wait, the reconstruction logic must match the overlap calculation.
    # The word is formed by concatenating segments $B_i, n, B_i$.
    # Overlap is between the suffix of the *previous entire segment* and the prefix of the *current segment*.
    # Previous segment ends with $B_{i-1}$.
    # Current segment starts with $B_i$.
    # So we overlap suffix of $B_{i-1}$ with prefix of $B_i$.
    
    # Let's re-verify the word construction.
    # Segment i: $B_i, n, B_i$.
    # We append to the main word.
    # Main word ends with $B_{i-1}$.
    # We want to append $B_i, n, B_i$ such that it overlaps with $B_{i-1}$.
    # Overlap $k$ means $B_{i-1}[-k:] == B_i[:k]$.
    # So we append $B_i[k:], n, B_i$.
    
    # Let's rebuild correctly.
    final_word = []
    for i, block_idx in enumerate(current_order):
        block = list(blocks[block_idx])
        if i == 0:
            final_word.extend(block)
            final_word.append(n)
            final_word.extend(block)
        else:
            prev_block = list(blocks[current_order[i-1]])
            # Find overlap
            max_ov = n - 1
            ov = 0
            for k in range(max_ov, 0, -1):
                if prev_block[-k:] == block[:k]:
                    ov = k
                    break
            # Append block[ov:]
            final_word.extend(block[ov:])
            final_word.append(n)
            final_word.extend(block)
            
    return "".join(map(str, final_word))
