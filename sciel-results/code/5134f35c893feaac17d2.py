import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Generate standard superpermutation for n-1
    # This is the "base" set of blocks
    def get_base_superperm(k):
        if k == 1:
            return [1]
        if k == 2:
            return [1, 2, 1]
        prev = get_base_superperm(k - 1)
        # Extract permutations of k-1 from prev
        perms = []
        seen = set()
        for i in range(len(prev) - (k - 2)):
            window = tuple(prev[i:i + k - 1])
            if len(set(window)) == k - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(list(window))
        
        new_word = []
        for p in perms:
            # Standard construction: p, k, p
            segment = p + [k] + p
            # Overlap with previous part of new_word
            # We want to overlap the suffix of new_word with prefix of segment
            # But actually, the standard construction is defined such that 
            # the overlap is handled by the permutation order in the recursive step.
            # Here we are just building the string for n-1 to extract blocks for n.
            # We can just concatenate with maximum overlap to keep it short, 
            # although the *order* of blocks is what matters for the next step.
            # The standard order is the one that produces the shortest string for n-1.
            
            # Let's just use the standard recursive string generation which is known to be optimal for small n
            # and provides a specific block order.
            
            # Actually, simpler: just build the string directly.
            if not new_word:
                new_word = segment[:]
            else:
                # Find max overlap between suffix of new_word and prefix of segment
                max_ov = 0
                max_possible = min(len(new_word), len(segment))
                for ov in range(max_possible, 0, -1):
                    if new_word[-ov:] == segment[:ov]:
                        max_ov = ov
                        break
                new_word.extend(segment[max_ov:])
        return new_word

    base_list = get_base_superperm(n - 1)
    
    # Step 2: Extract blocks (permutations of n-1) from base_list
    # The blocks are the distinct permutations of 1..n-1 appearing in base_list
    # In the standard superpermutation, they appear in a specific order.
    blocks = []
    seen = set()
    for i in range(len(base_list) - (n - 2)):
        window = tuple(base_list[i:i + n - 1])
        if len(set(window)) == n - 1:
            if window not in seen:
                seen.add(window)
                blocks.append(list(window))
                
    # We now have a list of blocks. We want to reorder them to maximize overlaps.
    # Overlap between block A and block B is the longest suffix of A matching prefix of B.
    
    def calc_overlap(a, b):
        max_ov = 0
        max_possible = min(len(a), len(b))
        for ov in range(max_possible, 0, -1):
            if a[-ov:] == b[:ov]:
                return ov
        return 0

    def total_overlap(order):
        ov_sum = 0
        for i in range(len(order) - 1):
            ov_sum += calc_overlap(order[i], order[i+1])
        return ov_sum

    # Initial order is the one from the standard construction
    current_order = blocks[:]
    current_ov = total_overlap(current_order)
    best_order = current_order[:]
    best_ov = current_ov
    
    start_time = time.time()
    time_limit = 115 # Leave 5s for safety
    
    # VNS Search
    # Neighborhoods:
    # 1. Swap adjacent
    # 2. Reverse segment (2-opt)
    # 3. Move element (Or-opt)
    
    # Precompute overlaps for speed?
    # N = (n-1)! = 720 for n=7.
    # Overlap matrix is 720x720. 518k entries.
    # This is feasible to precompute to speed up evaluation.
    
    # Map block index to block
    # We operate on indices for speed, then map back to blocks for overlap calc
    # Or just use the blocks directly if N is small enough.
    # For n=7, N=720. 720^2 = 518,400.
    # calc_overlap is O(n) = O(6). Total ~3M ops. Very fast.
    
    # Let's precompute the overlap matrix between all pairs of blocks
    # to make local search moves O(1) or O(log N) if we maintain sums, 
    # but simple recalculation of local changes is fine.
    
    # Actually, for a move, we only change 2 or 3 overlaps.
    # Swap i, i+1:
    # Old: ov(i-1, i) + ov(i, i+1) + ov(i+1, i+2)
    # New: ov(i-1, i+1) + ov(i+1, i) + ov(i, i+2)
    # We can compute these on the fly.
    
    def try_swap(order, i, j):
        # Swap i and j
        # Only valid if |i-j| is small for efficiency? 
        # VNS usually starts with small neighborhoods.
        
        # Calculate change in overlap
        # Indices involved: i-1, i, i+1 ... j-1, j, j+1
        # If we swap i and j, the overlaps affected are:
        # (i-1, i), (i, i+1) ... (j-1, j), (j, j+1)
        # This is complex for distant swaps.
        # Let's stick to adjacent swaps and local reversals first.
        
        pass

    # Simpler VNS:
    # N1: Adjacent swap
    # N2: Reverse segment of length 2..K
    # N3: Move block to random position
    
    improved = True
    neighborhood_idx = 0
    max_neighborhoods = 3
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        if neighborhood_idx == 0:
            # N1: Adjacent swaps
            for i in range(len(current_order) - 1):
                # Try swapping i and i+1
                # Old overlaps: ov(i-1, i) + ov(i, i+1) + ov(i+1, i+2)
                # New overlaps: ov(i-1, i+1) + ov(i+1, i) + ov(i, i+2)
                
                b_i = current_order[i]
                b_j = current_order[i+1]
                
                ov_old = 0
                ov_new = 0
                
                if i > 0:
                    b_prev = current_order[i-1]
                    ov_old += calc_overlap(b_prev, b_i)
                    ov_new += calc_overlap(b_prev, b_j)
                
                # Middle
                ov_old += calc_overlap(b_i, b_j)
                ov_new += calc_overlap(b_j, b_i)
                
                if i + 2 < len(current_order):
                    b_next = current_order[i+2]
                    ov_old += calc_overlap(b_j, b_next)
                    ov_new += calc_overlap(b_i, b_next)
                
                if ov_new > ov_old:
                    # Perform swap
                    current_order[i], current_order[i+1] = b_j, b_i
                    current_ov += (ov_new - ov_old)
                    if current_ov > best_ov:
                        best_ov = current_ov
                        best_order = current_order[:]
                    improved = True
                    # Restart N1? Or continue? VNS usually restarts or moves to next if no improvement.
                    # Let's restart N1 to exploit local gains.
                    break
            if improved:
                continue # Restart N1
            else:
                neighborhood_idx = 1
                continue
                
        elif neighborhood_idx == 1:
            # N2: Reverse segment
            # Try reversing segments of length 2 to 10
            seg_len = 2
            found_improvement = False
            while seg_len <= 10:
                for i in range(len(current_order) - seg_len + 1):
                    # Reverse i to i+seg_len-1
                    # This affects overlaps:
                    # (i-1, i) -> (i-1, i+seg_len-1)
                    # (i, i+1) ... (i+seg_len-2, i+seg_len-1) -> reversed internal overlaps
                    # (i+seg_len-1, i+seg_len) -> (i, i+seg_len)
                    
                    # Calculate old overlaps
                    ov_old = 0
                    if i > 0:
                        ov_old += calc_overlap(current_order[i-1], current_order[i])
                    
                    for k in range(i, i + seg_len - 1):
                        ov_old += calc_overlap(current_order[k], current_order[k+1])
                        
                    if i + seg_len < len(current_order):
                        ov_old += calc_overlap(current_order[i+seg_len-1], current_order[i+seg_len])
                    
                    # Calculate new overlaps
                    ov_new = 0
                    if i > 0:
                        ov_new += calc_overlap(current_order[i-1], current_order[i+seg_len-1])
                    
                    # Internal reversed
                    for k in range(i, i + seg_len - 1):
                        # Original k, k+1 becomes new k+seg_len-1, k+seg_len-2 ?
                        # Let's just extract the segment, reverse it, and calc
                        pass
                    
                    # Optimization: Just extract, reverse, calc total overlap of the whole thing?
                    # No, that's O(N).
                    # Let's just do a simple local check for small seg_len.
                    
                    # For simplicity and robustness, let's just try swapping distant elements (2-opt)
                    # which is a specific case of reversal.
                    
                    # Let's implement 2-opt properly:
                    # Reverse i..j
                    if seg_len == 2:
                        j = i + 1
                        # This is just a swap, already handled in N1? 
                        # Swap is not reversal of length 2 in terms of order change? 
                        # Reversal of [A, B] is [B, A]. Yes, it is a swap.
                        # So N1 covers 2-opt for length 2.
                        continue
                    
                    j = i + seg_len - 1
                    b_start = current_order[i]
                    b_end = current_order[j]
                    
                    # Old boundary overlaps
                    ov_old_bound = 0
                    if i > 0:
                        ov_old_bound += calc_overlap(current_order[i-1], b_start)
                    if j + 1 < len(current_order):
                        ov_old_bound += calc_overlap(b_end, current_order[j+1])
                    
                    # New boundary overlaps
                    ov_new_bound = 0
                    if i > 0:
                        ov_new_bound += calc_overlap(current_order[i-1], b_end)
                    if j + 1 < len(current_order):
                        ov_new_bound += calc_overlap(b_start, current_order[j+1])
                    
                    # Internal overlaps
                    # Old internal: ov(i, i+1) + ... + ov(j-1, j)
                    # New internal: ov(j, j-1) + ... + ov(i+1, i)
                    ov_old_int = 0
                    ov_new_int = 0
                    for k in range(i, j):
                        ov_old_int += calc_overlap(current_order[k], current_order[k+1])
                        ov_new_int += calc_overlap(current_order[k+1], current_order[k])
                    
                    delta = (ov_new_bound + ov_new_int) - (ov_old_bound + ov_old_int)
                    
                    if delta > 0:
                        # Perform reversal
                        current_order[i:j+1] = current_order[i:j+1][::-1]
                        current_ov += delta
                        if current_ov > best_ov:
                            best_ov = current_ov
                            best_order = current_order[:]
                        found_improvement = True
                        improved = True
                        # Restart N2 with same seg_len or smaller?
                        # Let's restart N2 from seg_len=2
                        seg_len = 2
                        break
                
                if found_improvement:
                    continue
                else:
                    seg_len += 1
                    if seg_len > 10:
                        neighborhood_idx = 2
                        continue
            if not found_improvement and neighborhood_idx == 1:
                # If no improvement in N2, move to N3
                pass
                
        else:
            # N3: Random Move (Or-opt)
            # Pick a random block and move it to a random position
            i = random.randint(0, len(current_order) - 1)
            j = random.randint(0, len(current_order) - 1)
            if i == j:
                continue
                
            b_move = current_order[i]
            
            # Calculate old overlaps involving i
            ov_old = 0
            if i > 0:
                ov_old += calc_overlap(current_order[i-1], b_move)
            if i + 1 < len(current_order):
                ov_old += calc_overlap(b_move, current_order[i+1])
            # Also the overlap between i-1 and i+1 disappears and is replaced by i-1 and i+1?
            # Wait, if we remove i, i-1 and i+1 become adjacent.
            # So old: ov(i-1, i) + ov(i, i+1)
            # New (at old pos): ov(i-1, i+1)
            
            # Calculate new overlaps at position j
            # We insert b_move at j.
            # If j > i, indices shift.
            # Let's just construct the new list and calc the local changes carefully.
            
            # Simpler: 
            # 1. Remove b_move from i.
            # 2. Insert at j.
            
            # Let's just do a full recalculation of the *affected* overlaps.
            # This is tricky with indices shifting.
            
            # Alternative: Just try a few random swaps/reversals if N1 and N2 fail.
            # Or, reset to best_order and start over?
            # VNS typically shakes the neighborhood.
            
            # Let's try a random 2-opt on best_order if stuck.
            # To save time, let's just break and reconstruct.
            # Given the time limit, if N1 and N2 (up to len 10) fail to improve,
            # further search is unlikely to yield significant gains in the remaining time.
            
            break

    # Reconstruct the word from best_order
    # Structure: B_1 n B_1 B_2 n B_2 ...
    word = []
    for i, block in enumerate(best_order):
        if i == 0:
            word.extend(block)
            word.append(n)
            word.extend(block)
        else:
            prev_block = best_order[i-1]
            # Overlap between prev_block (end of previous segment) and current block (start of new segment)
            # The previous segment ended with prev_block.
            # The new segment starts with current_block.
            # Overlap is suffix of prev_block with prefix of current_block.
            
            max_ov = 0
            max_possible = min(len(prev_block), len(current_block))
            for ov in range(max_possible, 0, -1):
                if prev_block[-ov:] == current_block[:ov]:
                    max_ov = ov
                    break
            
            # Add current_block starting from max_ov
            word.extend(current_block[max_ov:])
            word.append(n)
            word.extend(current_block)
            
    return "".join(map(str, word))
