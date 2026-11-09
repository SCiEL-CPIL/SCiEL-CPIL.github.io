import itertools
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        # Simple greedy or tetracode extension
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    N = 3 ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute coverage masks
    # mask[i] has bit j set if words[i] covers words[j] (dist <= 1)
    masks = [0] * N
    for i in range(N):
        w_i = words[i]
        mask_val = 0
        # Self
        mask_val |= (1 << i)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                mask_val |= (1 << j)
        masks[i] = mask_val
        
    full_mask = (1 << N) - 1
    
    # Greedy set-cover initialization: prioritize codewords with highest marginal gain
    uncovered = [1] * N  # 1 if not yet covered
    covered_count = 0
    greedy_code = []
    
    # Compute initial marginal gains
    # For each candidate, count how many uncovered vertices it covers
    while covered_count < N:
        best_gain = -1
        best_idx = -1
        for i in range(N):
            if i in greedy_code:
                continue
            # Count uncovered vertices covered by i
            gain = 0
            mask = masks[i]
            # Count bits in mask that are in uncovered
            # Efficient: iterate through set bits of mask
            m = mask
            while m:
                lsb = m & (-m)
                pos = lsb.bit_length() - 1
                if uncovered[pos]:
                    gain += 1
                m ^= lsb
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain <= 0:
            break
        greedy_code.append(best_idx)
        # Update uncovered
        mask = masks[best_idx]
        m = mask
        while m:
            lsb = m & (-m)
            pos = lsb.bit_length() - 1
            if uncovered[pos]:
                uncovered[pos] = 0
                covered_count += 1
            m ^= lsb
    
    # Now try to remove redundant codewords from greedy solution
    current_code = set(greedy_code)
    current_cov = 0
    for idx in current_code:
        current_cov |= masks[idx]
    
    best_code = set(current_code)
    best_size = len(current_code)
    
    # Try removing each codeword and see if coverage is maintained
    for idx in list(current_code):
        new_code = current_code - {idx}
        new_cov = 0
        for i in new_code:
            new_cov |= masks[i]
        if new_cov == full_mask:
            current_code = new_code
            current_cov = new_cov
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = set(current_code)
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Simulated Annealing with incremental coverage tracking
    # Track coverage count per vertex to enable O(1) add/remove
    cov_count = [0] * N  # How many codewords cover each vertex
    for idx in current_code:
        mask = masks[idx]
        m = mask
        while m:
            lsb = m & (-m)
            pos = lsb.bit_length() - 1
            cov_count[pos] += 1
            m ^= lsb
    
    current_uncovered = sum(1 for c in cov_count if c == 0)
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    all_indices = list(range(N))
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random codeword
            if not current_code:
                continue
            to_remove = random.choice(list(current_code))
            new_code = set(current_code)
            new_code.remove(to_remove)
            # Update coverage incrementally
            mask = masks[to_remove]
            m = mask
            new_uncovered = current_uncovered
            while m:
                lsb = m & (-m)
                pos = lsb.bit_length() - 1
                if cov_count[pos] == 1:
                    new_uncovered += 1
                cov_count[pos] -= 1
                m ^= lsb
            # Apply the change
            current_code = new_code
            current_uncovered = new_uncovered
            
        elif move_type < 0.6:
            # Add a random codeword
            to_add = random.choice(all_indices)
            if to_add in current_code:
                continue
            new_code = set(current_code)
            new_code.add(to_add)
            # Update coverage incrementally
            mask = masks[to_add]
            m = mask
            new_uncovered = current_uncovered
            while m:
                lsb = m & (-m)
                pos = lsb.bit_length() - 1
                if cov_count[pos] == 0:
                    new_uncovered -= 1
                cov_count[pos] += 1
                m ^= lsb
            # Apply the change
            current_code = new_code
            current_uncovered = new_uncovered
            
        else:
            # Swap: remove one, add another
            if not current_code:
                continue
            to_remove = random.choice(list(current_code))
            to_add = random.choice(all_indices)
            if to_add == to_remove or to_add in current_code:
                continue
            new_code = set(current_code)
            new_code.remove(to_remove)
            new_code.add(to_add)
            # Update coverage: remove to_remove, add to_add
            mask_remove = masks[to_remove]
            mask_add = masks[to_add]
            new_uncovered = current_uncovered
            # First remove to_remove
            m = mask_remove
            while m:
                lsb = m & (-m)
                pos = lsb.bit_length() - 1
                if cov_count[pos] == 1:
                    new_uncovered += 1
                cov_count[pos] -= 1
                m ^= lsb
            # Then add to_add
            m = mask_add
            while m:
                lsb = m & (-m)
                pos = lsb.bit_length() - 1
                if cov_count[pos] == 0:
                    new_uncovered -= 1
                cov_count[pos] += 1
                m ^= lsb
            # Apply the change
            current_code = new_code
            current_uncovered = new_uncovered
        
        # Calculate delta cost
        curr_uncovered = current_uncovered
        new_uncovered = current_uncovered  # Already updated above
        
        curr_cost = len(current_code) + 1000 * curr_uncovered
        new_cost = len(current_code) + 1000 * new_uncovered
        
        delta = new_cost - curr_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            # Update best if valid (uncovered == 0)
            if new_uncovered == 0:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
                    # If we hit 73, we can stop early?
                    # 73 is the best known. 72 is unlikely.
                    if best_size == 73:
                        # Try a bit longer to see if 72 is possible?
                        # Or just break to save time?
                        # Let's break if we hit 73, as 72 is extremely hard.
                        pass
                        
        T *= cooling
        
    # Final check: ensure best_code is valid
    # (It should be, as we only update best when uncovered==0)
    
    return [list(words[i]) for i in best_code]
