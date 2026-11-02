import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Standard construction
        word = [1]
        for m in range(2, n + 1):
            if time.time() > deadline:
                break
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 5, use the optimization strategy
    # 1. Build W_{n-1} standard
    # 2. Extract perms of n-1
    # 3. Optimize order of perms for inserting n
    
    # Build W_{n-1}
    word = [1]
    for m in range(2, n):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Extract perms of n-1 from word
    perms, seen = [], set()
    m_prev = n - 1
    for i in range(len(word) - m_prev + 1):
        window = tuple(word[i:i + m_prev])
        if len(set(window)) == m_prev and window not in seen:
            seen.add(window)
            perms.append(window)
            
    num_perms = len(perms)
    
    # Compute overlap matrix using string operations for speed
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    max_k = m_prev - 1
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    # Precompute suffixes and prefixes as strings for faster comparison
    perm_strs = [''.join(map(str, p)) for p in perms]
    
    for i in range(num_perms):
        pi_str = perm_strs[i]
        for j in range(num_perms):
            pj_str = perm_strs[j]
            # Check overlaps from largest to smallest
            for k in range(max_k, 0, -1):
                if pi_str[-k:] == pj_str[:k]:
                    overlap_matrix[i][j] = k
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break

    # Initial order: natural order
    order = list(range(num_perms))
    
    # Calculate current total overlap
    def calc_total_overlap(ord_list):
        total = 0
        for i in range(len(ord_list)):
            u = ord_list[i]
            v = ord_list[(i + 1) % len(ord_list)]
            total += overlap_matrix[u][v]
        return total

    best_overlap = calc_total_overlap(order)
    best_order = order[:]
    
    # Local Search: 2-opt and Or-opt
    improved = True
    while improved and time.time() < deadline - 8:
        improved = False
        
        # 2-opt
        for i in range(num_perms - 1):
            if time.time() > deadline - 8:
                break
            for j in range(i + 1, num_perms):
                u_idx = (i - 1) % num_perms
                v_idx = i
                w_idx = j
                x_idx = (j + 1) % num_perms
                
                u = best_order[u_idx]
                v = best_order[v_idx]
                w = best_order[w_idx]
                x = best_order[x_idx]
                
                delta = (overlap_matrix[u][w] + overlap_matrix[v][x]) - (overlap_matrix[u][v] + overlap_matrix[w][x])
                
                if delta > 0:
                    best_order[i:j+1] = reversed(best_order[i:j+1])
                    best_overlap += delta
                    improved = True
                    
        # Or-opt: move segments of length 1, 2, or 3
        for seg_len in [1, 2, 3]:
            if time.time() > deadline - 8:
                break
            for i in range(num_perms):
                if time.time() > deadline - 8:
                    break
                # Get the segment to move
                seg = best_order[i:i+seg_len]
                if len(seg) < seg_len:
                    continue
                    
                # Try inserting at all positions j (excluding the original position)
                for j in range(num_perms):
                    if j >= i and j < i + seg_len:
                        continue  # Skip original position
                    
                    # Calculate delta for moving seg from i to j
                    # Remove seg from position i, insert at position j
                    
                    # Case 1: j < i (insert before the segment's original position)
                    # Case 2: j >= i + seg_len (insert after the segment's original position)
                    
                    if j < i:
                        # After removal, positions shift. We need to be careful.
                        # Let's just try it and calculate the delta properly.
                        
                        # Before: ... A, [seg], B ...
                        # After: ... A, B, ..., [seg], ...
                        
                        # This is complex. Let's use a simpler approach:
                        # Just try the move and recalculate local overlap.
                        
                        # Save current state
                        old_order = best_order[:]
                        
                        # Perform move
                        best_order.pop(i)
                        best_order.pop(i) if seg_len > 1 else None
                        # Actually, let's do this more carefully
                        best_order = old_order  # Restore
                        new_order = best_order[:]
                        new_order.pop(i)
                        for _ in range(seg_len - 1):
                            new_order.pop(i)
                        # Insert at adjusted position
                        if j > i:
                            j -= seg_len
                        new_order.insert(j, seg[0])
                        for s in seg[1:]:
                            j += 1
                            new_order.insert(j, s)
                        
                        # Calculate overlap difference
                        old_total = 0
                        for idx in range(num_perms):
                            u = best_order[idx]
                            v = best_order[(idx + 1) % num_perms]
                            old_total += overlap_matrix[u][v]
                        
                        new_total = 0
                        for idx in range(num_perms):
                            u = new_order[idx]
                            v = new_order[(idx + 1) % num_perms]
                            new_total += overlap_matrix[u][v]
                        
                        delta = new_total - old_total
                        if delta > 0:
                            best_order = new_order
                            best_overlap += delta
                            improved = True
                            break  # Restart j loop after successful move
                    else:
                        # j >= i + seg_len
                        # After removal, position j becomes j - seg_len
                        j_adj = j - seg_len
                        
                        old_order = best_order[:]
                        new_order = best_order[:]
                        new_order.pop(i)
                        for _ in range(seg_len - 1):
                            new_order.pop(i)
                        new_order.insert(j_adj, seg[0])
                        for s in seg[1:]:
                            j_adj += 1
                            new_order.insert(j_adj, s)
                        
                        old_total = 0
                        for idx in range(num_perms):
                            u = best_order[idx]
                            v = best_order[(idx + 1) % num_perms]
                            old_total += overlap_matrix[u][v]
                        
                        new_total = 0
                        for idx in range(num_perms):
                            u = new_order[idx]
                            v = new_order[(idx + 1) % num_perms]
                            new_total += overlap_matrix[u][v]
                        
                        delta = new_total - old_total
                        if delta > 0:
                            best_order = new_order
                            best_overlap += delta
                            improved = True
                            break  # Restart j loop after successful move

    # Construct the final word efficiently
    # Build blocks and use a dictionary for fast overlap lookup
    blocks = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        blocks.append(block)
    
    new_word = []
    # For fast overlap, we'll track the last (m_prev) symbols
    # and check if they match the prefix of the next block
    
    for bi, block in enumerate(blocks):
        if bi == 0:
            new_word.extend(block)
        else:
            # Check for maximum overlap
            # The overlap can be at most m_prev (the length of a permutation)
            # Actually, the overlap is between the suffix of new_word and prefix of block
            # Maximum possible overlap is m_prev
            max_possible_overlap = min(len(new_word), m_prev, len(block))
            overlap = 0
            for o in range(max_possible_overlap, 0, -1):
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        
        if time.time() > deadline - 2:
            break
    
    return "".join(map(str, new_word))
