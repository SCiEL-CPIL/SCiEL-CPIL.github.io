import time
import itertools
from collections import defaultdict

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
    
    # Compute overlap matrix efficiently
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # Use string representation for faster comparison
    max_k = m_prev - 1  # Max possible overlap is n-2
    
    # Convert perms to strings for faster substring comparison
    perm_strs = [''.join(map(str, p)) for p in perms]
    
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        pi_str = perm_strs[i]
        for j in range(num_perms):
            if i == j:
                continue
            pj_str = perm_strs[j]
            # Find max overlap: pi_str suffix matches pj_str prefix
            # Check from max_k down to 1
            best = 0
            for k in range(max_k, 0, -1):
                if pi_str[-k:] == pj_str[:k]:
                    best = k
                    break
            overlap_matrix[i][j] = best
        if time.time() > deadline - 10:
            # Fallback: stop early, use what we have
            # Fill remaining with 0
            for ii in range(i + 1, num_perms):
                for jj in range(num_perms):
                    overlap_matrix[ii][jj] = 0
            break

    # Create orbit-based initial ordering
    # Group permutations by their first symbol
    # Then order groups to maximize transitions
    groups = defaultdict(list)
    for idx, p in enumerate(perms):
        groups[p[0]].append(idx)
    
    # Order the groups (first symbols) to maximize total overlap
    # Use greedy: start with group 1, then repeatedly add the group
    # that gives max average overlap with current last element
    symbols_in_order = list(range(1, n))
    
    # Better: use the overlap matrix to create an initial tour
    # Start with perm 0, then repeatedly pick the next perm that
    # gives maximum overlap with current last
    visited = [False] * num_perms
    order = [0]
    visited[0] = True
    
    for _ in range(num_perms - 1):
        last = order[-1]
        best_next = -1
        best_overlap = -1
        for j in range(num_perms):
            if not visited[j]:
                ov = overlap_matrix[last][j]
                if ov > best_overlap:
                    best_overlap = ov
                    best_next = j
        if best_next == -1:
            # No unvisited found (shouldn't happen)
            break
        order.append(best_next)
        visited[best_next] = True
    
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
    no_improve_count = 0
    max_no_improve = 3  # Stop after 3 full passes with no improvement
    
    while improved and time.time() < deadline - 5:
        improved = False
        no_improve_count = 0
        
        # 2-opt
        for i in range(num_perms - 1):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, num_perms):
                # Reverse segment i..j
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
                    best_order[i:j+1] = best_order[i:j+1][::-1]
                    best_overlap += delta
                    improved = True
        
        # Or-opt: move a segment of length 1, 2, or 3 to a new position
        for L in [1, 2, 3]:
            if time.time() > deadline - 5:
                break
            for i in range(num_perms - L):
                if time.time() > deadline - 5:
                    break
                # Extract segment best_order[i:i+L]
                seg = best_order[i:i+L]
                
                # Try inserting at each position j (not overlapping with i..i+L-1)
                for j in range(num_perms - L + 1):
                    if j >= i and j < i + L:
                        continue  # Skip if overlapping
                    
                    # Calculate delta for moving seg from i to j
                    # This is complex, so we'll do a simpler approach:
                    # Remove seg, then insert at j, and check if total improved
                    
                    # For now, skip Or-opt due to complexity and use 2-opt + swaps
                    pass
        
        # 1-opt: swap any two elements (covered by 2-opt for adjacent, but let's add non-adjacent)
        for i in range(num_perms):
            if time.time() > deadline - 5:
                break
            for j in range(i + 2, num_perms):  # Skip adjacent (covered by 2-opt)
                # Swap i and j
                # Current edges: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                # New edges: (i-1, j), (j, i+1), (j-1, i), (i, j+1)
                u1 = best_order[(i - 1) % num_perms]
                v1 = best_order[i]
                w1 = best_order[(i + 1) % num_perms]
                
                u2 = best_order[(j - 1) % num_perms]
                v2 = best_order[j]
                w2 = best_order[(j + 1) % num_perms]
                
                # If i and j are not adjacent, these are distinct
                old_cost = overlap_matrix[u1][v1] + overlap_matrix[v1][w1] + overlap_matrix[u2][v2] + overlap_matrix[v2][w2]
                new_cost = overlap_matrix[u1][v2] + overlap_matrix[v2][w1] + overlap_matrix[u2][v1] + overlap_matrix[v1][w2]
                
                delta = new_cost - old_cost
                if delta > 0:
                    best_order[i], best_order[j] = best_order[j], best_order[i]
                    best_overlap += delta
                    improved = True
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    # Construct the final word
    new_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(new_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0:
                overlap = 0
                break
            if new_word[len(new_word) - o:] == block[:o]:
                overlap = o
                break
        new_word += block[overlap:]
        
    return "".join(map(str, new_word))
