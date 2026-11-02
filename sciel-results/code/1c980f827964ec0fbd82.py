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
    
    # Compute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    max_k = m_prev - 1 # Max possible overlap is n-2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            # Check overlaps
            # pi suffix vs pj prefix
            # Max possible overlap is m_prev - 1
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap_matrix[i][j] = 0
                    break
                if pi[-k:] == pj[:k]:
                    overlap_matrix[i][j] = k
                    break
            if time.time() > deadline:
                # Fallback to natural order if too slow
                # This is unlikely for n=7 (720 perms)
                break
        if time.time() > deadline:
            break

    # Initial order: natural order
    order = list(range(num_perms))
    
    # Initial order: greedy nearest-neighbor
    used = [False] * num_perms
    order = [0]
    used[0] = True
    for _ in range(num_perms - 1):
        last = order[-1]
        best_next = -1
        best_ov = -1
        for j in range(num_perms):
            if not used[j]:
                ov = overlap_matrix[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        if best_next == -1:
            break
        order.append(best_next)
        used[best_next] = True
    
    # Calculate current total overlap (LINEAR, not circular)
    def calc_total_overlap(ord_list):
        total = 0
        for i in range(len(ord_list) - 1):
            total += overlap_matrix[ord_list[i]][ord_list[i + 1]]
        return total

    best_overlap = calc_total_overlap(order)
    best_order = order[:]
    
    # Local Search: single-element move (O(1) delta)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        for i in range(num_perms):
            if time.time() > deadline - 5:
                break
            elem = best_order[i]
            for j in range(num_perms):
                if j == i or j == i + 1:
                    continue
                
                if j < i:
                    del_left = overlap_matrix[best_order[i-1]][elem] if i > 0 else 0
                    del_right = overlap_matrix[elem][best_order[i+1]] if i < num_perms - 1 else 0
                    add_mid = overlap_matrix[best_order[i-1]][best_order[i+1]] if i > 0 and i < num_perms - 1 else 0
                    del_mid2 = overlap_matrix[best_order[j-1]][best_order[j]] if j > 0 else 0
                    add_left = overlap_matrix[best_order[j-1]][elem] if j > 0 else 0
                    add_right = overlap_matrix[elem][best_order[j]]
                else:
                    del_left = overlap_matrix[best_order[i-1]][elem] if i > 0 else 0
                    del_right = overlap_matrix[elem][best_order[i+1]] if i < num_perms - 1 else 0
                    add_mid = overlap_matrix[best_order[i-1]][best_order[i+1]] if i > 0 and i < num_perms - 1 else 0
                    del_mid2 = overlap_matrix[best_order[j]][best_order[j+1]] if j < num_perms - 1 else 0
                    add_left = overlap_matrix[best_order[j]][elem]
                    add_right = overlap_matrix[elem][best_order[j+1]] if j < num_perms - 1 else 0
                
                delta = (add_mid + add_left + add_right) - (del_left + del_right + del_mid2)
                
                if delta > 0:
                    best_order.pop(i)
                    if j > i:
                        j -= 1
                    best_order.insert(j, elem)
                    best_overlap += delta
                    improved = True
                    break
            if improved:
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
