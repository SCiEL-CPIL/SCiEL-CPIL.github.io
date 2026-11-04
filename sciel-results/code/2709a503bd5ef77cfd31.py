import time
import random

def construct(n):
    start_time = time.time()
    deadline = 145.0
    
    if n == 1:
        return "1"
    
    word = [1]
    
    for k in range(2, n + 1):
        if time.time() > deadline:
            break
            
        perms = []
        seen = set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        num_perms = len(perms)
        
        if num_perms == 0:
            continue
            
        if num_perms == 1:
            p = perms[0]
            block = list(p) + [k] + list(p)
            # Overlap
            max_overlap = min(len(word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if word[-o:] == block[:o]:
                    overlap = o
                    break
            word += block[overlap:]
            continue

        # Precompute overlap matrix
        # overlap[i][j] = max o such that suffix(perms[i], o) == prefix(perms[j], o)
        # Note: perms are tuples of length k-1
        # Max overlap is k-2 (if distinct)
        
        overlaps = [[0] * num_perms for _ in range(num_perms)]
        
        for i in range(num_perms):
            if time.time() > deadline:
                break
            pi = perms[i]
            for j in range(num_perms):
                if i == j:
                    continue
                pj = perms[j]
                # Check overlaps from k-2 down to 0
                max_o = k - 2
                # Optimization: if pi[-1] != pj[0], overlap 0
                # But we need to check suffix/prefix.
                found = 0
                for o in range(max_o, -1, -1):
                    if o == 0:
                        found = 0
                        break
                    # pi is tuple, slicing is O(o)
                    if pi[-o:] == pj[:o]:
                        found = o
                        break
                overlaps[i][j] = found

        # Greedy TSP
        best_order = None
        best_len = float('inf')
        
        num_starts = 20 if num_perms < 100 else 5
        if time.time() > deadline - 5:
            num_starts = 1
            
        for _ in range(num_starts):
            if time.time() > deadline - 1:
                break
                
            start_idx = random.randint(0, num_perms - 1)
            visited = [False] * num_perms
            order = [start_idx]
            visited[start_idx] = True
            current = start_idx
            current_len = 2*(k-1) + 1
            
            for _ in range(num_perms - 1):
                best_next = -1
                best_overlap = -1
                
                # Get overlaps from current
                row = overlaps[current]
                
                for next_idx in range(num_perms):
                    if visited[next_idx]:
                        continue
                    ov = row[next_idx]
                    if ov > best_overlap:
                        best_overlap = ov
                        best_next = next_idx
                        
                if best_next == -1:
                    break
                    
                visited[best_next] = True
                order.append(best_next)
                current_len += (2*(k-1) + 1) - best_overlap
                current = best_next
                
            if len(order) < num_perms:
                continue
                
            if current_len < best_len:
                best_len = current_len
                best_order = order
                
        if best_order is None:
            best_order = list(range(num_perms))
            
        # Construct new word
        new_word = []
        for idx, p_idx in enumerate(best_order):
            p = perms[p_idx]
            block = list(p) + [k] + list(p)
            
            if idx == 0:
                new_word = block
            else:
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(min(k, max_overlap), -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new_word[-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
                
        word = new_word

    return "".join(map(str, word))
