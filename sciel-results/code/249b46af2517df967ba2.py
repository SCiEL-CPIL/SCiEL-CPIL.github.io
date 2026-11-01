import itertools
import random
import time

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Generate permutations of 1..n-1
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Precompute overlaps
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # Max overlap is n-2
    max_ov_possible = n - 2
    overlaps = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            ov = 0
            for k in range(max_ov_possible, -1, -1):
                if pi[-k:] == pj[:k]:
                    ov = k
                    break
            overlaps[i][j] = ov
    
    # Find best path using greedy with random restarts
    best_path = None
    best_len = float('inf')
    
    # Precompute for each node, sort neighbors by overlap descending
    # This helps speed up the greedy step
    neighbor_order = []
    for i in range(N):
        # Create list of (overlap, j) and sort by overlap desc, then randomize ties
        pairs = [(overlaps[i][j], j) for j in range(N) if j != i]
        pairs.sort(key=lambda x: (-x[0], x[1]))
        neighbor_order.append(pairs)
    
    restarts = 0
    while time.time() < deadline - 5:  # Leave 5s for deletion
        restarts += 1
        start_node = random.randint(0, N - 1)
        visited = [False] * N
        visited[start_node] = True
        current = start_node
        path = [start_node]
        current_overlap_sum = 0
        
        for _ in range(N - 1):
            max_ov = -1
            next_node = -1
            candidates = []
            # Iterate through pre-sorted neighbors
            for ov, j in neighbor_order[current]:
                if not visited[j]:
                    if ov > max_ov:
                        max_ov = ov
                        candidates = [j]
                    elif ov == max_ov:
                        candidates.append(j)
            
            if not candidates:
                break
            
            next_node = random.choice(candidates)
            visited[next_node] = True
            current_overlap_sum += max_ov
            current = next_node
            path.append(next_node)
        
        # Calculate length
        block_len = 2 * (n - 1) + 1
        total_len = block_len + (N - 1) * block_len - current_overlap_sum
        
        if total_len < best_len:
            best_len = total_len
            best_path = path
    
    # Construct the word
    if best_path is None:
        # Fallback: classic recursive construction
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    word = []
    first_block = list(perms[best_path[0]]) + [n] + list(perms[best_path[0]])
    word.extend(first_block)
    
    for k in range(1, N):
        prev_idx = best_path[k - 1]
        curr_idx = best_path[k]
        curr_p = perms[curr_idx]
        curr_block = list(curr_p) + [n] + list(curr_p)
        ov = overlaps[prev_idx][curr_idx]
        word.extend(curr_block[ov:])
    
    # Systematic Deletion Polish
    if time.time() < deadline - 2:
        from collections import defaultdict
        
        w = list(word)
        L = len(w)
        
        # Count occurrences of each permutation in the current string
        perm_count = defaultdict(int)
        
        for i in range(L - n + 1):
            p = tuple(w[i:i + n])
            perm_count[p] += 1
        
        def remove_char(idx):
            nonlocal L, perm_count
            # Windows affected: start from max(0, idx-n+1) to min(idx, L-n)
            # Old windows:
            for s in range(max(0, idx - n + 1), min(idx, L - n) + 1):
                p = tuple(w[s:s + n])
                perm_count[p] -= 1
                if perm_count[p] == 0:
                    # Invalid, revert
                    for s2 in range(max(0, idx - n + 1), min(idx, L - n) + 1):
                        perm_count[tuple(w[s2:s2 + n])] += 1
                    return False
            
            # Perform deletion
            w.pop(idx)
            L -= 1
            
            # Increment counts for new windows
            # The windows starting at s in [max(0, idx-n+1), min(idx, L-n)] in the NEW string
            # Note: L is already decreased.
            for s in range(max(0, idx - n + 1), min(idx, L - n) + 1):
                p = tuple(w[s:s + n])
                perm_count[p] += 1
                
            return True

        # Scan right to left to allow cascading deletions without index adjustment issues
        i = len(w) - 1
        while i >= 0 and time.time() < deadline - 2:
            if remove_char(i):
                # If successful, the element that was at i+1 is now at i.
                # We keep i the same to check the new element at i.
                pass
            else:
                i -= 1
        
        word = w
    
    return "".join(map(str, word))
