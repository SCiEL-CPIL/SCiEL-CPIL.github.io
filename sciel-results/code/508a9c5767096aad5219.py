import itertools
import sys
import time

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n > 7:
        # Fallback to standard construction for larger n
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
        return "".join(map(str, word))

    # Heuristic for n=7 (and potentially 4,5,6)
    # Generate all permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    num_blocks = len(blocks)
    block_len = len(blocks[0])
    
    # Precompute overlaps using string representation for speed
    block_strs = ["".join(map(str, b)) for b in blocks]
    
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi_str = block_strs[i]
        for j in range(num_blocks):
            bj_str = block_strs[j]
            max_k = min(len(bi_str), len(bj_str))
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if bi_str[-k:] == bj_str[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    # Greedy TSP Heuristic
    current = 0
    unvisited = set(range(1, num_blocks))
    path = [0]
    
    while unvisited:
        best_next = -1
        best_overlap = -1
        for j in unvisited:
            if overlaps[current][j] > best_overlap:
                best_overlap = overlaps[current][j]
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next

    def calc_length(path):
        total = len(blocks[path[0]])
        for k in range(len(path)-1):
            total += len(blocks[path[k+1]]) - overlaps[path[k]][path[k+1]]
        return total

    def calc_total_overlap(path):
        total = 0
        for k in range(len(path)-1):
            total += overlaps[path[k]][path[k+1]]
        return total

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    # Or-Opt + 2-Opt Local Search
    improved = True
    max_iterations = 100
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iter_count += 1
        
        # Or-Opt: move single node
        for k in range(1, len(path)-1):
            if time.time() > deadline:
                break
            node = path[k]
            prev_node = path[k-1]
            next_node = path[k+1]
            
            old_ov = overlaps[prev_node][node] + overlaps[node][next_node]
            new_ov_removal = overlaps[prev_node][next_node]
            
            best_gain = 0
            best_pos = -1
            
            for m in range(0, len(path)-1):
                if m == k-1 or m == k:
                    continue
                
                left_node = path[m]
                right_node = path[m+1]
                
                new_ov_insertion = overlaps[left_node][node] + overlaps[node][right_node]
                gain = (new_ov_insertion + new_ov_removal) - old_ov
                
                if gain > best_gain:
                    best_gain = gain
                    best_pos = m
            
            if best_gain > 0 and best_pos != -1:
                path.pop(k)
                if k < best_pos:
                    best_pos -= 1
                path.insert(best_pos+1, node)
                improved = True
        
        # 2-Opt: reverse a segment
        for i in range(len(path)-1):
            if time.time() > deadline:
                break
            for j in range(i+2, len(path)):
                if time.time() > deadline:
                    break
                
                # Reverse path[i+1:j+1]
                # Old: A -> B -> ... -> C -> D
                # New: A -> C -> ... -> B -> D
                A = path[i]
                B = path[i+1]
                C = path[j]
                D = path[j+1] if j+1 < len(path) else None
                
                # Old overlaps at boundaries: ov(A,B) + ov(C,D) if D exists
                old_boundary = overlaps[A][B]
                if D is not None:
                    old_boundary += overlaps[C][D]
                
                # New overlaps at boundaries: ov(A,C) + ov(B,D) if D exists
                new_boundary = overlaps[A][C]
                if D is not None:
                    new_boundary += overlaps[B][D]
                
                # Internal overlaps change: sum of ov(path[t], path[t+1]) for t in [i+1, j-1]
                # becomes sum of ov(path[t+1], path[t]) for t in [i+1, j-1]
                # i.e., ov(X,Y) becomes ov(Y,X)
                
                old_internal = 0
                new_internal = 0
                for t in range(i+1, j):
                    old_internal += overlaps[path[t]][path[t+1]]
                    new_internal += overlaps[path[t+1]][path[t]]
                
                gain = (new_boundary + new_internal) - (old_boundary + old_internal)
                
                if gain > 0:
                    path[i+1:j+1] = reversed(path[i+1:j+1])
                    improved = True

    # Build the string
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
