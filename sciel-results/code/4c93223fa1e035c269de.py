import itertools

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n > 7:
        # Fallback for larger n, though task is n=7
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

    # Heuristic for n=7
    prev_perms = list(itertools.permutations(range(1, n)))
    
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    num_blocks = len(blocks)
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if bi[-k:] == bj[:k]:
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
        
    # Calculate total overlap
    def get_total_overlap(p):
        total = 0
        for k in range(len(p)-1):
            total += overlaps[p[k]][p[k+1]]
        return total

    current_overlap = get_total_overlap(path)
    
    # Local Search: Adjacent Swaps
    improved = True
    max_iters = 100 # Safety limit
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(len(path)-1):
            # Try swapping path[i] and path[i+1]
            old_ov = overlaps[path[i-1]][path[i]] + overlaps[path[i]][path[i+1]] + overlaps[path[i+1]][path[i+2]] if i > 0 and i+2 < len(path) else 0
            # Calculate old contribution carefully
            # Indices involved: i-1, i, i+1, i+2
            # Old: ov[i-1, i] + ov[i, i+1] + ov[i+1, i+2]
            # New: ov[i-1, i+1] + ov[i+1, i] + ov[i, i+2]
            
            # Handle boundaries
            left_idx = i-1
            right_idx = i+2
            
            old_val = 0
            new_val = 0
            
            if left_idx >= 0:
                old_val += overlaps[path[left_idx]][path[i]]
                new_val += overlaps[path[left_idx]][path[i+1]]
            else:
                # No left neighbor, no cost saved/added from left
                pass

            old_val += overlaps[path[i]][path[i+1]]
            new_val += overlaps[path[i+1]][path[i]]
            
            if right_idx < len(path):
                old_val += overlaps[path[i+1]][path[right_idx]]
                new_val += overlaps[path[i]][path[right_idx]]
            
            if new_val > old_val:
                path[i], path[i+1] = path[i+1], path[i]
                current_overlap += (new_val - old_val)
                improved = True

    # Build the string
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
