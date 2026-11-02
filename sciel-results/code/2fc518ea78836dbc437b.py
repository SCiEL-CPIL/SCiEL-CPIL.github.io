import itertools
import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    if n < 7:
        return _classic_construct(n)
    
    if n > 7:
        # Fallback to classic for larger n, as search is too hard
        return _classic_construct(n)
    
    # n == 7
    symbols = list(range(1, n)) # 1..6
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms) # 720
    
    # Generate blocks
    # Block for perm p is p + (7,) + p
    blocks = []
    for p in perms:
        blocks.append(p + (n,) + p)
        
    # Precompute overlaps
    # overlap[i][j] is the max k such that blocks[i][-k:] == blocks[j][:k]
    # We want to maximize sum of overlaps.
    # Overlap matrix
    overlaps = [[0]*num_perms for _ in range(num_perms)]
    
    # Optimization: Since blocks are small, we can just compare strings/tuples
    # To speed up, we can convert to strings? Or just use tuples.
    # Tuples are fine.
    
    for i in range(num_perms):
        b_i = blocks[i]
        for j in range(num_perms):
            b_j = blocks[j]
            # Max possible overlap is min(len(b_i), len(b_j)) - 1 ? 
            # Actually, if b_i ends with b_j, overlap is len(b_j).
            # But they are distinct blocks? No, P is unique, so blocks are distinct.
            # Max overlap is len(b_i)-1?
            # b_i = P 7 P. b_j = Q 7 Q.
            # If P == Q, overlap is 13. But i != j so P != Q.
            # So max overlap is 12?
            # Let's just check from min(len) down to 0.
            max_ov = 0
            # We can limit search to reasonable overlaps.
            # Usually overlaps are small.
            for k in range(min(len(b_i), len(b_j)), -1, -1):
                if b_i[-k:] == b_j[:k]:
                    max_ov = k
                    break
            overlaps[i][j] = max_ov
            
    if time.time() > deadline:
        return _classic_construct(n)
        
    # Greedy TSP to find initial path
    # Start at 0
    visited = [False] * num_perms
    path = [0]
    visited[0] = True
    
    for _ in range(num_perms - 1):
        last = path[-1]
        best_next = -1
        best_ov = -1
        for j in range(num_perms):
            if not visited[j]:
                ov = overlaps[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        path.append(best_next)
        visited[best_next] = True
        
    if time.time() > deadline:
        return _classic_construct(n)
        
    # 2-opt improvement
    # We want to maximize total overlap.
    # Total overlap = sum(overlaps[path[i]][path[i+1]])
    
    def calc_total_overlap(p):
        total = 0
        for i in range(len(p) - 1):
            total += overlaps[p[i]][p[i+1]]
        return total
        
    current_path = path
    current_score = calc_total_overlap(current_path)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Standard 2-opt
        for i in range(num_perms - 1):
            for j in range(i + 1, num_perms):
                # Reverse segment i+1 to j
                # New path: ... p[i], p[j], p[j-1], ..., p[i+1], p[j+1] ...
                # Old edges: (p[i], p[i+1]) and (p[j], p[j+1])
                # New edges: (p[i], p[j]) and (p[i+1], p[j+1])
                # Note: indices in path
                a = current_path[i]
                b = current_path[i+1]
                c = current_path[j]
                d = current_path[j+1] if j + 1 < num_perms else None
                
                if d is None:
                    continue
                    
                old_cost = overlaps[a][b] + overlaps[c][d]
                new_cost = overlaps[a][c] + overlaps[b][d]
                
                if new_cost > old_cost:
                    # Reverse
                    current_path[i+1:j+1] = current_path[i+1:j+1][::-1]
                    current_score += new_cost - old_cost
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
                
    # Construct result
    result = []
    for idx in current_path:
        b = blocks[idx]
        if not result:
            result.extend(b)
        else:
            # Find max overlap with last part of result
            # Since we used the overlap matrix, we can just use the precomputed value
            # But the overlap might be different if the path changed?
            # No, the overlap is property of the two blocks.
            ov = overlaps[current_path[len(result) // len(blocks) - 1]][idx] # This is hard to index
            # Better: just compute overlap on the fly for the final assembly
            # Or store the overlaps in the path?
            # Let's just compute max overlap with the end of the current result list.
            # Result is a list of ints.
            # To be safe, compute max overlap.
            max_ov = 0
            # Max possible is 12
            for k in range(min(len(result), 12), -1, -1):
                if result[-k:] == list(b[:k]):
                    max_ov = k
                    break
            result.extend(b[max_ov:])
            
    return "".join(map(str, result))

def _classic_construct(n):
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
