import math
import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate standard superpermutation up to n-1
    if n <= 1:
        return "1"
        
    word = [1]
    for m in range(2, n):
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
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if time.time() > deadline - 20:
        # Not enough time for optimization, return standard
        return "".join(map(str, word))

    # Now optimize the last step (m=n)
    # Get perms of n-1 from word
    perms, seen = [], set()
    for i in range(len(word) - n + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    num_perms = len(perms)
    
    # If num_perms is small, we can do better search
    # For n=7, num_perms = 720.
    
    # Calculate overlap cost between all pairs
    # cost[i][j] = length of block_i + length of block_j - max_overlap(block_i, block_j)
    # Actually, we just need max_overlap.
    # Block length is 2*(n-1)+1 = 2n-1.
    
    # To save time, we can just use Nearest Neighbor heuristic on the overlap matrix.
    # Precompute overlaps? 720x720 matrix is 518k entries.
    # Each overlap calc takes O(n^2) or O(n). n=7. Very fast.
    
    overlaps = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p_i = perms[i]
        block_i = list(p_i) + [n] + list(p_i)
        for j in range(num_perms):
            if i == j: continue
            p_j = perms[j]
            block_j = list(p_j) + [n] + list(p_j)
            # Max overlap of block_i suffix and block_j prefix
            max_o = min(len(block_i), len(block_j))
            o = 0
            for k in range(max_o, -1, -1):
                if k == 0 or block_i[len(block_i)-k:] == block_j[:k]:
                    o = k
                    break
            overlaps[i][j] = o
            
    # Nearest Neighbor
    visited = [False]*num_perms
    order = [0]
    visited[0] = True
    
    for _ in range(num_perms - 1):
        last = order[-1]
        best_next = -1
        best_overlap = -1
        for j in range(num_perms):
            if not visited[j]:
                if overlaps[last][j] > best_overlap:
                    best_overlap = overlaps[last][j]
                    best_next = j
        if best_next == -1: break
        order.append(best_next)
        visited[best_next] = True
        
    # Construct new word
    new_word = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(new_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                overlap = o
                break
        new_word += block[overlap:]
        
    # Check if new_word is shorter than word + standard step
    # Standard step would use 'perms' order.
    # Let's just check if new_word is valid and shorter than the previous best (which we don't have explicitly, but we know standard length).
    # Actually, we should compare with the result of the standard step.
    
    std_new_word = []
    for p in perms:
        block = list(p) + [n] + list(p)
        max_overlap = min(len(std_new_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0 or std_new_word[len(std_new_word) - o:] == block[:o]:
                overlap = o
                break
        std_new_word += block[overlap:]
        
    if len(new_word) < len(std_new_word):
        word = new_word
    else:
        word = std_new_word
        
    # Try to trim
    candidate = word[:-1]
    if len(candidate) >= n:
        seen_perms = set()
        for i in range(len(candidate) - n + 1):
            w = tuple(candidate[i:i+n])
            if len(set(w)) == n:
                seen_perms.add(w)
                if len(seen_perms) == math.factorial(n):
                    break
        if len(seen_perms) == math.factorial(n):
            word = candidate
            
    return "".join(map(str, word))
