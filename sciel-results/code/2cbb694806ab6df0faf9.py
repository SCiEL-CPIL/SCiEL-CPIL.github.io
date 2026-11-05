import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, 10s margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132412132413214321" # Standard construction result? 
        # Actually, let's just use the recursive builder for small n to be safe, 
        # or hardcode if known. 
        # For n=7, we use the TSP approach.
        
    # For n=7, we use the optimized construction.
    # For other n, we can use the standard recursive construction.
    # The task specifies n=7.
    
    if n != 7:
        # Fallback to standard recursive construction for other n
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
                # Find overlap
                max_overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_overlap = o
                        break
                new += block[max_overlap:]
            word = new
        return "".join(map(str, word))

    # Specific logic for n=7
    symbols = list(range(1, 7)) # 1..6
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # We only care about k up to 5.
    overlaps = [[0] * N for _ in range(N)]
    
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            max_k = 0
            # Check k=5 down to 1
            # pi[1:] == pj[:5]
            if pi[1:] == pj[:5]:
                max_k = 5
            elif pi[2:] == pj[:4]:
                max_k = 4
            elif pi[3:] == pj[:3]:
                max_k = 3
            elif pi[4:] == pj[:2]:
                max_k = 2
            elif pi[5:] == pj[:1]:
                max_k = 1
            overlaps[i][j] = max_k
            
    # Greedy Nearest Neighbor
    path = [0]
    visited = [False] * N
    visited[0] = True
    current = 0
    
    for _ in range(N - 1):
        best_next = -1
        best_overlap = -1
        for j in range(N):
            if not visited[j]:
                if overlaps[current][j] > best_overlap:
                    best_overlap = overlaps[current][j]
                    best_next = j
        path.append(best_next)
        visited[best_next] = True
        current = best_next
        
    # 2-Opt Improvement
    # Calculate total overlap of current path
    def get_path_overlap(p):
        total = 0
        for i in range(len(p) - 1):
            total += overlaps[p[i]][p[i+1]]
        return total
        
    current_overlap = get_path_overlap(path)
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, N):
                # Reverse path[i+1 : j+1]
                # Old edges: (p[i], p[i+1]) and (p[j], p[j+1]) if j < N-1
                # New edges: (p[i], p[j]) and (p[i+1], p[j+1]) if j < N-1
                
                gain = 0
                # Remove old edges
                gain -= overlaps[path[i]][path[i+1]]
                if j < N - 1:
                    gain -= overlaps[path[j]][path[j+1]]
                else:
                    # Last edge in path is not followed by anything in the sum?
                    # The sum is over consecutive pairs in the path.
                    # The path is a sequence.
                    # Edges are (0,1), (1,2), ..., (N-2, N-1).
                    # If we reverse i+1..j, the edges affected are:
                    # (i, i+1) becomes (i, j)
                    # (j, j+1) becomes (i+1, j+1) -- if j < N-1
                    # If j == N-1, the edge (j, j+1) doesn't exist.
                    # Wait, the path is a list of nodes.
                    # Overlap is sum of overlaps[path[k]][path[k+1]] for k=0..N-2.
                    
                    pass
                
                # Add new edges
                gain += overlaps[path[i]][path[j]]
                if j < N - 1:
                    gain += overlaps[path[i+1]][path[j+1]]
                
                if gain > 0:
                    # Perform reversal
                    path[i+1 : j+1] = reversed(path[i+1 : j+1])
                    current_overlap += gain
                    improved = True
        
    # Construct word
    word = []
    for idx, p_idx in enumerate(path):
        p = perms[p_idx]
        block = list(p) + [7] + list(p)
        
        if idx == 0:
            word.extend(block)
        else:
            # Find overlap with current word end
            # Suffix of word vs Prefix of block
            # Max overlap is 5 (based on P overlap)
            # But we can just check up to len(block)
            max_ov = 0
            # Optimization: we know the theoretical max is 5
            # But let's be safe and check up to 5
            for k in range(5, -1, -1):
                if word[-k:] == block[:k]:
                    max_ov = k
                    break
            word.extend(block[max_ov:])
            
    # Verify
    # Check if all 5040 permutations are present
    # Generate all windows of length 7
    windows = set()
    for i in range(len(word) - 6):
        windows.add(tuple(word[i:i+7]))
        
    # Check if all perms of 1..7 are in windows
    # 5040 perms
    all_perms = set(itertools.permutations(range(1, 8)))
    
    if windows == all_perms:
        return "".join(map(str, word))
    else:
        # Fallback to standard construction if invalid (should not happen)
        # Or return the word anyway if we are out of time? 
        # Better to return a valid one.
        # Let's quickly build the standard one
        word_std = [1]
        for m in range(2, 8):
            perms_std, seen = [], set()
            for i in range(len(word_std) - m + 2):
                window = tuple(word_std[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_std.append(window)
            new = []
            for p in perms_std:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word_std = new
        return "".join(map(str, word_std))
