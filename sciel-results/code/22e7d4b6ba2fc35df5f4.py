import numpy as np
from itertools import permutations
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Classic construction
    classic_word = _classic_construct(n)
    best_word = classic_word
    best_len = len(classic_word)
    
    # If n is small, classic is likely optimal or very close.
    # For n=7, we attempt TSP optimization.
    if n >= 4 and n <= 7:
        # For n=4,5,6 classic is optimal (length matches known shortest).
        # For n=7, classic is 5913, shortest is 5905.
        # TSP approach might find something better.
        if n == 7:
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)
        elif n < 7:
            # Could try TSP here too, but classic is already optimal for 4,5,6?
            # Let's check.
            # n=4: 33. n=5: 153. n=6: 873.
            # If TSP finds shorter, we take it.
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)

    return best_word

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
            # Find overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _tsp_construct(n, deadline):
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Convert to strings for faster overlap checks
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute overlap matrix efficiently
    # overlap[i][j] = longest suffix of perm_strs[i] matching prefix of perm_strs[j]
    overlap = np.zeros((N, N), dtype=np.int8)
    
    # For each i, compute overlaps with all j
    # Use the fact that max overlap is n-1
    max_ov = n - 1
    for i in range(N):
        si = perm_strs[i]
        # Get suffixes of si of lengths 1..max_ov
        # For each j, check if prefix of perm_strs[j] matches
        for j in range(N):
            sj = perm_strs[j]
            for l in range(max_ov, 0, -1):
                if si[-l:] == sj[:l]:
                    overlap[i][j] = l
                    break
    
    if time.time() > deadline - 15:
        return None
    
    # Nearest Neighbor with efficient unvisited tracking
    visited = np.zeros(N, dtype=bool)
    visited[0] = True
    path = [0]
    curr = 0
    
    # Track unvisited count for early termination
    unvisited_count = N - 1
    for step in range(N - 1):
        # Get overlaps from curr to all nodes
        row = overlap[curr]
        # Mask visited nodes with -1
        row_masked = row.copy()
        row_masked[visited] = -1
        next_idx = int(np.argmax(row_masked))
        if next_idx == 0 and visited[0]:
            # All visited
            break
        visited[next_idx] = True
        path.append(next_idx)
        curr = next_idx
        unvisited_count -= 1
        
        if step % 500 == 0 and time.time() > deadline - 20:
            return None
    
    if len(path) != N:
        return None
    
    # 2-Opt improvement - limited passes, early termination
    # For N=5040, full 2-opt is O(N^2) per pass. Limit to a few passes.
    improved = True
    max_passes = 5
    for p in range(max_passes):
        if time.time() > deadline - 10:
            break
        improved = False
        # Only do local 2-opt: check reversals of segments of length up to some limit
        # Full 2-opt is too slow. Use Or-opt style: try moving small segments.
        # Or do limited 2-opt with early break.
        # Let's do a simplified version: for each position i, try reversing path[i:j] for small j-i
        for i in range(1, N - 1):
            if time.time() > deadline - 8:
                break
            # Try a few j values
            for j in range(i + 1, min(i + 50, N)):
                # Compute cost change
                old_cost = 0
                new_cost = 0
                
                # Edge (i-1, i)
                old_cost += n - overlap[path[i-1]][path[i]]
                # Edge (j, j+1) if j < N-1
                if j < N - 1:
                    old_cost += n - overlap[path[j]][path[j+1]]
                
                # New edge (i-1, j)
                new_cost += n - overlap[path[i-1]][path[j]]
                # New edge (i, j+1) if j < N-1
                if j < N - 1:
                    new_cost += n - overlap[path[i]][path[j+1]]
                
                if new_cost < old_cost:
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
                    i += 1  # Skip ahead after modification
                    if i >= N - 1:
                        break
        
        if not improved:
            break
    
    # Build string
    s = perm_strs[path[0]]
    for k in range(1, N):
        p_prev = perm_strs[path[k-1]]
        p_curr = perm_strs[path[k]]
        ov = overlap[path[k-1]][path[k]]
        s += p_curr[ov:]
    
    # Verify validity
    if len(s) - n + 1 < N:
        return None
    
    windows = set()
    for i in range(len(s) - n + 1):
        w = s[i:i+n]
        windows.add(w)
    
    if len(windows) == N:
        return s
    else:
        return None
