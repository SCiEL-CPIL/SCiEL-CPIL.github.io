import numpy as np
from itertools import permutations
import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Classic construction as baseline
    classic_word = _classic_construct(n)
    best_word = classic_word
    best_len = len(classic_word)
    
    # For n <= 6, classic is optimal or very close; skip heavy TSP
    if n >= 7 and n <= 7:
        # Try TSP-based construction
        try:
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)
        except Exception:
            pass
    
    return best_word

def _classic_construct(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return "1"
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
            # Find maximum overlap
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
    """
    TSP-based construction:
    1. Generate all n! permutations
    2. Compute overlap matrix
    3. Solve TSP approximately using nearest-neighbor + 2-opt
    4. Build word from optimal order
    """
    N = _factorial(n)
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute: for each l in 1..n-1, map prefix of length l -> list of indices
    prefix_maps = [{} for _ in range(n)]
    for l in range(1, n):
        for j in range(N):
            pref = perm_strs[j][:l]
            if pref not in prefix_maps[l]:
                prefix_maps[l][pref] = []
            prefix_maps[l][pref].append(j)
    
    # Compute overlap matrix
    overlap = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        si = perm_strs[i]
        for l in range(n - 1, 0, -1):
            suffix = si[-l:]
            if suffix in prefix_maps[l]:
                for j in prefix_maps[l][suffix]:
                    if overlap[i][j] == 0:
                        overlap[i][j] = l
    
    # Check time
    if time.time() > deadline - 30:
        return None
    
    # Now solve TSP: minimize sum of (n - overlap[path[k]][path[k+1]])
    # This is equivalent to maximizing sum of overlaps
    
    # Nearest neighbor heuristic
    visited = np.zeros(N, dtype=bool)
    path = [0]
    visited[0] = True
    curr = 0
    
    for _ in range(N - 1):
        # Find unvisited j with max overlap[curr][j]
        ov = overlap[curr].copy()
        ov[visited] = -1
        next_idx = np.argmax(ov)
        if ov[next_idx] < 0:
            break
        visited[next_idx] = True
        path.append(next_idx)
        curr = next_idx
        
        if time.time() > deadline - 20:
            return None
    
    # 2-opt improvement
    improved = True
    max_iters = 50
    iter_count = 0
    while improved and iter_count < max_iters and time.time() < deadline - 10:
        improved = False
        iter_count += 1
        for i in range(1, N - 1):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, N):
                # Consider reversing path[i:j+1]
                # Old edges: (i-1, i) and (j, j+1) if exists
                # New edges: (i-1, j) and (i, j+1) if exists
                
                old_cost = 0
                new_cost = 0
                
                # Edge (i-1, i)
                if i > 0:
                    old_cost += n - overlap[path[i-1]][path[i]]
                    new_cost += n - overlap[path[i-1]][path[j]]
                
                # Edge (j, j+1)
                if j < N - 1:
                    old_cost += n - overlap[path[j]][path[j+1]]
                    new_cost += n - overlap[path[i]][path[j+1]]
                
                if new_cost < old_cost:
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
    
    # Build the word
    s = perm_strs[path[0]]
    for k in range(1, N):
        p_prev = perm_strs[path[k-1]]
        p_curr = perm_strs[path[k]]
        # Find overlap
        ov = 0
        for l in range(n - 1, 0, -1):
            if p_prev[-l:] == p_curr[:l]:
                ov = l
                break
        s += p_curr[ov:]
    
    # Verify validity
    windows = set()
    for i in range(len(s) - n + 1):
        w = s[i:i+n]
        windows.add(w)
    
    if len(windows) == N:
        return s
    else:
        return None

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
