import numpy as np
from itertools import permutations
import time
import sys
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Classic construction as baseline
    classic_word = _classic_construct(n)
    best_word = classic_word
    best_len = len(classic_word)
    
    if n >= 7:
        # Try greedy Hamiltonian path with multiple restarts
        try:
            t, t_len = _greedy_hamiltonian(n, deadline)
            if t is not None:
                if t_len < best_len:
                    best_word = t
                    best_len = t_len
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

def _greedy_hamiltonian(n, deadline):
    """
    Greedy Hamiltonian path construction with randomized restarts.
    
    Key idea: Instead of building a full overlap matrix (O(n!^2) memory/time),
    we use prefix/suffix indexing to quickly find candidates for each step.
    
    For each current permutation, we want to find the unvisited permutation
    with maximum overlap. We precompute for each suffix length l, a map from
    suffix string to list of permutation indices. This allows O(n) lookup per
    candidate check.
    """
    N = _factorial(n)
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute prefix maps: for each l in 1..n-1, map prefix of length l -> list of indices
    prefix_maps = [dict() for _ in range(n)]
    for l in range(1, n):
        for j in range(N):
            pref = perm_strs[j][:l]
            if pref not in prefix_maps[l]:
                prefix_maps[l][pref] = []
            prefix_maps[l][pref].append(j)
    
    # Precompute suffix for each permutation for quick lookup
    suffixes = {}
    for i in range(N):
        suffixes[i] = {}
        for l in range(1, n):
            suffixes[i][l] = perm_strs[i][-l:]
    
    best_word = None
    best_len = N * n  # Upper bound
    
    # Multiple restarts with different starting permutations
    max_restarts = 50
    random.seed(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Choose starting permutation
        if restart == 0:
            start = 0
        else:
            start = random.randint(0, N - 1)
        
        visited = [False] * N
        path = [start]
        visited[start] = True
        curr = start
        
        # Greedy: at each step, pick unvisited permutation with max overlap
        for step in range(N - 1):
            if time.time() > deadline - 5:
                break
            
            # Find best next permutation
            best_ov = -1
            best_next = -1
            
            # Check overlaps from n-1 down to 1
            for l in range(n - 1, 0, -1):
                suffix = suffixes[curr][l]
                candidates = prefix_maps[l].get(suffix, [])
                for j in candidates:
                    if not visited[j]:
                        if l > best_ov:
                            best_ov = l
                            best_next = j
                if best_ov == n - 1:
                    break  # Can't do better
            
            if best_next == -1:
                # No overlap found, pick random unvisited
                unvisited = [i for i in range(N) if not visited[i]]
                if not unvisited:
                    break
                best_next = random.choice(unvisited)
                best_ov = 0
            
            visited[best_next] = True
            path.append(best_next)
            curr = best_next
        
        # Check if we got a complete path
        if len(path) < N:
            continue
        
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
            if len(s) < best_len:
                best_len = len(s)
                best_word = s
    
    return best_word, best_len

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
