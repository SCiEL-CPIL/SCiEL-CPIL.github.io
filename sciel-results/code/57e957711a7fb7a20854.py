import time
import random
import itertools
import numpy as np
from collections import defaultdict

def compute_total_overlap(path, ov_list):
    """Compute total overlap for a given path ordering."""
    total = 0
    for i in range(len(path) - 1):
        total += ov_list[path[i]][path[i+1]]
    return total

def local_search_polish(path, ov_list, max_iters=1000, time_limit=None, start_time=None):
    """Improve path using 2-opt style moves."""
    best_path = path[:]
    best_score = compute_total_overlap(best_path, ov_list)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        # Try swapping adjacent elements
        for i in range(len(best_path) - 1):
            # Compute score change if we swap best_path[i] and best_path[i+1]
            old_score = ov_list[best_path[i-1] if i > 0 else best_path[i]][best_path[i]] if i > 0 else 0
            old_score += ov_list[best_path[i]][best_path[i+1]]
            if i + 1 < len(best_path) - 1:
                old_score += ov_list[best_path[i+1]][best_path[i+2]]
            
            new_score = 0
            if i > 0:
                new_score += ov_list[best_path[i-1]][best_path[i+1]]
            new_score += ov_list[best_path[i+1]][best_path[i]]
            if i + 1 < len(best_path) - 1:
                new_score += ov_list[best_path[i]][best_path[i+2]]
            
            if new_score > old_score:
                best_path[i], best_path[i+1] = best_path[i+1], best_path[i]
                best_score = new_score
                improved = True
                break
        
        # Try 2-opt: reverse a segment
        if not improved:
            for i in range(len(best_path) - 2):
                for j in range(i + 2, min(i + 5, len(best_path))):
                    # Score change for reversing best_path[i..j]
                    old_score = 0
                    new_score = 0
                    
                    # Before i
                    if i > 0:
                        old_score += ov_list[best_path[i-1]][best_path[i]]
                        new_score += ov_list[best_path[i-1]][best_path[j]]
                    
                    # Internal edges
                    for k in range(i, j):
                        old_score += ov_list[best_path[k]][best_path[k+1]]
                    for k in range(i, j):
                        new_score += ov_list[best_path[j-k]][best_path[j-k-1]]
                    
                    # After j
                    if j + 1 < len(best_path):
                        old_score += ov_list[best_path[j]][best_path[j+1]]
                        new_score += ov_list[best_path[i]][best_path[j+1]]
                    
                    if new_score > old_score:
                        best_path[i:j+1] = best_path[i:j+1][::-1]
                        best_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_path

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract distinct (m-1) permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # For m >= 4, use optimized greedy with multiple restarts
        if len(perms) > 5:
            # Check time
            if time.time() > deadline - 30:
                # Not enough time for optimization, use simple order
                ordered_perms = perms
            else:
                # Compute overlap matrix efficiently using numpy
                N = len(perms)
                max_ov = m - 1
                
                # Convert perms to numpy array for efficient comparison
                # Each perm is a tuple of length m-1
                # We'll compute overlap matrix using vectorized operations
                
                # For each pair (i,j), overlap is the max k such that
                # perms[i][-k:] == perms[j][:k]
                
                # Create numpy arrays for all perms
                perms_arr = np.array(perms, dtype=np.int8)  # shape (N, m-1)
                
                # overlap_matrix[i,j] = max k where perms[i][-(k):] == perms[j][:k]
                overlap_matrix = np.zeros((N, N), dtype=np.int8)
                
                # For each possible overlap length k from max_ov down to 1
                for k in range(max_ov, 0, -1):
                    # For each k, check which pairs have overlap >= k
                    # perms[i][-(k):] == perms[j][:k]
                    # This means: for all t in 0..k-1, perms[i][m-1-k+t] == perms[j][t]
                    
                    # Vectorized check:
                    # left = perms_arr[:, m-1-k:]  # shape (N, k)
                    # right = perms_arr[:, :k]      # shape (N, k)
                    # match = np.all(left[:, None, :] == right[None, :, :], axis=2)  # shape (N, N)
                    
                    # But this creates a large array. For N=720, k=6, it's 720*720*6 = 3M elements
                    # Let's do it more carefully
                    
                    left = perms_arr[:, m-1-k:]  # shape (N, k)
                    right = perms_arr[:, :k]      # shape (N, k)
                    
                    # Check element-wise equality
                    # match[i,j] = all(left[i,t] == right[j,t] for t in range(k))
                    # We can do: left[:, None, :] == right[None, :, :] gives (N, N, k)
                    # Then np.all(..., axis=2) gives (N, N)
                    
                    # For memory efficiency with N=720, k up to 6:
                    # (720, 720, 6) = 3.1M elements, ~3MB for int8. OK.
                    
                    if k <= 3:
                        # Direct comparison
                        match = np.all(left[:, None, :] == right[None, :, :], axis=2)
                    else:
                        # For larger k, do it in chunks to save memory
                        # Actually, let's just do it directly, it should be fine
                        match = np.all(left[:, None, :] == right[None, :, :], axis=2)
                    
                    # Only set where overlap_matrix is still 0 (we want max overlap)
                    overlap_matrix[match & (overlap_matrix == 0)] = k
                
                # Greedy path with multiple restarts
                # Determine number of restarts based on time remaining
                time_remaining = deadline - time.time() - 10  # Leave 10s for construction
                if time_remaining < 5:
                    n_restarts = 10
                elif time_remaining < 15:
                    n_restarts = 50
                elif time_remaining < 30:
                    n_restarts = 100
                else:
                    n_restarts = 200
                
                best_path = None
                best_score = -1
                
                # Convert to list of lists for faster access
                ov_list = overlap_matrix.tolist()
                
                for r in range(n_restarts):
                    # Random start
                    start = random.randint(0, N - 1)
                    path = [start]
                    visited = [False] * N
                    visited[start] = True
                    current = start
                    current_score = 0
                    
                    for _ in range(N - 1):
                        # Find next node with max overlap
                        best_next = -1
                        best_ov = -1
                        for j in range(N):
                            if not visited[j]:
                                ov = ov_list[current][j]
                                if ov > best_ov:
                                    best_ov = ov
                                    best_next = j
                        
                        if best_next == -1:
                            for j in range(N):
                                if not visited[j]:
                                    best_next = j
                                    break
                        
                        visited[best_next] = True
                        path.append(best_next)
                        current_score += ov_list[current][best_next]
                        current = best_next
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_path = path
                    
                    # Check time
                    if time.time() > deadline - 15:
                        break
                
                if best_path is not None:
                    ordered_perms = [perms[i] for i in best_path]
                else:
                    ordered_perms = perms
        
        else:
            ordered_perms = perms
        
        # Build the new word
        # For each permutation p in ordered_perms, we add: p + [m] + p
        # But we can overlap consecutive blocks
        new = []
        for idx in range(len(ordered_perms)):
            p = ordered_perms[idx]
            block = list(p) + [m] + list(p)
            
            if idx == 0:
                new += block
            else:
                # Find maximum overlap between end of new and start of block
                # new ends with previous p
                # block starts with current p
                # Max possible overlap is len(p) = m-1
                max_ov = min(len(new), len(block))
                ov = 0
                for k in range(max_ov, 0, -1):
                    if new[-k:] == block[:k]:
                        ov = k
                        break
                new += block[ov:]
        
        word = new
        
        if time.time() > deadline:
            break
    
    return "".join(map(str, word))
