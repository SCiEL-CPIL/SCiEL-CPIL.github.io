import time
import random
import math
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if n <= 6:
        # Standard recursive construction is optimal or near-optimal for small n
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
                # Find max overlap
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Generate all permutations of 1..6
        perms = list(permutations(range(1, 7)))
        N = len(perms)  # 720
        
        # Precompute overlap matrix
        # overlap[i][j] = longest suffix of perms[i] matching prefix of perms[j]
        # Since perms are permutations of 1..6, max overlap is 5.
        overlap_mat = np.zeros((N, N), dtype=np.int8)
        
        # Efficient method:
        # For k in 1..5:
        #   Map prefix_k -> list of indices
        #   For each i:
        #      suffix_k = perms[i][-k:]
        #      if suffix_k in map:
        #          for j in map[suffix_k]:
        #             overlap_mat[i][j] = k
        
        for k in range(1, 6):
            prefix_map = {}
            for j in range(N):
                key = tuple(perms[j][:k])
                if key not in prefix_map:
                    prefix_map[key] = []
                prefix_map[key].append(j)
            
            for i in range(N):
                key = tuple(perms[i][6-k:])
                if key in prefix_map:
                    for j in prefix_map[key]:
                        overlap_mat[i][j] = k

        # Objective: Maximize sum of overlaps in the path
        # Start with a random or greedy path
        random.seed(42)
        
        # Greedy initialization
        path = [0]
        unvisited = set(range(1, N))
        while unvisited:
            last = path[-1]
            best_next = -1
            best_ov = -1
            for j in unvisited:
                ov = overlap_mat[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
            path.append(best_next)
            unvisited.remove(best_next)
            
        current_score = sum(overlap_mat[path[i]][path[i+1]] for i in range(N-1))
        best_path = path[:]
        best_score = current_score
        
        # 2-Opt improvement
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(N - 1):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline - 10:
                        break
                    # Calculate delta for reversing segment i..j
                    p_i = path[i]
                    p_j = path[j]
                    
                    old_ov = 0
                    new_ov = 0
                    
                    if i > 0:
                        left = path[i-1]
                        old_ov += overlap_mat[left][p_i]
                        new_ov += overlap_mat[left][p_j]
                    
                    if j < N - 1:
                        right = path[j+1]
                        old_ov += overlap_mat[p_j][right]
                        new_ov += overlap_mat[p_i][right]
                    
                    # Only compute if segment is small? 
                    # For N=720, summing internal edges is O(N).
                    # Total O(N^3) is too slow.
                    # We will only try 2-Opt for small segments (j - i < 20)
                    if j - i > 20:
                        continue
                        
                    for k in range(i, j):
                        old_ov += overlap_mat[path[k]][path[k+1]]
                        new_ov += overlap_mat[path[k+1]][path[k]]
                    
                    delta = new_ov - old_ov
                    if delta > 0:
                        path[i:j+1] = path[i:j+1][::-1]
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_path = path[:]
                        improved = True
                        
            if current_score > best_score:
                best_score = current_score
                best_path = path[:]

        # Build final word
        def build_word_from_path(p):
            w = []
            for idx in p:
                perm = perms[idx]
                block = list(perm) + [7] + list(perm)
                overlap = 0
                min_len = min(len(w), len(block))
                for o in range(min_len, -1, -1):
                    if w[len(w) - o:] == block[:o]:
                        overlap = o
                        break
                w += block[overlap:]
            return "".join(map(str, w))

        final_word = build_word_from_path(best_path)
        return final_word

    # Fallback for n > 7
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
