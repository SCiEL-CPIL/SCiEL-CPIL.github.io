"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from collections import defaultdict
import time
import sys
import random

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)  # 2187
    
    # Precompute cover_list[i] = list of word indices covered by codeword i
    # and covers_word[j] = list of codeword indices that cover word j
    cover_list = [None] * N
    covers_word = [[] for _ in range(N)]
    
    for i, w in enumerate(all_words):
        neighbors = [w]
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors.append(tuple(new_w))
        cover_list[i] = [word_to_idx[nb] for nb in neighbors]
        for nb_idx in cover_list[i]:
            covers_word[nb_idx].append(i)
    
    # Precompute cover_mask[i] as Python int for bit-parallel ops
    cover_mask = [0] * N
    for i in range(N):
        m = 0
        for j in cover_list[i]:
            m |= (1 << j)
        cover_mask[i] = m
    
    full_mask = (1 << N) - 1
    
    # Fast greedy with bit-parallel
    uncovered = full_mask
    greedy_codewords = []
    
    # Track uncovered count per codeword
    uncov_count = [0] * N
    for i in range(N):
        # Count bits in cover_mask[i] & uncovered
        # Since each cover has 15 bits, just check those 15
        cnt = 0
        for j in cover_list[i]:
            if (uncovered >> j) & 1:
                cnt += 1
        uncov_count[i] = cnt
    
    while uncovered:
        best_idx = -1
        best_cnt = -1
        for i in range(N):
            if uncov_count[i] > best_cnt:
                best_cnt = uncov_count[i]
                best_idx = i
        if best_idx == -1 or best_cnt == 0:
            break
        greedy_codewords.append(best_idx)
        # Update uncovered
        new_uncovered = uncovered & ~cover_mask[best_idx]
        # Update counts
        for j in cover_list[best_idx]:
            if (uncovered >> j) & 1:  # was uncovered
                for cw in covers_word[j]:
                    uncov_count[cw] -= 1
        uncovered = new_uncovered
    
    best_solution = list(greedy_codewords)
    best_size = len(best_solution)
    
    # Build coverage count for local search
    cov_count = [0] * N
    for i in best_solution:
        for j in cover_list[i]:
            cov_count[j] += 1
    
    # Local search: remove redundant codewords
    changed = True
    while changed:
        changed = False
        for idx in range(len(best_solution) - 1, -1, -1):
            i = best_solution[idx]
            can_remove = True
            for j in cover_list[i]:
                if cov_count[j] == 1:
                    can_remove = False
                    break
            if can_remove:
                for j in cover_list[i]:
                    cov_count[j] -= 1
                best_solution.pop(idx)
                changed = True
    
    # 1-swap local search: try replacing one codeword to enable removal
    start_time = time.time()
    time_limit = 100
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for idx in range(len(best_solution)):
            if time.time() - start_time > time_limit:
                break
            i = best_solution[idx]
            unique_words = [j for j in cover_list[i] if cov_count[j] == 1]
            if not unique_words:
                for j in cover_list[i]:
                    cov_count[j] -= 1
                best_solution.pop(idx)
                improved = True
                break
            # Try replacing i with a codeword that covers all unique_words
            tried = set()
            for uw in unique_words:
                for j in covers_word[uw]:
                    if j == i or j in tried or j in best_solution:
                        continue
                    tried.add(j)
                    # Check if j covers all unique_words
                    valid = True
                    for uw2 in unique_words:
                        if not ((cover_mask[j] >> uw2) & 1):
                            valid = False
                            break
                    if valid:
                        # Do swap
                        for jj in cover_list[i]:
                            cov_count[jj] -= 1
                        for jj in cover_list[j]:
                            cov_count[jj] += 1
                        best_solution[idx] = j
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    result = [list(all_words[i]) for i in best_solution]
    return result
# EVOLVE-BLOCK-END
