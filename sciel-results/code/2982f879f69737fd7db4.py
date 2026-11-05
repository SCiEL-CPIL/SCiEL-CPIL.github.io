"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Precompute gcd table for speed
    gcd_table = [[0] * 512 for _ in range(512)]
    for i in range(1, 512):
        for j in range(1, 512):
            gcd_table[i][j] = gcd(i, j)
    
    # Greedy: pick largest a first, maintaining pairwise coprime
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd_table[a][b] == 1 for b in chosen):
            chosen.append(a)
    
    # Score function
    def score(lst):
        return sum(1.0 / (512 - a) for a in lst)
    
    current_score = score(chosen)
    chosen_set = set(chosen)
    chosen_list = list(chosen)
    
    # Local search with multiple passes and random restarts
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(chosen_list)):
            a = chosen_list[i]
            # Try to find a better replacement
            best_candidate = None
            best_gain = 0
            for b in range(511, 0, -1):
                if b in chosen_set:
                    continue
                # Check if b is coprime to all other chosen elements
                ok = True
                for j in range(len(chosen_list)):
                    if j == i:
                        continue
                    if gcd_table[b][chosen_list[j]] != 1:
                        ok = False
                        break
                if not ok:
                    continue
                gain = 1.0 / (512 - b) - 1.0 / (512 - a)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b
            if best_candidate is not None and best_gain > 0:
                # Make the swap
                chosen_set.remove(a)
                chosen_set.add(best_candidate)
                chosen_list[i] = best_candidate
                current_score += best_gain
                improved = True
                if (time.time() - start_time) > deadline:
                    break
    
    # Try adding new elements if possible
    while (time.time() - start_time) < deadline:
        added = False
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if all(gcd_table[a][b] == 1 for b in chosen_list):
                chosen_list.append(a)
                chosen_set.add(a)
                current_score += 1.0 / (512 - a)
                added = True
                break
        if not added:
            break
    
    # Second round of local search after additions
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(chosen_list)):
            a = chosen_list[i]
            best_candidate = None
            best_gain = 0
            for b in range(511, 0, -1):
                if b in chosen_set:
                    continue
                ok = True
                for j in range(len(chosen_list)):
                    if j == i:
                        continue
                    if gcd_table[b][chosen_list[j]] != 1:
                        ok = False
                        break
                if not ok:
                    continue
                gain = 1.0 / (512 - b) - 1.0 / (512 - a)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b
            if best_candidate is not None and best_gain > 0:
                chosen_set.remove(a)
                chosen_set.add(best_candidate)
                chosen_list[i] = best_candidate
                current_score += best_gain
                improved = True
                if (time.time() - start_time) > deadline:
                    break
    
    return chosen_list
# EVOLVE-BLOCK-END
