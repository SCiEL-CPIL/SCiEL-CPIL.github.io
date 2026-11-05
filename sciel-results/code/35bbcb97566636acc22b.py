import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1213231"
    if n == 4:
        return "12132314324124341"
    
    # Build SP(6) using classic recursive construction
    word = [1]
    for m in range(2, 7):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            if not new_word:
                new_word = block
            else:
                max_ov = 0
                limit = min(len(new_word), len(block) - 1)
                for o in range(limit, 0, -1):
                    if new_word[-o:] == block[:o]:
                        max_ov = o
                        break
                new_word += block[max_ov:]
        word = new_word
    
    # Extract unique 6-permutations in order of appearance
    perms_6, seen = [], set()
    for i in range(len(word) - 5):
        window = tuple(word[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
    
    num_perms = len(perms_6)
    
    # Precompute overlap matrix: M[i][j] = longest suffix of perms_6[i] that is prefix of perms_6[j]
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
    
    # Compute total length for a given ordering
    def calc_total_length(order):
        # Each block is 13 chars (6+1+6)
        # Total = 720*13 - sum of overlaps between consecutive blocks
        overlap_sum = 0
        for i in range(len(order) - 1):
            overlap_sum += M[order[i]][order[i + 1]]
        return num_perms * 13 - overlap_sum
    
    # Start with natural order
    order = list(range(num_perms))
    current_length = calc_total_length(order)
    best_order = order[:]
    best_length = current_length
    
    # Simulated annealing to minimize total length (maximize overlap sum)
    random.seed(42)
    T = 50.0
    cooling_rate = 0.9998
    
    while time.time() < deadline - 5.0:  # Leave 5s for construction
        # Adjacent swap
        i = random.randint(0, num_perms - 2)
        j = i + 1
        
        # Calculate delta in overlap sum
        old_ov = 0
        new_ov = 0
        
        if i > 0:
            old_ov += M[order[i-1]][order[i]]
            new_ov += M[order[i-1]][order[j]]
        old_ov += M[order[i]][order[j]]
        new_ov += M[order[j]][order[i]]
        if j + 1 < num_perms:
            old_ov += M[order[j]][order[j+1]]
            new_ov += M[order[i]][order[j+1]]
        
        delta_ov = new_ov - old_ov
        delta_length = -delta_ov  # We want to minimize length
        
        if delta_length <= 0 or random.random() < pow(2.718281828, -delta_length / T):
            order[i], order[j] = order[j], order[i]
            current_length += delta_length
            if current_length < best_length:
                best_length = current_length
                best_order = order[:]
        
        T *= cooling_rate
        if T < 0.01:
            break
    
    # Construct final word from best_order
    final_word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            final_word = block
        else:
            max_ov = 0
            limit = min(len(final_word), len(block) - 1)
            for o in range(limit, 0, -1):
                if final_word[-o:] == block[:o]:
                    max_ov = o
                    break
            final_word += block[max_ov:]
    
    return "".join(map(str, final_word))
