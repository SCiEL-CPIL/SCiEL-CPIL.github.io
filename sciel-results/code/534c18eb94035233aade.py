import time
import random
import sys
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Classic recursive construction
    def classic_construct(n):
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
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    if n != 7:
        word = classic_construct(n)
        return "".join(map(str, word))
    
    # Special optimization for n=7
    # Step 1: Build SP(6) and extract permutations of 1..6 in order
    word_6 = [1]
    for m in range(2, 7):
        perms, seen = [], set()
        for i in range(len(word_6) - m + 2):
            window = tuple(word_6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word_6 = new
    
    # Extract unique 6-permutations in order of appearance
    perms_6, seen = [], set()
    for i in range(len(word_6) - 5):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
    
    num_perms = len(perms_6)
    
    # Step 2: Precompute overlap matrix using numpy for speed
    # M[i][j] = longest suffix of perms_6[i] that is a prefix of perms_6[j]
    # Convert perms to numpy array of shape (num_perms, 6)
    perms_arr = np.array(perms_6, dtype=np.int8)  # shape (720, 6)
    
    # For each k from 6 down to 0, check if suffix of length k of p1 equals prefix of length k of p2
    # M[i][j] = max k such that perms_arr[i, 6-k:] == perms_arr[j, :k]
    M = np.zeros((num_perms, num_perms), dtype=np.int8)
    
    for k in range(6, 0, -1):
        # For each pair (i,j), check if perms_arr[i, 6-k:] == perms_arr[j, :k]
        # This is a matrix comparison
        suffixes = perms_arr[:, 6-k:]  # shape (num_perms, k)
        prefixes = perms_arr[:, :k]    # shape (num_perms, k)
        # Check where suffixes[i] == prefixes[j] for all j
        # Use broadcasting: suffixes[:, np.newaxis, :] == prefixes[np.newaxis, :, :]
        # This creates a (num_perms, num_perms, k) boolean array
        match = np.all(suffixes[:, np.newaxis, :] == prefixes[np.newaxis, :, :], axis=2)
        # Only set M[i][j] = k where not already set (we go from high k to low k)
        mask = match & (M == 0)
        M[mask] = k
    
    # Step 3: Local search to optimize ordering
    # Total length = 720 * 13 - sum of overlaps
    # So we want to maximize sum of overlaps
    
    def calc_overlap_sum(order):
        total = 0
        for i in range(len(order) - 1):
            total += M[order[i]][order[i + 1]]
        return total
    
    random.seed(42)
    
    # Better initial order: nearest neighbor greedy
    order = [0]
    remaining = set(range(1, num_perms))
    for _ in range(num_perms - 1):
        last = order[-1]
        best_next = -1
        best_ov = -1
        for j in remaining:
            ov = M[last][j]
            if ov > best_ov:
                best_ov = ov
                best_next = j
        order.append(best_next)
        remaining.remove(best_next)
    
    current_score = calc_overlap_sum(order)
    best_order = order[:]
    best_score = current_score
    
    # Simulated annealing with 2-opt moves
    T = 100.0  # initial temperature
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        # Pick a random move type: adjacent swap (50%) or 2-opt reversal (50%)
        if random.random() < 0.5:
            # Adjacent swap
            i = random.randint(0, num_perms - 2)
            j = i + 1
            
            new_order = order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            # Calculate delta in score
            old_score = 0
            new_score = 0
            
            # Affected edges: (i-1,i), (i,i+1), (i+1,i+2)
            if i > 0:
                old_score += M[order[i-1]][order[i]]
                new_score += M[order[i-1]][order[i+1]]
            old_score += M[order[i]][order[i+1]]
            new_score += M[order[i+1]][order[i]]
            if i + 2 < num_perms:
                old_score += M[order[i+1]][order[i+2]]
                new_score += M[order[i]][order[i+2]]
            
            delta = new_score - old_score
        else:
            # 2-opt: reverse a segment [i, j]
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            
            new_order = order[:]
            new_order[i:j+1] = reversed(new_order[i:j+1])
            
            # Calculate delta in score
            # Only edges affected: (i-1, i), (i, i+1) -> (i-1, j), (j, j-1)
            # and (j, j+1) -> (i+1, i), (i+1, i+2) if they exist
            # More precisely:
            # Old edges involving the reversed segment:
            #   If i > 0: (i-1, i)
            #   (i, i+1), (i+1, i+2), ..., (j-1, j)  [internal edges, reversed]
            #   If j < num_perms-1: (j, j+1)
            # New edges:
            #   If i > 0: (i-1, j)
            #   (j, j-1), (j-1, j-2), ..., (i+1, i)  [internal edges, reversed]
            #   If j < num_perms-1: (i, j+1)
            
            old_score = 0
            new_score = 0
            
            if i > 0:
                old_score += M[order[i-1]][order[i]]
                new_score += M[order[i-1]][order[j]]
            
            # Internal edges: old has (i,i+1), (i+1,i+2), ..., (j-1,j)
            # new has (j,j-1), (j-1,j-2), ..., (i+1,i)
            for t in range(i, j):
                old_score += M[order[t]][order[t+1]]
                new_score += M[order[j-(t-i)]][order[j-(t-i)-1]]
            
            if j < num_perms - 1:
                old_score += M[order[j]][order[j+1]]
                new_score += M[order[i]][order[j+1]]
            
            delta = new_score - old_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            order = new_order
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_order = order[:]
        
        T *= cooling_rate
        if T < 0.01:
            break
    
    # Step 4: Construct the final word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            ov = 0
            max_ov = min(len(word), len(block))
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]
    
    return "".join(map(str, word))
