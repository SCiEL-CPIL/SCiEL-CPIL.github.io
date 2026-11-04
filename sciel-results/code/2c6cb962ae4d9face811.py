import time
import random
import sys

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
    
    # Step 2: Precompute overlap matrix
    # M[i][j] = longest suffix of perms_6[i] that is a prefix of perms_6[j]
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            # Max possible overlap is 6
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
    
    # Step 3: Local search to optimize ordering
    # Total length = sum of block lengths - sum of overlaps
    # Each block has length 13 (6 + 1 + 6)
    # Total length = 720 * 13 - sum of overlaps
    # So we want to maximize sum of overlaps
    
    def calc_overlap_sum(order):
        total = 0
        for i in range(len(order) - 1):
            total += M[order[i]][order[i + 1]]
        return total
    
    random.seed(42)
    
    # Start with the natural order
    order = list(range(num_perms))
    current_score = calc_overlap_sum(order)
    best_order = order[:]
    best_score = current_score
    
    # Simulated annealing
    T = 100.0  # initial temperature
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        # Pick a random move: swap two elements
        i = random.randint(0, num_perms - 2)
        j = i + 1  # adjacent swap for efficiency
        
        new_order = order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        # Calculate delta in score
        # Old contributions: M[order[i-1]][order[i]] + M[order[i]][order[i+1]] + M[order[i+1]][order[i+2]]
        # New contributions: M[order[i-1]][order[i+1]] + M[order[i+1]][order[i]] + M[order[i]][order[i+2]]
        
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
