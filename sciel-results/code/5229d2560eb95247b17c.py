"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Build the superpermutation up to layer n-1
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_ov = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # If n is small, just return what we have
    if n <= 2:
        return "".join(map(str, word))
    
    # Extract all unique (n-1)-permutations from the current word in order of appearance
    perms, seen = [], set()
    for i in range(len(word) - n + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    num_perms = len(perms)
    
    # Precompute overlap matrix
    # M[i][j] = longest suffix of perms[i] that is a prefix of perms[j]
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms[i]
        for j in range(num_perms):
            p2 = perms[j]
            # Max possible overlap is n-1
            for k in range(n - 1, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
    
    # Calculate total overlap for a given ordering
    def calc_overlap_sum(order):
        total = 0
        for i in range(len(order) - 1):
            total += M[order[i]][order[i + 1]]
        return total
    
    # Start with the natural order
    order = list(range(num_perms))
    current_score = calc_overlap_sum(order)
    best_order = order[:]
    best_score = current_score
    
    # Simulated annealing to optimize ordering
    random.seed(42)
    T = 50.0  # initial temperature
    cooling_rate = 0.9999
    
    while time.time() < deadline:
        # Pick a random adjacent swap
        i = random.randint(0, num_perms - 2)
        j = i + 1
        
        new_order = order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        # Calculate delta in score
        old_score = 0
        new_score = 0
        
        if i > 0:
            old_score += M[order[i-1]][order[i]]
            new_score += M[order[i-1]][order[j]]
        old_score += M[order[i]][order[j]]
        new_score += M[order[j]][order[i]]
        if j + 1 < num_perms:
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
    
    # Construct the final word from best_order
    final_word = []
    for idx, p_idx in enumerate(best_order):
        p = perms[p_idx]
        block = list(p) + [n] + list(p)
        if idx == 0:
            final_word = block
        else:
            max_ov = min(len(final_word), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if final_word[-k:] == block[:k]:
                    ov = k
                    break
            final_word += block[ov:]
    
    return "".join(map(str, final_word))
# EVOLVE-BLOCK-END
