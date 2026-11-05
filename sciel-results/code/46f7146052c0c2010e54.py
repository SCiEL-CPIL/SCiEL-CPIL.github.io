import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Classic construction for n != 7
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
                # Calculate overlap with new
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n != 7:
        return classic_construct(n)
    
    # Optimization for n=7
    # 1. Generate permutations of 1..6 in the order of SP(6)
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
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word_6 = new
        
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 2):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    num_perms = len(perms_6) # 720
    
    # 2. Precompute overlap matrix
    # M[i][j] = longest suffix of perms_6[i] matching prefix of perms_6[j]
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            # Max overlap 6
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
                    
    # 3. Simulated Annealing
    # Initial order: Greedy
    order = [0]
    unvisited = set(range(1, num_perms))
    while unvisited:
        last = order[-1]
        best_next = -1
        best_ov = -1
        # To speed up, we can just pick a random unvisited with high overlap
        # But greedy is fine for 720 nodes
        for j in unvisited:
            ov = M[last][j]
            if ov > best_ov:
                best_ov = ov
                best_next = j
        order.append(best_next)
        unvisited.remove(best_next)
        
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score

    current_order = order[:]
    current_score = calc_score(current_order)
    best_order = current_order[:]
    best_score = current_score
    
    random.seed(42)
    T = 1.0
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        new_order = current_order[:]
        
        if move_type < 0.4:
            # Swap
            i = random.randint(0, num_perms - 2)
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
        elif move_type < 0.7:
            # 2-Opt (reverse segment)
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            new_order[i:j+1] = new_order[i:j+1][::-1]
        else:
            # Move
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j: continue
            item = new_order.pop(i)
            new_order.insert(j, item)
            
        new_score = calc_score(new_order)
        delta = new_score - current_score
        
        if delta > 0:
            current_order = new_order
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_order = new_order[:]
        else:
            # Accept with probability exp(delta / T)
            if random.random() < pow(2.71828, delta / T):
                current_order = new_order
                current_score = new_score
        
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor temperature to allow some exploration
            
    # 4. Construct word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            max_ov = min(len(word), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]
            
    return "".join(map(str, word))
