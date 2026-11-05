import time
import random
import math
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Classic construction
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n != 7:
        return classic_construct(n)
    
    # Optimization for n=7
    # 1. Get the list of permutations of 1..6 in the order they appear in SP(6)
    word_6 = [1]
    for m in range(2, 7): # up to 6
        perms, seen = [], set()
        for i in range(len(word_6) - m + 2):
            window = tuple(word_6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_6 = new
        
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 2):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    # perms_6 is a list of 720 tuples of length 6
    num_perms = len(perms_6)
    
    # 2. Precompute overlap matrix
    # M[i][j] = overlap between perms_6[i] and perms_6[j]
    # Overlap is longest suffix of i matching prefix of j
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
                    
    # 3. Local Search with Delta Calculations and Restarts
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score
    
    random.seed(42)
    
    best_order = list(range(num_perms))
    best_score = calc_score(best_order)
    
    # Multiple restarts with simulated annealing
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
            
        if restart == 0:
            order = list(range(num_perms))
        else:
            # Perturb the best order
            order = best_order[:]
            # Apply random swaps
            num_swaps = num_perms // 10
            for _ in range(num_swaps):
                i = random.randint(0, num_perms - 2)
                j = random.randint(0, num_perms - 2)
                if i < j:
                    i, j = j, i
                order[i], order[i+1] = order[i+1], order[i]
        
        current_score = calc_score(order)
        
        # Simulated Annealing parameters
        temp = 1.0
        min_temp = 0.01
        cool_rate = 0.9999
        
        while time.time() < deadline - 2.0 and temp > min_temp:
            # Pick a random move
            move_type = random.random()
            if move_type < 0.5:
                # Swap adjacent
                i = random.randint(0, num_perms - 2)
                a = order[i]
                b = order[i+1]
                
                # Calculate Delta
                delta = 0
                prev_a = order[i-1] if i > 0 else None
                next_b = order[i+2] if i+2 < num_perms else None
                
                if prev_a is not None:
                    delta -= M[prev_a][a]
                    delta += M[prev_a][b]
                # Middle edge
                delta -= M[a][b]
                delta += M[b][a]
                if next_b is not None:
                    delta -= M[b][next_b]
                    delta += M[a][next_b]
                    
                new_score = current_score + delta
                if new_score > best_score:
                    # Apply swap
                    order[i], order[i+1] = b, a
                    current_score = new_score
                    best_score = current_score
                    best_order = order[:]
                else:
                    # Accept with probability exp(-delta/temp) if delta < 0
                    if delta < 0 and random.random() < math.exp(-delta / temp):
                        order[i], order[i+1] = b, a
                        current_score = new_score
            else:
                # Move element i to position j
                i = random.randint(0, num_perms - 1)
                j = random.randint(0, num_perms - 1)
                if i == j: continue
                
                if i < j:
                    # Remove i, insert at j
                    x = order[i]
                    y = order[j]
                    
                    prev_x = order[i-1] if i > 0 else None
                    next_y = order[j+1] if j+1 < num_perms else None
                    
                    delta = 0
                    
                    if j == i + 1:
                        if prev_x is not None:
                            delta -= M[prev_x][x]
                            delta += M[prev_x][y]
                        delta -= M[x][y]
                        delta += M[y][x]
                        if next_y is not None:
                            delta -= M[y][next_y]
                            delta += M[x][next_y]
                    else:
                        b_first = order[i+1]
                        b_last = order[j-1]
                        
                        if prev_x is not None:
                            delta -= M[prev_x][x]
                            delta += M[prev_x][b_first]
                        
                        delta -= M[x][b_first]
                        
                        delta -= M[b_last][y]
                        delta += M[b_last][x]
                        
                        delta += M[x][y]
                        
                        item = order.pop(i)
                        order.insert(j, item)
                        current_score += delta
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_order = order[:]
                        elif delta < 0 and random.random() < math.exp(-delta / temp):
                            pass
                        else:
                            order.pop(j)
                            order.insert(i, item)
                            current_score -= delta
                            
                else:
                    # i > j
                    x = order[i]
                    y = order[j]
                    
                    prev_y = order[j-1] if j > 0 else None
                    next_x = order[i+1] if i+1 < num_perms else None
                    
                    delta = 0
                    
                    if j == i - 1:
                        if prev_y is not None:
                            delta -= M[prev_y][y]
                            delta += M[prev_y][x]
                        delta -= M[y][x]
                        delta += M[x][y]
                        if next_x is not None:
                            delta -= M[x][next_x]
                            delta += M[y][next_x]
                    else:
                        b_first = order[j+1]
                        b_last = order[i-1]
                        
                        if prev_y is not None:
                            delta -= M[prev_y][y]
                            delta += M[prev_y][b_first]
                        
                        delta -= M[y][b_first]
                        
                        delta -= M[b_last][x]
                        delta += M[b_last][y]
                        
                        delta += M[y][x]
                        
                        item = order.pop(i)
                        order.insert(j, item)
                        current_score += delta
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_order = order[:]
                        elif delta < 0 and random.random() < math.exp(-delta / temp):
                            pass
                        else:
                            order.pop(j)
                            order.insert(i, item)
                            current_score -= delta

            temp *= cool_rate

    # 4. Construct word from best_order using precomputed overlaps
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            prev_idx = best_order[idx-1]
            ov = M[prev_idx][p_idx]
            word += block[ov:]
            
    return "".join(map(str, word))
