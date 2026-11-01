import time
import random
import math

def standard_superpermutation(n):
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
    return word

def get_perm_order(n):
    if n == 1:
        return [(1,)]
    
    word = standard_superpermutation(n)
    perms, seen = [], set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if window not in seen:
            seen.add(window)
            perms.append(window)
    return perms

def merge_blocks(blocks):
    if not blocks:
        return []
    result = list(blocks[0])
    for i in range(1, len(blocks)):
        b = blocks[i]
        max_overlap = min(len(result), len(b))
        for o in range(max_overlap, -1, -1):
            if result[-o:] == b[:o]:
                result += b[o:]
                break
    return result

def calc_length(blocks):
    if not blocks:
        return 0
    total = len(blocks[0])
    for i in range(1, len(blocks)):
        b_prev = blocks[i-1]
        b_curr = blocks[i]
        max_overlap = min(len(b_prev), len(b_curr))
        for o in range(max_overlap, -1, -1):
            if b_prev[-o:] == b_curr[:o]:
                total += len(b_curr) - o
                break
    return total

def optimize_order(blocks, time_limit=100):
    current_order = list(range(len(blocks)))
    current_len = calc_length([blocks[i] for i in current_order])
    
    best_order = current_order[:]
    best_len = current_len
    
    start_time = time.time()
    temperature = 100.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if random.random() < 0.5:
            i, j = random.sample(range(len(blocks)), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            i, j = random.sample(range(len(blocks)), 2)
            if i > j: i, j = j, i
            new_order = current_order[:]
            new_order[i:j+1] = reversed(new_order[i:j+1])
        
        new_len = calc_length([blocks[i] for i in new_order])
        
        delta = new_len - current_len
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_order = new_order
            current_len = new_len
            if current_len < best_len:
                best_len = current_len
                best_order = current_order[:]
        
        temperature *= cooling
        if temperature < 1.0:
            break
            
    return best_order, best_len

def construct(n):
    if n < 7:
        word = standard_superpermutation(n)
        return "".join(map(str, word))
    
    if n == 7:
        perms = get_perm_order(6)
        
        blocks = []
        for p in perms:
            block = list(p) + [7] + list(p)
            blocks.append(block)
        
        best_order, best_len = optimize_order(blocks, time_limit=100)
        
        ordered_blocks = [blocks[i] for i in best_order]
        word = merge_blocks(ordered_blocks)
        
        return "".join(map(str, word))
        
    word = standard_superpermutation(n)
    return "".join(map(str, word))
