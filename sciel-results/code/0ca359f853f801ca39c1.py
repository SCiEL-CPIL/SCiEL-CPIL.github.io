import time
import random
from itertools import permutations

def construct(n):
    if n <= 5:
        return _classic_construct(n)
    
    if n == 6:
        return _construct_optimized(6)
        
    if n == 7:
        # Construct S6 optimally first
        s6 = _construct_optimized(6)
        return _construct_optimized_from_base(s6, 7)

    # Fallback for other n
    return _classic_construct(n)

def _classic_construct(n):
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

def _construct_optimized(n):
    if n == 6:
        base_word = _classic_construct(5)
    else:
        base_word = _classic_construct(n-1)
        
    return _construct_optimized_from_base(base_word, n)

def _construct_optimized_from_base(base_word, n):
    m = n - 1
    perms, seen = [], set()
    for i in range(len(base_word) - m + 1):
        window = base_word[i:i+m]
        if window not in seen:
            seen.add(window)
            if set(window) == set(range(1, n)):
                perms.append(window)
                
    blocks = [p + str(n) + p for p in perms]
    num_blocks = len(blocks)
    
    # Precompute overlap matrix
    overlap = [[0]*num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j: continue
            b1 = blocks[i]
            b2 = blocks[j]
            max_k = min(len(b1), len(b2))
            for k in range(max_k, 0, -1):
                if b1[-k:] == b2[:k]:
                    overlap[i][j] = k
                    break
                    
    order = list(range(num_blocks))
    
    def calc_cost(ord):
        total = len(blocks[ord[0]])
        for i in range(len(ord) - 1):
            total += len(blocks[ord[i+1]]) - overlap[ord[i]][ord[i+1]]
        return total
        
    current_cost = calc_cost(order)
    best_cost = current_cost
    best_order = order[:]
    
    start_time = time.time()
    timeout = 5 if n == 6 else 110
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for i in range(num_blocks - 1):
            if time.time() - start_time > timeout:
                break
            a = order[i]
            b = order[i+1]
            prev = order[i-1] if i > 0 else None
            next_ = order[i+2] if i < num_blocks - 2 else None
            
            old_cost = 0
            if prev is not None:
                old_cost += len(blocks[a]) - overlap[prev][a]
            old_cost += len(blocks[b]) - overlap[a][b]
            if next_ is not None:
                old_cost += len(blocks[next_]) - overlap[b][next_]
                
            new_cost = 0
            if prev is not None:
                new_cost += len(blocks[b]) - overlap[prev][b]
            new_cost += len(blocks[a]) - overlap[b][a]
            if next_ is not None:
                new_cost += len(blocks[next_]) - overlap[a][next_]
            
            if new_cost < old_cost:
                order[i], order[i+1] = order[i+1], order[i]
                current_cost = calc_cost(order)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = order[:]
                improved = True
                
    new_word = list(blocks[best_order[0]])
    for k in range(1, len(best_order)):
        prev_idx = best_order[k-1]
        curr_idx = best_order[k]
        b_curr = blocks[curr_idx]
        o = overlap[prev_idx][curr_idx]
        new_word += list(b_curr[o:])
        
    word_str = "".join(map(str, new_word))
    
    if _is_valid_superpermutation(word_str, n):
        return word_str
    else:
        return _classic_construct(n)

def _is_valid_superpermutation(word, n):
    for p in permutations(range(1, n+1)):
        p_str = "".join(map(str, p))
        if p_str not in word:
            return False
    return True
