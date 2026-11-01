import time
import random
import numpy as np
from itertools import permutations

_SMALL_N_CACHE = {}

def _standard_superperm(n):
    if n in _SMALL_N_CACHE:
        return _SMALL_N_CACHE[n]
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
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    res = "".join(map(str, word))
    _SMALL_N_CACHE[n] = res
    return res

def construct(n):
    if n <= 4:
        return _standard_superperm(n)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    perms = list(permutations(range(1, n + 1)))
    num_perms = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Get initial order from standard superperm
    word_std = _standard_superperm(n)
    w_list = [int(c) for c in word_std]
    order = []
    seen = set()
    for i in range(len(w_list) - n + 1):
        p = tuple(w_list[i:i+n])
        if p not in seen:
            seen.add(p)
            order.append(perm_to_idx[p])
            
    if len(order) != num_perms:
        return word_std # Fallback
        
    overlap_cache = {}
    
    def get_cost(i, j):
        key = (i, j)
        if key in overlap_cache:
            return overlap_cache[key]
        u = perm_strs[i]
        v = perm_strs[j]
        # Max k such that u[-k:] == v[:k]
        # Since they are distinct, k < n
        for k in range(n - 1, -1, -1):
            if k == 0:
                cost = n
                break
            if u[-k:] == v[:k]:
                cost = n - k
                break
        overlap_cache[key] = cost
        return cost

    current_order = order
    current_cost = n
    for i in range(num_perms - 1):
        current_cost += get_cost(current_order[i], current_order[i+1])
        
    best_order = current_order[:]
    best_cost = current_cost
    
    # Local Search
    # Use a random seed for reproducibility? No, random is fine.
    
    while time.time() < deadline:
        # Swap adjacent
        i = random.randint(0, num_perms - 2)
        i1 = i
        i2 = i + 1
        
        old_c = 0
        new_c = 0
        
        if i1 > 0:
            old_c += get_cost(current_order[i1-1], current_order[i1])
            new_c += get_cost(current_order[i1-1], current_order[i2])
        
        old_c += get_cost(current_order[i1], current_order[i2])
        new_c += get_cost(current_order[i2], current_order[i1])
        
        if i2 < num_perms - 1:
            old_c += get_cost(current_order[i2], current_order[i2+1])
            new_c += get_cost(current_order[i1], current_order[i2+1])
            
        delta = new_c - old_c
        
        if delta < 0:
            current_order[i1], current_order[i2] = current_order[i2], current_order[i1]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_order = current_order[:]
                
    # Build word
    w = ""
    for idx in best_order:
        p = perm_strs[idx]
        if not w:
            w = p
        else:
            # Max overlap
            for k in range(n - 1, -1, -1):
                if k == 0:
                    w += p
                    break
                if w[-k:] == p[:k]:
                    w += p[k:]
                    break
            
    return w
