import time
import random
import numpy as np

def _count_k222_bit(adj_list, n):
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj_list[a] >> b) & 1:
                continue
            common_ab = adj_list[a] & adj_list[b]
            common_ab &= ~((1 << (b + 1)) - 1)
            if common_ab == 0:
                continue
            common_list = []
            mask = common_ab
            while mask:
                lsb = mask & (-mask)
                c = lsb.bit_length() - 1
                common_list.append(c)
                mask ^= lsb
            for ci in range(len(common_list)):
                c = common_list[ci]
                for di in range(ci + 1, len(common_list)):
                    d = common_list[di]
                    if not (adj_list[c] >> d) & 1:
                        continue
                    common_abcd = adj_list[a] & adj_list[b] & adj_list[c] & adj_list[d]
                    common_abcd &= ~((1 << (d + 1)) - 1)
                    cnt = common_abcd.bit_count()
                    if cnt >= 2:
                        count += cnt * (cnt - 1) // 2
    return count // 3

def _total_violations_bit(adj_list, n):
    v1 = _count_k222_bit(adj_list, n)
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj_list[i] ^ (1 << i)
    v0 = _count_k222_bit(comp, n)
    return v0 + v1

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    q = 29
    n_target = 30
    
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    adj_list = [0] * n_target
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj_list[i] |= (1 << j)
                adj_list[j] |= (1 << i)
    
    random.seed(42)
    for i in range(q):
        if random.random() < 0.5:
            adj_list[29] |= (1 << i)
            adj_list[i] |= (1 << 29)
    
    current_violations = _total_violations_bit(adj_list, n_target)
    best_adj_list = adj_list[:]
    best_violations = current_violations
    
    if current_violations > 0 and time.time() < deadline:
        temperature = 1.0
        cooling_rate = 0.99
        
        while time.time() < deadline - 5 and temperature > 0.01:
            i = random.randint(0, n_target - 1)
            j = random.randint(0, n_target - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            if (adj_list[i] >> j) & 1:
                adj_list[i] &= ~(1 << j)
                adj_list[j] &= ~(1 << i)
            else:
                adj_list[i] |= (1 << j)
                adj_list[j] |= (1 << i)
            
            new_violations = _total_violations_bit(adj_list, n_target)
            
            delta = new_violations - current_violations
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_adj_list = adj_list[:]
                    if current_violations == 0:
                        break
            else:
                if (adj_list[i] >> j) & 1:
                    adj_list[i] &= ~(1 << j)
                    adj_list[j] &= ~(1 << i)
                else:
                    adj_list[i] |= (1 << j)
                    adj_list[j] |= (1 << i)
            
            temperature *= cooling_rate
        
        adj_list = best_adj_list
    
    # If we couldn't find a valid 30-vertex coloring, fall back to n=29
    if _total_violations_bit(adj_list, n_target) > 0:
        n = q
        adj_list = adj_list[:q]
    else:
        n = n_target
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(1 if (adj_list[i] >> j) & 1 else 0)
    
    return {'n': n, 'colors': colors}
