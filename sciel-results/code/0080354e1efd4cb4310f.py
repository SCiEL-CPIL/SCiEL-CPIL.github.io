import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    val_to_idx = {v: i for i, v in enumerate(candidates)}

    adj = [0] * num_candidates
    
    for i in range(num_candidates):
        u = candidates[i]
        for j in range(i + 1, num_candidates):
            v = candidates[j]
            if u * v <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_set_bits = 0
    best_set_len = 0
    
    def mis_search(U, P):
        nonlocal best_set_bits, best_set_len
        
        if U == 0:
            if bin(P).count('1') > best_set_len:
                best_set_len = bin(P).count('1')
                best_set_bits = P
            return
            
        if bin(P).count('1') + bin(U).count('1') <= best_set_len:
            return
            
        if time.time() > deadline:
            return
            
        # Pick vertex with highest degree in U
        best_v = -1
        best_deg = -1
        temp = U
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = bin(adj[idx] & U).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = idx
            temp ^= lsb
            
        if best_v == -1:
            return
            
        v_idx = best_v
        v_bit = 1 << v_idx
        
        new_U1 = U & ~(adj[v_idx] | v_bit)
        mis_search(new_U1, P | v_bit)
        
        new_U2 = U & ~v_bit
        mis_search(new_U2, P)

    initial_U = (1 << num_candidates) - 1
    mis_search(initial_U, 0)
    
    if best_set_bits == 0:
        return [2]
        
    best_seq = []
    for i in range(num_candidates):
        if (best_set_bits >> i) & 1:
            best_seq.append(candidates[i])
            
    def is_valid_sequence(seq):
        if not seq:
            return True
        if len(seq) == 1:
            return True
        if seq != sorted(seq):
            return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        if len(set(seq)) != len(seq):
            return False
            
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in prods:
                    return False
                prods.add(prod)
        return True

    if is_valid_sequence(best_seq):
        # Try to improve by adding elements
        for x in range(2, 129):
            if x in best_seq:
                continue
            new_seq = best_seq + [x]
            new_seq.sort()
            if is_valid_sequence(new_seq):
                best_seq = new_seq
        return best_seq
    
    # Repair
    seq = best_seq[:]
    valid = False
    attempts = 0
    while not valid and attempts < 100 and time.time() < deadline - 10:
        valid = is_valid_sequence(seq)
        if valid:
            break
            
        n = len(seq)
        prods = {}
        collision_found = False
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in prods:
                    seq.pop(j)
                    collision_found = True
                    break
                else:
                    prods[prod] = (i, j)
            if collision_found:
                break
        
        if not collision_found:
            break
            
        attempts += 1
        
    if is_valid_sequence(seq):
        # Try to improve by adding elements
        for x in range(2, 129):
            if x in seq:
                continue
            new_seq = seq + [x]
            new_seq.sort()
            if is_valid_sequence(new_seq):
                seq = new_seq
        return seq
        
    # Fallback: Pure Greedy
    seq = []
    prods = set()
    suffix_prods = []
    for x in range(2, 129):
        new_sp = [x]
        valid_add = True
        if x in prods:
            valid_add = False
        else:
            for p in suffix_prods:
                np = x * p
                if np in prods:
                    valid_add = False
                    break
                new_sp.append(np)
        
        if valid_add:
            seq.append(x)
            prods.update(new_sp)
            suffix_prods = new_sp
            
    return seq
