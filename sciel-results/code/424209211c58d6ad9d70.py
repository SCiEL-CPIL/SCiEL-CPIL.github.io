import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 128
    numbers = list(range(1, n + 1))
    idx = {x: i for i, x in enumerate(numbers)}
    N = n

    # 1. Identify all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))

    if time.time() > deadline:
        return numbers

    # 2. Precompute conflict structures using bitmasks
    # conflict_mask[x] = bitmask of all y such that (x,y,z) is a bad triple for some z
    # conflict_pairs_mask[x][y] = bitmask of z such that (x,y,z) is a bad triple
    # For can_add(x, S): check if any pair (u,v) in conflicts[x] both in S
    # Equivalent: for each u in S, check if there exists v in S such that (u,v) is in conflict_pairs[x]
    
    # Better approach: conflict_pairs[x] = set of frozensets {u,v}
    # can_add(x, S) = all((u not in S) or (v not in S) for u,v in conflict_pairs[x])
    
    # Use bitmask: for each x, store a list of (u_bit, v_bit) pairs
    # can_add: for each (u_bit, v_bit), check (S_mask & u_bit) == 0 or (S_mask & v_bit) == 0
    
    conflict_pairs_list = [[] for _ in range(N + 1)]
    for a, b, c in bad_triples:
        ia, ib, ic = idx[a], idx[b], idx[c]
        conflict_pairs_list[ia].append((ib, ic))
        conflict_pairs_list[ib].append((ia, ic))
        conflict_pairs_list[ic].append((ia, ib))
    
    # Convert to bitmask pairs for fast checking
    # For each x, store list of (u_bit, v_bit) where u_bit = 1 << u, v_bit = 1 << v
    conflict_bit_pairs = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for u, v in conflict_pairs_list[i - 1]:
            conflict_bit_pairs[i].append((1 << u, 1 << v))
    
    # can_add(x, S_mask): S_mask is an integer bitmask
    def can_add(x, S_mask):
        for ub, vb in conflict_bit_pairs[x]:
            if (S_mask & ub) and (S_mask & vb):
                return False
        return True

    # 3. Greedy initialization
    def greedy_init():
        S_mask = (1 << (N + 1)) - 2  # all bits 1..N set
        S_mask &= ~1  # clear bit 0
        
        # Count how many bad triples each element is in
        counts = [0] * (N + 1)
        for a, b, c in bad_triples:
            counts[a] += 1
            counts[b] += 1
            counts[c] += 1
        
        active = [True] * len(bad_triples)
        
        while True:
            max_count = 0
            candidates = []
            for x in range(1, N + 1):
                if S_mask & (1 << x):
                    if counts[x] > max_count:
                        max_count = counts[x]
                        candidates = [x]
                    elif counts[x] == max_count and max_count > 0:
                        candidates.append(x)
            
            if not candidates or max_count == 0:
                break
            
            target = random.choice(candidates)
            S_mask &= ~(1 << target)
            
            for i, (a, b, c) in enumerate(bad_triples):
                if not active[i]:
                    continue
                if target in (a, b, c):
                    active[i] = False
                    for x in (a, b, c):
                        if x != target:
                            counts[x] -= 1
        
        return S_mask

    # 4. Simulated Annealing with bitmask
    def simulate_annealing(S_mask, iterations=1500, initial_temp=8.0, cooling_rate=0.995):
        current_size = bin(S_mask).count('1')
        best_mask = S_mask
        best_size = current_size
        temp = initial_temp
        
        for it in range(iterations):
            if time.time() > deadline - 3.0:
                break
            
            move_type = random.random()
            
            if move_type < 0.35:
                # Try to add
                not_in_mask = ((1 << (N + 1)) - 2) & ~S_mask
                if not_in_mask:
                    # Pick random bit from not_in_mask
                    bits = []
                    m = not_in_mask
                    while m:
                        lb = m & (-m)
                        bits.append(lb)
                        m ^= lb
                    if bits:
                        bit = random.choice(bits)
                        x = bit.bit_length() - 1
                        if can_add(x, S_mask):
                            new_mask = S_mask | bit
                            new_size = current_size + 1
                            if new_size > best_size:
                                best_size = new_size
                                best_mask = new_mask
                            S_mask = new_mask
                            current_size = new_size
            elif move_type < 0.65:
                # Try to remove
                if S_mask:
                    bits = []
                    m = S_mask
                    while m:
                        lb = m & (-m)
                        bits.append(lb)
                        m ^= lb
                    if bits:
                        bit = random.choice(bits)
                        x = bit.bit_length() - 1
                        new_mask = S_mask & ~bit
                        new_size = current_size - 1
                        delta = new_size - current_size
                        if random.random() < pow(2.718281828, delta / temp):
                            S_mask = new_mask
                            current_size = new_size
            else:
                # Swap
                if S_mask:
                    not_in_mask = ((1 << (N + 1)) - 2) & ~S_mask
                    if not_in_mask:
                        # Pick random from S
                        s_bits = []
                        m = S_mask
                        while m:
                            lb = m & (-m)
                            s_bits.append(lb)
                            m ^= lb
                        if s_bits:
                            x_bit = random.choice(s_bits)
                            # Pick random from not_in
                            n_bits = []
                            m = not_in_mask
                            while m:
                                lb = m & (-m)
                                n_bits.append(lb)
                                m ^= lb
                            if n_bits:
                                y_bit = random.choice(n_bits)
                                y = y_bit.bit_length() - 1
                                new_mask = S_mask & ~x_bit
                                if can_add(y, new_mask):
                                    new_mask |= y_bit
                                    new_size = current_size  # same size
                                    # Accept if it might help (always accept same-size swaps with prob)
                                    if random.random() < 0.5:
                                        S_mask = new_mask
                                        current_size = new_size
                                        if current_size > best_size:
                                            best_size = current_size
                                            best_mask = S_mask
            
            temp *= cooling_rate
            if temp < 0.05:
                break
        
        return best_mask, best_size

    # 5. Local search: try removing one element and greedily adding back
    def local_search(S_mask):
        S_mask = S_mask
        improved = True
        max_iters = 20
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            if time.time() > deadline - 5.0:
                break
            
            s_list = []
            m = S_mask
            while m:
                lb = m & (-m)
                s_list.append(lb.bit_length() - 1)
                m ^= lb
            random.shuffle(s_list)
            
            best_gain = 0
            best_mask = S_mask
            
            for y in s_list:
                if time.time() > deadline - 5.0:
                    break
                tentative = S_mask & ~(1 << y)
                # Try to add all possible
                for x in range(1, N + 1):
                    if not (tentative & (1 << x)) and can_add(x, tentative):
                        tentative |= (1 << x)
                gain = bin(tentative).count('1') - bin(S_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_mask = tentative
            
            if best_gain > 0:
                S_mask = best_mask
                improved = True
        
        return S_mask

    # 6. Exhaustive add
    def exhaustive_add(S_mask):
        S_mask = S_mask
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            for x in range(1, N + 1):
                if not (S_mask & (1 << x)) and can_add(x, S_mask):
                    S_mask |= (1 << x)
                    improved = True
        return S_mask

    # Main search
    best_mask = 0
    best_size = -1
    
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        S_mask = greedy_init()
        S_mask, size = simulate_annealing(S_mask, iterations=1200, initial_temp=8.0, cooling_rate=0.995)
        S_mask = local_search(S_mask)
        S_mask = exhaustive_add(S_mask)
        
        if bin(S_mask).count('1') > best_size:
            best_size = bin(S_mask).count('1')
            best_mask = S_mask
    
    # Final polish
    if best_size > 0:
        best_mask = exhaustive_add(best_mask)
        best_size = bin(best_mask).count('1')
    
    if best_size <= 0:
        return [1]
    
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        result.append(lb.bit_length() - 1)
        m ^= lb
    result.sort()
    return result
