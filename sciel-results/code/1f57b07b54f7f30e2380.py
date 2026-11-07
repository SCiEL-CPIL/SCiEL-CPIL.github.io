import time
import math
import random

NMAX = 256
RANGE = list(range(2, NMAX + 1))
N = len(RANGE)
val_to_idx = {v: i for i, v in enumerate(RANGE)}
idx_to_val = RANGE

# Precompute conflict masks
# conflict_mask[a_idx][b_idx] = bitmask of c such that a | b*c
# This means if a, b, c are all in S and c is in conflict_mask[a][b], constraint violated
conflict_mask = [[0] * N for _ in range(N)]
for i in range(N):
    a = idx_to_val[i]
    for j in range(N):
        b = idx_to_val[j]
        mask = 0
        g = math.gcd(a, b)
        req = a // g
        for c_val in range(2, NMAX + 1):
            if c_val % req == 0:
                mask |= (1 << val_to_idx[c_val])
        conflict_mask[i][j] = mask


def is_valid(S_list):
    """Check if S_list is a valid set (no a|b*c for distinct a,b,c)."""
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            g = math.gcd(a, b)
            req = a // g
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S_list[k]
                if c % req == 0:
                    return False
    return True


def compute_pressure(S_mask, i):
    """
    For element i in S, compute how many pairs (b,c) in S\{i} would violate
    if i were removed... actually, compute degree: number of (j,k) distinct from i
    in S such that i | val[j]*val[k].
    """
    a_idx = i
    a_val = idx_to_val[i]
    deg = 0
    for j in range(N):
        if j == i or not ((S_mask >> j) & 1):
            continue
        b_val = idx_to_val[j]
        g = math.gcd(a_val, b_val)
        req = a_val // g
        # Count c in S, c != i, c != j, such that c % req == 0
        for k in range(N):
            if k == i or k == j or not ((S_mask >> k) & 1):
                continue
            c_val = idx_to_val[k]
            if c_val % req == 0:
                deg += 1
    return deg


def greedy_construct(deadline):
    """
    Greedy: start empty, add elements one by one.
    Choose element that maximizes "value" = how many new valid triples it allows minus
    how many conflicts it creates.
    """
    S = []
    S_set = set()
    remaining = list(RANGE)
    random.shuffle(remaining)
    
    while time.time() < deadline and remaining:
        best_elem = None
        best_score = -1
        
        for elem in remaining:
            if elem in S_set:
                continue
            # Check if elem can be added
            can_add = True
            n = len(S)
            for i in range(n):
                a = S[i]
                g = math.gcd(elem, a)
                req_a = a // g  # for a | elem * c
                for j in range(n):
                    if i == j:
                        continue
                    c = S[j]
                    if c % req_a == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            # Also check elem | a * c for all distinct a, c in S
            if can_add:
                for i in range(n):
                    a = S[i]
                    g = math.gcd(elem, a)
                    req_elem = elem // g  # for elem | a * c
                    for j in range(n):
                        if i == j:
                            continue
                        c = S[j]
                        if c % req_elem == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
            
            if not can_add:
                continue
            
            # Score: prefer larger numbers (they're "harder" to satisfy)
            score = elem
            
            if score > best_score:
                best_score = score
                best_elem = elem
        
        if best_elem is None:
            break
        
        S.append(best_elem)
        S_set.add(best_elem)
        remaining.remove(best_elem)
    
    return S


def local_search_remove(S_list, deadline):
    """
    Try to remove elements from S_list. If removing an element allows adding
    more elements back, do it.
    """
    S_set = set(S_list)
    S_list = list(S_list)
    best_size = len(S_list)
    
    while time.time() < deadline:
        improved = False
        # Try removing each element
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            elem_to_remove = S_list[i]
            new_S = S_list[:i] + S_list[i+1:]
            new_S_set = set(new_S)
            
            # Try to add back elements from RANGE
            for elem in RANGE:
                if elem in new_S_set:
                    continue
                # Check if elem can be added to new_S
                can_add = True
                n = len(new_S)
                for x in range(n):
                    a = new_S[x]
                    g = math.gcd(elem, a)
                    req_a = a // g
                    for y in range(n):
                        if x == y:
                            continue
                        c = new_S[y]
                        if c % req_a == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    for x in range(n):
                        a = new_S[x]
                        g = math.gcd(elem, a)
                        req_elem = elem // g
                        for y in range(n):
                            if x == y:
                                continue
                            c = new_S[y]
                            if c % req_elem == 0:
                                can_add = False
                                break
                        if not can_add:
                            break
                
                if can_add:
                    new_S.append(elem)
                    new_S_set.add(elem)
            
            if len(new_S) > best_size:
                S_list = new_S
                best_size = len(new_S)
                improved = True
                break
        
        if not improved:
            break
    
    return S_list


def local_search_swap(S_list, deadline):
    """
    Try swapping: remove one element, add another.
    """
    S_list = list(S_list)
    best_size = len(S_list)
    
    while time.time() < deadline:
        improved = False
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            elem_out = S_list[i]
            base_S = S_list[:i] + S_list[i+1:]
            base_S_set = set(base_S)
            
            for elem_in in RANGE:
                if elem_in in base_S_set:
                    continue
                # Check if elem_in can be added to base_S
                can_add = True
                n = len(base_S)
                for x in range(n):
                    a = base_S[x]
                    g = math.gcd(elem_in, a)
                    req_a = a // g
                    for y in range(n):
                        if x == y:
                            continue
                        c = base_S[y]
                        if c % req_a == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    for x in range(n):
                        a = base_S[x]
                        g = math.gcd(elem_in, a)
                        req_elem = elem_in // g
                        for y in range(n):
                            if x == y:
                                continue
                            c = base_S[y]
                            if c % req_elem == 0:
                                can_add = False
                                break
                        if not can_add:
                            break
                
                if can_add:
                    new_S = base_S + [elem_in]
                    if len(new_S) > best_size:
                        S_list = new_S
                        best_size = len(new_S)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return S_list


def simulate_annealing(S_list, deadline):
    """
    Simulated annealing: accept moves that don't improve with some probability.
    Move: remove one element, or swap one element.
    """
    S_list = list(S_list)
    best_S = list(S_list)
    best_size = len(S_list)
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline and T > 0.01:
        n = len(S_list)
        if n == 0:
            break
        
        # Random move
        move_type = random.random()
        new_S = list(S_list)
        
        if move_type < 0.5 and n > 1:
            # Remove a random element
            i = random.randint(0, n-1)
            new_S.pop(i)
        elif move_type < 0.8 and n > 0:
            # Swap
            i = random.randint(0, n-1)
            elem_out = S_list[i]
            base = S_list[:i] + S_list[i+1:]
            candidates = [e for e in RANGE if e not in set(base)]
            if candidates:
                elem_in = random.choice(candidates)
                # Check validity
                can_add = True
                for x in base:
                    g = math.gcd(elem_in, x)
                    req_a = x // g
                    for y in base:
                        if x == y:
                            continue
                        if y % req_a == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    for x in base:
                        g = math.gcd(elem_in, x)
                        req_elem = elem_in // g
                        for y in base:
                            if x == y:
                                continue
                            if y % req_elem == 0:
                                can_add = False
                                break
                        if not can_add:
                            break
                if can_add:
                    new_S = base + [elem_in]
                else:
                    continue
            else:
                continue
        else:
            # Add a random element
            candidates = [e for e in RANGE if e not in set(S_list)]
            if not candidates:
                continue
            elem_in = random.choice(candidates)
            can_add = True
            for x in S_list:
                g = math.gcd(elem_in, x)
                req_a = x // g
                for y in S_list:
                    if x == y:
                        continue
                    if y % req_a == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                for x in S_list:
                    g = math.gcd(elem_in, x)
                    req_elem = elem_in // g
                    for y in S_list:
                        if x == y:
                            continue
                        if y % req_elem == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
            if can_add:
                new_S = S_list + [elem_in]
            else:
                continue
        
        # Accept with probability
        delta = len(new_S) - len(S_list)
        if delta >= 0:
            S_list = new_S
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_S = list(new_S)
        else:
            if random.random() < math.exp(delta / T):
                S_list = new_S
        
        T *= cooling
    
    return best_S


def construct():
    start_time = time.time()
    total_deadline = start_time + 85.0
    
    best = []
    best_size = 0
    
    # Phase 1: Multiple greedy constructions
    n_greedy = 5
    greedy_time = (total_deadline - start_time) / (n_greedy + 3)
    
    for g in range(n_greedy):
        dl = start_time + greedy_time * (g + 1)
        random.shuffle(RANGE)  # This doesn't work since RANGE is a fixed list
        # Instead, we'll just run greedy which shuffles internally
        S = greedy_construct(dl)
        if len(S) > best_size:
            best_size = len(S)
            best = list(S)
    
    # Phase 2: Local search improvements
    remaining = total_deadline - time.time()
    if remaining > 5:
        ls_deadline = time.time() + remaining * 0.4
        S = local_search_remove(best, ls_deadline)
        if len(S) > best_size:
            best_size = len(S)
            best = list(S)
    
    remaining = total_deadline - time.time()
    if remaining > 5:
        ls_deadline = time.time() + remaining * 0.4
        S = local_search_swap(best, ls_deadline)
        if len(S) > best_size:
            best_size = len(S)
            best = list(S)
    
    # Phase 3: Simulated annealing
    remaining = total_deadline - time.time()
    if remaining > 10:
        sa_deadline = time.time() + remaining * 0.8
        S = simulate_annealing(best, sa_deadline)
        if len(S) > best_size:
            best_size = len(S)
            best = list(S)
    
    # Final verification
    res = list(best)
    if not is_valid(res):
        # Fallback
        res = [256]
    
    return res
