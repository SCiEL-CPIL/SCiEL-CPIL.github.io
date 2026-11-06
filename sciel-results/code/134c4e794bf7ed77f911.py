import math
import random
import time

def construct():
    deadline = time.time() + 115.0
    N = 511
    C = 512
    
    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)
    
    # Precompute coprime masks as bitsets
    # coprime_mask[a] is a bitset where bit b is set if gcd(a, b) == 1
    # We use Python integers as bitsets (N+1 bits, which fits easily)
    coprime_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a != b and math.gcd(a, b) == 1:
                mask |= (1 << b)
        coprime_mask[a] = mask
    
    # Conflict mask: conflict_mask[a] has bit b set if gcd(a,b) > 1
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a != b and math.gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_mask[a] = mask
    
    def set_to_mask(S):
        mask = 0
        for a in S:
            mask |= (1 << a)
        return mask
    
    def mask_to_set(mask):
        S = set()
        for a in range(1, N + 1):
            if mask & (1 << a):
                S.add(a)
        return S
    
    def can_add_bitset(a, S_mask):
        # a can be added if no element in S_mask conflicts with a
        # i.e., S_mask & conflict_mask[a] == 0
        return (S_mask & conflict_mask[a]) == 0
    
    def find_best_add_bitset(S_mask):
        best_a = None
        best_w_val = -1.0
        for a in range(1, N + 1):
            if S_mask & (1 << a):
                continue
            if can_add_bitset(a, S_mask):
                if w[a] > best_w_val:
                    best_w_val = w[a]
                    best_a = a
        return best_a
    
    # Initial Solution: Greedy by weight (descending a)
    S = set()
    S_mask = 0
    S.add(1)
    S_mask |= (1 << 1)
    
    # Sort numbers by weight descending (i.e., a descending)
    candidates = list(range(N, 0, -1))
    for a in candidates:
        if S_mask & (1 << a):
            continue
        if can_add_bitset(a, S_mask):
            S.add(a)
            S_mask |= (1 << a)
            
    current_score = sum(w[a] for a in S)
    best_S = S.copy()
    best_score = current_score
    
    rng = random.Random(42)
    
    # Local Search with bitset operations
    improved = True
    stall_count = 0
    max_stall = 50
    
    while time.time() < deadline - 5.0:
        if improved and stall_count > max_stall:
            # Perturbation: remove 2-3 random elements and re-add greedily
            S_list = list(S)
            rng.shuffle(S_list)
            n_remove = min(3, len(S_list))
            for x in S_list[:n_remove]:
                S.discard(x)
                S_mask &= ~(1 << x)
            
            # Re-add greedily
            for a in range(N, 0, -1):
                if S_mask & (1 << a):
                    continue
                if can_add_bitset(a, S_mask):
                    S.add(a)
                    S_mask |= (1 << a)
            
            current_score = sum(w[a] for a in S)
            stall_count = 0
            improved = False
            continue
        
        improved = False
        S_list = list(S)
        rng.shuffle(S_list)
        
        for x in S_list:
            if time.time() > deadline - 5.0:
                break
            
            new_S_mask = S_mask & ~(1 << x)
            
            # Try 1-Add: Remove x, Add best y
            y = find_best_add_bitset(new_S_mask)
            if y is not None:
                delta = w[y] - w[x]
                if delta > 1e-9:
                    S_mask = new_S_mask | (1 << y)
                    S = mask_to_set(S_mask)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S.copy()
                    improved = True
                    stall_count = 0
                    break
                    
            # Try 2-Add: Remove x, Add y, Add z
            # Find top candidates for y
            y_candidates = []
            for a in range(1, N + 1):
                if new_S_mask & (1 << a):
                    continue
                if can_add_bitset(a, new_S_mask):
                    y_candidates.append(a)
            
            y_candidates.sort(key=lambda a: -w[a])
            
            for y in y_candidates[:8]:
                new_S_y_mask = new_S_mask | (1 << y)
                z = find_best_add_bitset(new_S_y_mask)
                if z is not None:
                    delta = w[y] + w[z] - w[x]
                    if delta > 1e-9:
                        S_mask = new_S_y_mask | (1 << z)
                        S = mask_to_set(S_mask)
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S.copy()
                        improved = True
                        stall_count = 0
                        break
                        
            if improved:
                break
            
            # Try 2-Remove-1-Add: Remove x, v, Add best u
            # This is important when two elements block one high-weight element
            for v in S_list:
                if v == x:
                    continue
                new_S_v_mask = new_S_mask & ~(1 << v)
                u = find_best_add_bitset(new_S_v_mask)
                if u is not None:
                    delta = w[u] - w[x] - w[v]
                    if delta > 1e-9:
                        S_mask = new_S_v_mask | (1 << u)
                        S = mask_to_set(S_mask)
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S.copy()
                        improved = True
                        stall_count = 0
                        break
            if improved:
                break
            
            # Try 1-Remove: Just remove x if it has low weight and doesn't enable better additions
            # This is rarely beneficial but can help escape local optima
            if len(S) > 1 and w[x] < 0.02:
                new_S_mask = S_mask & ~(1 << x)
                # Check if we can add something back
                y = find_best_add_bitset(new_S_mask)
                if y is None or w[y] < w[x] * 0.5:
                    # Removing x and not adding back might be good if x is very low weight
                    # But this is risky, so we only do it if we're stuck
                    if stall_count > 20:
                        S_mask = new_S_mask
                        S = mask_to_set(S_mask)
                        current_score -= w[x]
                        improved = True
                        stall_count = 0
                        break
        
        if not improved:
            stall_count += 1
    
    # Final polish: try all possible single swaps and double swaps
    for x in list(S):
        if time.time() > deadline - 2.0:
            break
        new_S_mask = S_mask & ~(1 << x)
        y = find_best_add_bitset(new_S_mask)
        if y is not None:
            delta = w[y] - w[x]
            if delta > 1e-9:
                S_mask = new_S_mask | (1 << y)
                S = mask_to_set(S_mask)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
    
    return list(best_S)
