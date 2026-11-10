import time
import random
import sys

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin

    # 1. Precompute all 4-APs and conflict masks using bitsets
    masks = [[] for _ in range(N + 1)]
    
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            
            m = (1 << x1) | (1 << x2) | (1 << x3) | (1 << x4)
            
            masks[x1].append(m ^ (1 << x1))
            masks[x2].append(m ^ (1 << x2))
            masks[x3].append(m ^ (1 << x3))
            masks[x4].append(m ^ (1 << x4))

    def can_add(x, S):
        """Check if x can be added to set S (represented as int bitset)"""
        for m in masks[x]:
            if (m & S).bit_count() >= 3:
                return False
        return True

    def greedy_add(S, candidates):
        """Greedily add elements from candidates to S"""
        for v in candidates:
            if not ((S >> v) & 1):
                if can_add(v, S):
                    S |= (1 << v)
        return S

    def greedy_add_priority(S, candidates):
        """Greedily add elements, prioritizing those that conflict least"""
        # Sort candidates by how many conflicts they'd create
        scored = []
        for v in candidates:
            if (S >> v) & 1:
                continue
            # Count how many 3-element subsets of APs containing v are in S
            conflict_score = 0
            for m in masks[v]:
                c = (m & S).bit_count()
                if c >= 3:
                    conflict_score += 100  # Can't add
                else:
                    conflict_score += c
            scored.append((conflict_score, v))
        
        scored.sort()
        for _, v in scored:
            if can_add(v, S):
                S |= (1 << v)
        return S

    # 2. Initial Heuristics
    best_set = 0
    best_size = 0
    
    # Heuristic 1: Base-4 (Salem-Spencer)
    S_base4 = 0
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            S_base4 |= (1 << i)
    
    if S_base4.bit_count() > best_size:
        best_set = S_base4
        best_size = S_base4.bit_count()

    # Heuristic 2: Multiple random greedy starts
    for seed in range(30):
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = 0
        for x in order:
            if can_add(x, S):
                S |= (1 << x)
        
        if S.bit_count() > best_size:
            best_set = S
            best_size = S.bit_count()

    # Heuristic 3: Greedy from base4 with priority
    S = S_base4
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    S = greedy_add(S, candidates)
    if S.bit_count() > best_size:
        best_set = S
        best_size = S.bit_count()

    # 3. Local Search with iterative deepening
    rng = random.Random(42)
    
    def elements_in(S):
        return [i for i in range(1, N + 1) if (S >> i) & 1]

    def elements_out(S):
        return [i for i in range(1, N + 1) if not ((S >> i) & 1)]

    # Phase 1: Single removal + repair
    while time.time() < deadline - 10:
        S = best_set
        elems = elements_in(S)
        if not elems:
            break
        
        u = rng.choice(elems)
        S_temp = S & ~(1 << u)
        
        # Try multiple repair orders
        improved = False
        for attempt in range(3):
            candidates = elements_out(S_temp)
            rng.shuffle(candidates)
            S_new = greedy_add(S_temp, candidates)
            
            if S_new.bit_count() > best_size:
                best_set = S_new
                best_size = S_new.bit_count()
                improved = True
                break
        
        if not improved and time.time() < deadline - 10:
            # Try with priority ordering
            candidates = elements_out(S_temp)
            S_new = greedy_add_priority(S_temp, candidates)
            if S_new.bit_count() > best_size:
                best_set = S_new
                best_size = S_new.bit_count()

    # Phase 2: Double removal + repair
    while time.time() < deadline - 5:
        S = best_set
        elems = elements_in(S)
        if len(elems) < 2:
            break
        
        u1, u2 = rng.sample(elems, 2)
        S_temp = S & ~(1 << u1) & ~(1 << u2)
        
        for attempt in range(2):
            candidates = elements_out(S_temp)
            rng.shuffle(candidates)
            S_new = greedy_add(S_temp, candidates)
            
            if S_new.bit_count() > best_size:
                best_set = S_new
                best_size = S_new.bit_count()
                break

    # Phase 3: Triple removal + repair
    while time.time() < deadline - 3:
        S = best_set
        elems = elements_in(S)
        if len(elems) < 3:
            break
        
        u1, u2, u3 = rng.sample(elems, 3)
        S_temp = S & ~(1 << u1) & ~(1 << u2) & ~(1 << u3)
        
        candidates = elements_out(S_temp)
        rng.shuffle(candidates)
        S_new = greedy_add(S_temp, candidates)
        
        if S_new.bit_count() > best_size:
            best_set = S_new
            best_size = S_new.bit_count()

    # Phase 4: Swap refinement - replace one element with potentially multiple
    while time.time() < deadline - 1:
        S = best_set
        elems_in = elements_in(S)
        elems_out = elements_out(S)
        
        if not elems_in or not elems_out:
            break
        
        u = rng.choice(elems_in)
        S_temp = S & ~(1 << u)
        
        # Try adding multiple elements from outside
        candidates = elems_out[:]
        rng.shuffle(candidates)
        
        S_new = S_temp
        added = 0
        for v in candidates:
            if can_add(v, S_new):
                S_new |= (1 << v)
                added += 1
            if added >= 2:
                break
        
        if S_new.bit_count() > best_size:
            best_set = S_new
            best_size = S_new.bit_count()

    # Final polish: try all single swaps
    if time.time() < deadline - 0.5:
        S = best_set
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            elems_in = elements_in(S)
            elems_out = elements_out(S)
            
            for u in elems_in:
                if time.time() > deadline - 0.5:
                    break
                S_temp = S & ~(1 << u)
                for v in elems_out:
                    if can_add(v, S_temp):
                        S_new = S_temp | (1 << v)
                        # Try to add more
                        remaining = [x for x in elems_out if x != v]
                        rng.shuffle(remaining)
                        S_final = greedy_add(S_new, remaining)
                        
                        if S_final.bit_count() > S.bit_count():
                            S = S_final
                            improved = True
                            break
                if improved:
                    break
        
        if S.bit_count() > best_size:
            best_set = S
            best_size = S.bit_count()

    # Convert bitset to list
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    return sorted(result)
