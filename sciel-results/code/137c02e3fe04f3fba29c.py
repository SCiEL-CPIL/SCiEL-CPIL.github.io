"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    
    DEADLINE = time.time() + 82
    
    N = 512
    rng = random.Random(42)
    
    # Build conflict bitsets
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if i * j // math.gcd(i, j) > N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def is_valid(s):
        s_list = sorted(s)
        for idx, a in enumerate(s_list):
            for b in s_list[idx+1:]:
                if (conflict_mask[a] >> b) & 1:
                    return False
        return True
    
    def greedy(order):
        result = set()
        for x in order:
            ok = True
            for y in result:
                if (conflict_mask[x] >> y) & 1:
                    ok = False
                    break
            if ok:
                result.add(x)
        return result
    
    # Strong initial solution: divisors of highly composite numbers + greedy fill
    def strong_initial():
        # Start with divisors of 360 (many small numbers with good lcm properties)
        s = set()
        for d in range(1, N + 1):
            if 360 % d == 0:
                s.add(d)
        # Also add divisors of other highly composite numbers
        for base in [120, 240, 480, 72, 144, 288]:
            for d in range(1, N + 1):
                if base % d == 0 and d not in s:
                    s.add(d)
        return s
    
    def add_nonconflicting(s):
        s = set(s)
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for x in range(1, N + 1):
                if x in s:
                    continue
                ok = True
                for y in s:
                    if (conflict_mask[x] >> y) & 1:
                        ok = False
                        break
                if ok:
                    s.add(x)
                    changed = True
        return s
    
    # Get initial best from multiple strategies
    best = strong_initial()
    best = add_nonconflicting(best)
    
    for order in [range(1, N + 1), range(N, 0, -1)]:
        cand = greedy(order)
        cand = add_nonconflicting(cand)
        if len(cand) > len(best):
            best = cand
    
    order3 = sorted(range(1, N + 1), key=lambda x: bin(conflict_mask[x]).count('1'))
    cand = greedy(order3)
    cand = add_nonconflicting(cand)
    if len(cand) > len(best):
        best = cand
    
    for _ in range(15):
        order5 = list(range(1, N + 1))
        rng.shuffle(order5)
        cand = greedy(order5)
        cand = add_nonconflicting(cand)
        if len(cand) > len(best):
            best = cand
    
    # Efficient local search with precomputed conflict union
    def local_search_swaps(s, max_rounds=20):
        s = set(s)
        
        for _ in range(max_rounds):
            if time.time() >= DEADLINE:
                break
            improved = False
            s_list = sorted(s)
            for x in s_list:
                if time.time() >= DEADLINE:
                    break
                # Compute blocked set for s \ {x}
                blocked = 0
                for z in s:
                    if z != x:
                        blocked |= conflict_mask[z]
                
                # Available: not in s, not blocked
                s_bits = 0
                for elem in s:
                    s_bits |= (1 << elem)
                
                available = (~blocked) & (~s_bits) & ((1 << (N + 1)) - 1) & ~1
                available &= ~s_bits
                
                # Try to find a beneficial swap
                while available:
                    lsb = available & (-available)
                    y = lsb.bit_length() - 1
                    if y < 1 or y > N:
                        available &= available - 1
                        continue
                    # Check if y actually doesn't conflict with s\{x}
                    # (already guaranteed by construction, but verify)
                    s.remove(x)
                    s.add(y)
                    improved = True
                    break
                    available &= available - 1
                
                if improved:
                    break
            if not improved:
                break
        return s
    
    best = local_search_swaps(best, max_rounds=40)
    
    # More restarts with local search
    for _ in range(50):
        if time.time() >= DEADLINE:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = greedy(order)
        candidate = add_nonconflicting(candidate)
        candidate = local_search_swaps(candidate, max_rounds=15)
        if len(candidate) > len(best):
            best = candidate
    
    # Final polish: try to add any remaining non-conflicting elements
    best = add_nonconflicting(best)
    best = local_search_swaps(best, max_rounds=20)
    best = add_nonconflicting(best)
    
    result = sorted(best)
    if not is_valid(set(result)):
        return list(range(1, math.isqrt(N) + 1))
    
    return result
# EVOLVE-BLOCK-END
