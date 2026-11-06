"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    import time
    
    deadline = time.time() + 85.0
    
    # Precompute conflicts: x and y conflict if lcm(x,y) <= 256
    conflicts = [set() for _ in range(257)]
    for a in range(2, 257):
        for b in range(a + 1, 257):
            if a * b // gcd(a, b) <= 256:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    def is_compatible(x, S):
        for y in S:
            if y in conflicts[x]:
                return False
        return True
    
    def total_score(S):
        return sum(1.0 / x for x in S)
    
    # Greedy initialization: add numbers in order of increasing value (small x first)
    selected = []
    for x in range(2, 257):
        if all(y not in conflicts[x] for y in selected):
            selected.append(x)
    
    best_set = set(selected)
    best_score = total_score(best_set)
    
    all_numbers = list(range(2, 257))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a number not in set
        for x in all_numbers:
            if x in best_set:
                continue
            if is_compatible(x, best_set):
                best_score += 1.0 / x
                best_set.add(x)
                improved = True
                break
        
        if time.time() >= deadline:
            break
        
        # Try swaps: remove x, add y
        for x in list(best_set):
            if time.time() >= deadline:
                break
            best_y = None
            best_gain = 0.0
            for y in all_numbers:
                if y in best_set or y == x:
                    continue
                gain = 1.0 / y - 1.0 / x
                if gain <= 0:
                    continue
                ok = True
                for z in best_set:
                    if z != x and z in conflicts[y]:
                        ok = False
                        break
                if ok and gain > best_gain:
                    best_gain = gain
                    best_y = y
            if best_y is not None:
                best_set.discard(x)
                best_set.add(best_y)
                best_score += best_gain
                improved = True
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
