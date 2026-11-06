"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    
    deadline = time.time() + 80
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [1.0 / d for d in denominators]
    
    # Greedy initialization: add items in order of largest reciprocal first
    order = sorted(range(n), key=lambda i: -recip[i])
    selected = set()
    current_sum = 0.0
    
    for i in order:
        if current_sum + recip[i] < 1.0:
            selected.add(i)
            current_sum += recip[i]
    
    best_sum = current_sum
    best_set = frozenset(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-add moves
        for i in range(n):
            if i not in selected:
                ns = current_sum + recip[i]
                if ns < 1.0 and ns > best_sum:
                    selected.add(i)
                    current_sum = ns
                    best_sum = ns
                    best_set = frozenset(selected)
                    improved = True
        
        # 1-remove + 1-add moves
        if not improved and time.time() < deadline - 2:
            for i in list(selected):
                if time.time() > deadline - 2:
                    break
                for j in range(n):
                    if j in selected:
                        continue
                    ns = current_sum - recip[i] + recip[j]
                    if ns < 1.0 and ns > best_sum:
                        selected.discard(i)
                        selected.add(j)
                        current_sum = ns
                        best_sum = ns
                        best_set = frozenset(selected)
                        improved = True
                        break
                if improved:
                    break
        
        # 2-remove + 2-add moves
        if not improved and time.time() < deadline - 10:
            sel_list = list(selected)
            for ii in range(len(sel_list)):
                if time.time() > deadline - 5:
                    break
                for jj in range(ii + 1, len(sel_list)):
                    i, j = sel_list[ii], sel_list[jj]
                    base = current_sum - recip[i] - recip[j]
                    for k in range(n):
                        if k in selected:
                            continue
                        s1 = base + recip[k]
                        if s1 >= 1.0:
                            continue
                        for l in range(n):
                            if l in selected or l == k:
                                continue
                            s2 = s1 + recip[l]
                            if s2 < 1.0 and s2 > best_sum:
                                selected.discard(i)
                                selected.discard(j)
                                selected.add(k)
                                selected.add(l)
                                current_sum = s2
                                best_sum = s2
                                best_set = frozenset(selected)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    result = sorted(best_set)
    
    # Verify with exact arithmetic
    total = Fraction(0)
    for idx in result:
        total += Fraction(1, denominators[idx])
    
    if total >= 1 or not result:
        return [0]
    
    return result
# EVOLVE-BLOCK-END
