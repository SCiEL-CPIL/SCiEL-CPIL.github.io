"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 256
    
    # Precompute 4-AP conflict masks: for each x, which 4-APs contain x
    # And for each 4-AP, the set of elements
    four_aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    three_aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 2 + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    # For fast 4-AP checking, precompute for each element x, the list of 4-APs it's in
    element_to_4aps = {x: [] for x in range(1, n + 1)}
    for i, (a, b, c, d) in enumerate(four_aps):
        for x in (a, b, c, d):
            element_to_4aps[x].append(i)
    
    def can_add_fast(x, S_set):
        for idx in element_to_4aps[x]:
            a, b, c, d = four_aps[idx]
            others = [y for y in (a, b, c, d) if y != x]
            if all(y in S_set for y in others):
                return False
        return True
    
    def count_3aps_fast(S_set):
        count = 0
        for (a, b, c) in three_aps:
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count
    
    # Greedy
    S = set()
    for x in range(1, n + 1):
        if can_add_fast(x, S):
            S.add(x)
    
    best_S = set(S)
    best_score = count_3aps_fast(best_S)
    
    # Local search
    for restart in range(200):
        S = set(best_S)
        if restart > 0:
            S_list = sorted(S)
            k = random.randint(1, min(5, len(S)))
            to_remove = random.sample(S_list, k)
            for x in to_remove:
                S.discard(x)
            for x in range(1, n + 1):
                if x not in S and can_add_fast(x, S):
                    S.add(x)
        
        improved = True
        while improved:
            improved = False
            cs = count_3aps_fast(S)
            # Try adding
            for x in range(1, n + 1):
                if x not in S and can_add_fast(x, S):
                    new_S = S | {x}
                    ns = count_3aps_fast(new_S)
                    if ns > cs:
                        S = new_S
                        cs = ns
                        improved = True
                        break
            # Try swap
            if not improved:
                S_list = list(S)
                random.shuffle(S_list)
                for x_out in S_list[:30]:
                    for x_in in range(1, n + 1):
                        if x_in in S:
                            continue
                        new_S = (S - {x_out}) | {x_in}
                        # Check 4-AP
                        ok = True
                        for idx in element_to_4aps[x_in]:
                            a, b, c, d = four_aps[idx]
                            if all(y in new_S for y in (a, b, c, d)):
                                ok = False
                                break
                        if not ok:
                            continue
                        ns = count_3aps_fast(new_S)
                        if ns > cs:
                            S = new_S
                            cs = ns
                            improved = True
                            break
                    if improved:
                        break
        
        fs = count_3aps_fast(S)
        if fs > best_score:
            best_S = set(S)
            best_score = fs
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
