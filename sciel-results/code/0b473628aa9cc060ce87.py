import time
import random
from itertools import combinations

def has_2_coloring_and_get(n, edges):
    """
    Returns (is_2_colorable, coloring).
    coloring is a list of 0/1 if found, else None.
    """
    if not edges:
        return True, [0] * n

    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    all_mask = (1 << n) - 1
    
    def propagate(mask0, mask1):
        changed = True
        while changed:
            changed = False
            for em in edge_masks:
                c0 = bin(em & mask0).count('1')
                if c0 == 4:
                    remaining = em & ~(mask0 | mask1)
                    if remaining == 0:
                        return None, None, True
                    mask1 |= remaining
                    changed = True
                elif c0 == 5:
                    return None, None, True
                    
                c1 = bin(em & mask1).count('1')
                if c1 == 4:
                    remaining = em & ~(mask0 | mask1)
                    if remaining == 0:
                        return None, None, True
                    mask0 |= remaining
                    changed = True
                elif c1 == 5:
                    return None, None, True
        return mask0, mask1, False

    def solve(mask0, mask1):
        pm0, pm1, conflict = propagate(mask0, mask1)
        if conflict:
            return None
        if (pm0 | pm1) == all_mask:
            color = [0] * n
            for i in range(n):
                if (pm0 >> i) & 1: color[i] = 0
                else: color[i] = 1
            return color
        unassigned = all_mask ^ (pm0 | pm1)
        v_bit = unassigned & -unassigned
        
        # Try 0
        res = solve(pm0 | v_bit, pm1)
        if res: return res
        # Try 1
        res = solve(pm0, pm1 | v_bit)
        if res: return res
        return None

    coloring = solve(0, 0)
    if coloring:
        return True, coloring
    else:
        return False, None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = None
    best_count = float('inf')
    
    while time.time() < deadline:
        n = random.randint(9, 13)
        num_edges = random.randint(35, 50)
        all_possible = list(combinations(range(n), 5))
        random.shuffle(all_possible)
        edges = [list(e) for e in all_possible[:num_edges]]
        
        is_2c, coloring = has_2_coloring_and_get(n, edges)
        
        if not is_2c:
            S = edges[:]
            removals = 0
            max_removals = 100
            while removals < max_removals and time.time() < deadline:
                removals += 1
                if len(S) <= best_count:
                    break
                idx = random.randint(0, len(S) - 1)
                S_temp = S[:idx] + S[idx+1:]
                
                is_2c_temp, _ = has_2_coloring_and_get(n, S_temp)
                if not is_2c_temp:
                    S = S_temp
                    if len(S) < best_count:
                        best_count = len(S)
                        best_edges = S[:]
                else:
                    pass
                    
            if len(S) < best_count:
                best_count = len(S)
                best_edges = S[:]
                
        else:
            v0 = [i for i in range(n) if coloring[i] == 0]
            v1 = [i for i in range(n) if coloring[i] == 1]
            
            new_e = None
            if len(v0) >= 5:
                new_e = list(random.sample(v0, 5))
            elif len(v1) >= 5:
                new_e = list(random.sample(v1, 5))
                
            if new_e and new_e not in edges:
                edges.append(new_e)

    if best_edges is None:
        edges = [list(e) for e in combinations(range(9), 5)]
        return {'n': 9, 'edges': edges}
        
    return {'n': n, 'edges': best_edges}
