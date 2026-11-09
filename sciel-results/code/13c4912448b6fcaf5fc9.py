import random
import time
from itertools import combinations
from math import gcd, exp

def construct():
    deadline = time.time() + 145
    
    # 1. Generate Pool
    # We want points that create many collinear triples to minimize the max independent set.
    # Grid points are good.
    pool = []
    for x in range(-50, 51, 2):
        for y in range(-50, 51, 2):
            pool.append((x, y))
    
    # Add some structured points to ensure density
    for x in range(-20, 21):
        pool.append((x, x*x))
        pool.append((x, 2*x*x))
        pool.append((x, -x*x))
        
    # Remove duplicates
    pool = list(set(pool))
    random.shuffle(pool)
    # Limit pool size for speed
    pool = pool[:300]
    n_pool = len(pool)
    
    # Precompute line keys for all pairs in pool
    def get_line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        if g == 0: g = 1
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * p1[0] - dx_n * p1[1]
        return (dx_n, dy_n, c)

    line_map = {}
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            lk = get_line_key(pool[i], pool[j])
            if lk:
                if lk not in line_map:
                    line_map[lk] = set()
                line_map[lk].add(i)
                line_map[lk].add(j)
    
    # Build list of lines with >= 3 points
    lines = []
    for lk, idx_set in line_map.items():
        if len(idx_set) >= 3:
            lines.append(sorted(idx_set))
            
    # Helper: Check if a set of indices is valid (no 4 collinear)
    def is_valid(selected_indices):
        sel_set = set(selected_indices)
        for line in lines:
            count = 0
            for idx in line:
                if idx in sel_set:
                    count += 1
                    if count >= 4:
                        return False
        return True

    # Helper: Compute number of collinear triples
    def count_triples(selected_indices):
        sel_set = set(selected_indices)
        triples = 0
        for line in lines:
            count = 0
            for idx in line:
                if idx in sel_set:
                    count += 1
            if count >= 3:
                if count == 3:
                    triples += 1
                elif count == 4:
                    triples += 4
                elif count == 5:
                    triples += 10
        return triples

    # Exact Solver for Max No-3-Collinear Subset
    def calc_exact_in(selected_indices):
        n = 20
        local_map = {idx: i for i, idx in enumerate(selected_indices)}
        
        # Precompute conflict masks: conflict[v][u] is bitmask of w such that (v,u,w) is collinear
        # Note: (v,u,w) collinear implies (v,w,u) etc.
        # We only need to check if adding v to S creates a triple.
        # If we have S, and want to add v, we check for any u in S if there exists w in S such that (v,u,w) is a triple.
        # This is equivalent to: for any u in S, (conflict[v][u] & S) != 0.
        
        conflict = [[0]*n for _ in range(n)]
        
        for line in lines:
            local_line = [local_map[idx] for idx in line if idx in local_map]
            if len(local_line) >= 3:
                for i_idx, i in enumerate(local_line):
                    for j in local_line:
                        if i == j: continue
                        for k in local_line:
                            if k != i and k != j:
                                conflict[i][j] |= (1 << k)
        
        # Search for largest valid subset size
        # We try sizes from 20 down to 1
        # For a given size k, we do a backtracking search to see if a valid subset of size k exists.
        
        def exists_valid_subset(size_k, avail_mask, selected_mask, count):
            if count == size_k:
                return True
            if count + bin(avail_mask).count('1') < size_k:
                return False
                
            # Pick a node from avail
            # Heuristic: pick node with highest degree in conflict graph?
            # For simplicity, pick first available.
            node = (avail_mask & -avail_mask).bit_length() - 1
            rest_avail = avail_mask & ~(1 << node)
            
            # Option 1: Don't pick node
            if exists_valid_subset(size_k, rest_avail, selected_mask, count):
                return True
            
            # Option 2: Pick node
            # Check if node can be added to selected_mask
            valid = True
            temp_sel = selected_mask
            while temp_sel:
                u_bit = temp_sel & -temp_sel
                u = u_bit.bit_length() - 1
                if (conflict[node][u] & selected_mask) != 0:
                    valid = False
                    break
                temp_sel ^= u_bit
                
            if valid:
                if exists_valid_subset(size_k, rest_avail, selected_mask | (1 << node), count + 1):
                    return True
            return False
            
        for k in range(20, 0, -1):
            if exists_valid_subset(k, (1 << 20) - 1, 0, 0):
                return k
        return 0

    # Phase 1: Simulated Annealing
    def proxy_score(selected):
        if not is_valid(selected):
            return -1000000
        return count_triples(selected)
    
    best_solution = random.sample(range(n_pool), 20)
    while not is_valid(best_solution):
        best_solution = random.sample(range(n_pool), 20)
        
    best_score = proxy_score(best_solution)
    current_solution = best_solution[:]
    current_score = best_score
    
    temp = 10.0
    cooling = 0.999
    min_temp = 0.1
    step = 0
    
    while time.time() < deadline - 40: # Leave 40s for exact eval and local search
        sel_idx = random.randint(0, 19)
        unselected = [i for i in range(n_pool) if i not in set(current_solution)]
        if not unselected: break
        new_point = random.choice(unselected)
        
        new_solution = current_solution[:]
        new_solution[sel_idx] = new_point
        
        new_score = proxy_score(new_solution)
        
        delta = new_score - current_score
        if delta >= 0:
            accept = True
        else:
            exponent = -delta / temp
            if exponent > 700:
                accept = False
            else:
                accept = random.random() < exp(-exponent)
        if accept:
            current_solution = new_solution
            current_score = new_score
            if current_score > best_score:
                best_solution = current_solution[:]
                best_score = current_score
        
        temp *= cooling
        step += 1
        
    # Phase 2: Local Search on Exact Score
    # We take the best solution from SA and try to improve it using exact score.
    # Since exact score is expensive, we only do this for the best few candidates.
    # For now, we just evaluate the best solution and try a few swaps.
    
    best_exact = calc_exact_in(best_solution)
    best_exact_solution = best_solution[:]
    
    # Try 20 random swaps to see if we can lower the exact score
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        sel_idx = random.randint(0, 19)
        unselected = [i for i in range(n_pool) if i not in set(best_exact_solution)]
        if not unselected: continue
        new_point = random.choice(unselected)
        
        candidate = best_exact_solution[:]
        candidate[sel_idx] = new_point
        
        if not is_valid(candidate):
            continue
            
        # Check if this candidate has a lower exact score
        # To save time, we can first check if it has more triples (proxy)
        # If proxy is worse, it's unlikely to have a lower exact score, but not guaranteed.
        # Let's just compute exact if proxy is better or similar.
        if count_triples(candidate) < count_triples(best_exact_solution) - 5:
             continue # Likely worse, skip exact calc to save time? 
             # Actually, we want MIN exact score. More triples usually means lower max independent set.
             # So if triples are fewer, exact score is likely higher (worse).
             # So if triples(candidate) < triples(best) - 5, skip.
        
        exact_cand = calc_exact_in(candidate)
        if exact_cand < best_exact:
            best_exact = exact_cand
            best_exact_solution = candidate[:]
            
    return [[pool[i][0], pool[i][1]] for i in best_exact_solution]
