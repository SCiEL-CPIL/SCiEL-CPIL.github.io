import random
import time
from itertools import combinations
from math import gcd, exp

def construct():
    deadline = time.time() + 148
    
    # Build a dense pool with structured points for many collinear triples
    pool = []
    # Dense grid
    for x in range(-40, 41, 2):
        for y in range(-40, 41, 2):
            pool.append((x, y))
    # Add points on several lines/curves to increase collinearity
    for x in range(-30, 31):
        pool.append((x, x * x))
        pool.append((x, 2 * x * x))
        pool.append((x, -x * x))
        pool.append((x, x * x + x))
    # Add some diagonal lines
    for d in range(-20, 21, 3):
        for x in range(-30, 31, 3):
            pool.append((x, x + d))
    
    # Remove duplicates and limit
    pool = list(set(pool))
    random.shuffle(pool)
    pool = pool[:400]
    n_pool = len(pool)
    
    # Precompute line keys
    def get_line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            g = 1
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
    
    lines = []
    for lk, idx_set in line_map.items():
        if len(idx_set) >= 3:
            lines.append(sorted(idx_set))
    
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

    def count_triples(selected_indices):
        sel_set = set(selected_indices)
        triples = 0
        for line in lines:
            count = 0
            for idx in line:
                if idx in sel_set:
                    count += 1
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6
        return triples

    # Fast exact solver using bitmask branch and bound
    def calc_exact_in(selected_indices):
        n = 20
        local_map = {idx: i for i, idx in enumerate(selected_indices)}
        
        # Build conflict: for each point i, for each point j, mask of points k collinear with i,j
        conflict = [[0] * n for _ in range(n)]
        
        for line in lines:
            local_line = [local_map[idx] for idx in line if idx in local_map]
            if len(local_line) >= 3:
                for i in local_line:
                    for j in local_line:
                        if i == j:
                            continue
                        for k in local_line:
                            if k != i and k != j:
                                conflict[i][j] |= (1 << k)
        
        # Also build for each pair (i,j) the set of points that would complete a triple
        # If we have selected set S and want to add point v, we need:
        # For all u in S, (conflict[v][u] & S) == 0
        
        full_mask = (1 << n) - 1
        best_size = 0
        
        def backtrack(avail, selected, count):
            nonlocal best_size
            if count > best_size:
                best_size = count
            if count + bin(avail).count('1') <= best_size:
                return
            
            # Pick a candidate from avail
            if avail == 0:
                return
            
            # Heuristic: pick point with most conflicts with selected
            # Try each available point
            node = (avail & -avail).bit_length() - 1
            rest = avail & ~(1 << node)
            
            # Option 1: don't pick node
            backtrack(rest, selected, count)
            
            # Option 2: pick node if valid
            valid = True
            temp = selected
            while temp:
                u_bit = temp & -temp
                u = u_bit.bit_length() - 1
                if (conflict[node][u] & selected) != 0:
                    valid = False
                    break
                temp ^= u_bit
            
            if valid:
                backtrack(rest, selected | (1 << node), count + 1)
        
        backtrack(full_mask, 0, 0)
        return best_size

    # Simulated Annealing
    def sa_optimize(seed, time_budget):
        random.seed(seed)
        
        # Initial solution
        best_solution = None
        for _ in range(100):
            candidate = random.sample(range(n_pool), 20)
            if is_valid(candidate):
                best_solution = candidate
                break
        if best_solution is None:
            best_solution = random.sample(range(n_pool), 20)
        
        best_score = count_triples(best_solution)
        current_solution = best_solution[:]
        current_score = best_score
        
        temp = 20.0
        cooling = 0.9995
        min_temp = 0.5
        
        while time.time() < deadline - time_budget and temp > min_temp:
            sel_idx = random.randint(0, 19)
            unselected = [i for i in range(n_pool) if i not in set(current_solution)]
            if not unselected:
                break
            new_point = random.choice(unselected)
            
            new_solution = current_solution[:]
            new_solution[sel_idx] = new_point
            
            if not is_valid(new_solution):
                temp *= cooling
                continue
                
            new_score = count_triples(new_solution)
            
            delta = new_score - current_score
            if delta >= 0:
                accept = True
            else:
                exponent = -delta / temp
                if exponent > 500:
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
        
        return best_solution, best_score

    # Multi-start SA
    best_overall = None
    best_triples = -1
    
    # Run several SA restarts
    for seed in [42, 123, 456, 789]:
        if time.time() > deadline - 30:
            break
        solution, triples = sa_optimize(seed, 30)
        if triples > best_triples:
            best_triples = triples
            best_overall = solution
    
    # Final local search with exact evaluation
    best_exact = calc_exact_in(best_overall)
    best_exact_solution = best_overall[:]
    
    # Try improvements
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        sel_idx = random.randint(0, 19)
        unselected = [i for i in range(n_pool) if i not in set(best_exact_solution)]
        if not unselected:
            continue
        new_point = random.choice(unselected)
        
        candidate = best_exact_solution[:]
        candidate[sel_idx] = new_point
        
        if not is_valid(candidate):
            continue
        
        # Use proxy first to filter
        if count_triples(candidate) < count_triples(best_exact_solution) - 3:
            continue
        
        exact_cand = calc_exact_in(candidate)
        if exact_cand < best_exact:
            best_exact = exact_cand
            best_exact_solution = candidate[:]
    
    return [[pool[i][0], pool[i][1]] for i in best_exact_solution]
