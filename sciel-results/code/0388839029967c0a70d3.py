import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    def compute_score(U):
        if not U or 0 not in U or len(U) < 2:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        # Use vectorized operations for speed
        U_arr = np.array(U, dtype=np.int64)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        if n_sum == 0:
            return -1
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        return 1 + math.log(ratio) / math.log(base)
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand or len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Known good Golomb rulers and structured sets
    known_sets = [
        [0, 1, 4, 10, 18, 23, 29, 34, 42, 47],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10],
        [0, 1, 4],
        [0, 1, 3],
        [0, 1, 4, 9, 15, 22, 31],
        [0, 1, 4, 9, 15, 22],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 9],
        [0, 2, 7, 15, 22, 31, 39],
        [0, 2, 7, 15, 22, 31],
        [0, 2, 7, 15, 22],
        [0, 2, 7, 15],
        [0, 1, 5, 12, 22, 30, 41],
        [0, 1, 5, 12, 22, 30],
        [0, 1, 5, 12, 22],
        [0, 1, 5, 12],
    ]
    
    for s in known_sets:
        try_set(s)
    
    # Base-B digit representation genetic algorithm
    def to_digits(val, base, num_digits):
        """Represent val in base, padded to num_digits digits"""
        digits = []
        v = val
        for _ in range(num_digits):
            digits.append(v % base)
            v //= base
        return digits
    
    def from_digits(digits, base):
        val = 0
        for d in reversed(digits):
            val = val * base + d
        return val
    
    def mutate_digits(digits, base, mutation_rate=0.3):
        """Mutate individual digits"""
        new_digits = list(digits)
        for i in range(len(new_digits)):
            if random.random() < mutation_rate:
                new_digits[i] = random.randint(0, base - 1)
        return new_digits
    
    def crossover(d1, d2):
        """Uniform crossover of digit vectors"""
        n = len(d1)
        child = []
        for i in range(n):
            child.append(d1[i] if random.random() < 0.5 else d2[i])
        return child
    
    def local_search(cand, max_iters=200):
        """Local search: try adding, removing, replacing elements"""
        curr = sorted(set(cand))
        if not curr or 0 not in curr or len(curr) < 2:
            return curr
        curr_score = compute_score(curr)
        no_improve = 0
        max_no_improve = 20
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            n_curr = len(curr)
            max_cur = max(curr) if curr else 0
            curr_set = set(curr)
            
            # Try adding - focus on gaps and structured positions
            if n_curr < 15:
                candidates = []
                # Small gaps
                for i in range(1, max_cur + 1):
                    if i not in curr_set:
                        candidates.append(i)
                # Structured positions based on existing elements
                for el in curr:
                    for delta in [1, 2, 3, 4, 5, 8, 10, 13]:
                        new_el = el + delta
                        if new_el > max_cur + 20 and new_el > 50:
                            continue
                        if new_el not in curr_set:
                            candidates.append(new_el)
                random.shuffle(candidates)
                for new_el in candidates[:30]:  # Limit evaluations per iteration
                    if new_el in curr_set:
                        continue
                    cand2 = sorted(curr + [new_el])
                    s = compute_score(cand2)
                    if s > curr_score:
                        curr = cand2
                        curr_score = s
                        improved = True
                        no_improve = 0
                        try_set(curr)
                        break
            
            # Try replacing
            if not improved and n_curr >= 2:
                for i in range(n_curr):
                    if curr[i] == 0:
                        continue
                    old_el = curr[i]
                    # Try nearby replacements and structured ones
                    replacement_candidates = []
                    for delta in range(-10, 11):
                        new_el = old_el + delta
                        if new_el > 0 and new_el not in curr_set and new_el != old_el:
                            replacement_candidates.append(new_el)
                    for delta in [1, 2, 3, 5, 8, 13, 21]:
                        new_el = old_el + delta
                        if new_el > 0 and new_el not in curr_set:
                            replacement_candidates.append(new_el)
                    random.shuffle(replacement_candidates)
                    for new_el in replacement_candidates[:20]:
                        if new_el in curr_set:
                            continue
                        cand2 = list(curr)
                        cand2[i] = new_el
                        cand2 = sorted(set(cand2))
                        if 0 not in cand2 or len(cand2) < 2:
                            continue
                        s = compute_score(cand2)
                        if s > curr_score:
                            curr = cand2
                            curr_score = s
                            improved = True
                            no_improve = 0
                            try_set(curr)
                            break
                    if improved:
                        break
            
            # Try removing
            if not improved and n_curr > 2:
                for i in range(n_curr):
                    if curr[i] == 0:
                        continue
                    cand2 = [x for j, x in enumerate(curr) if j != i]
                    s = compute_score(cand2)
                    if s > curr_score:
                        curr = cand2
                        curr_score = s
                        improved = True
                        no_improve = 0
                        try_set(curr)
                        break
            
            if improved:
                continue
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        
        return curr
    
    def simulated_annealing(cand, max_iters=500):
        """Simulated annealing for lattice gas model"""
        curr = sorted(set(cand))
        if not curr or 0 not in curr or len(curr) < 2:
            return curr
        curr_score = compute_score(curr)
        best_curr = list(curr)
        best_score_sa = curr_score
        
        T = 0.1  # Initial temperature
        cooling_rate = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            n_curr = len(curr)
            max_cur = max(curr) if curr else 0
            curr_set = set(curr)
            
            # Generate a move
            move_type = random.random()
            new_curr = list(curr)
            
            if move_type < 0.3 and n_curr < 15:
                # Add element
                new_el = random.randint(1, max_cur + 15)
                if new_el not in curr_set:
                    new_curr.append(new_el)
            elif move_type < 0.6 and n_curr > 2:
                # Remove element
                idx = random.randint(1, len(new_curr) - 1)
                new_curr.pop(idx)
            elif move_type < 0.85:
                # Replace element
                idx = random.randint(1, len(new_curr) - 1)
                new_el = random.randint(1, max_cur + 15)
                if new_el not in curr_set:
                    new_curr[idx] = new_el
            else:
                # Swap two elements
                if len(new_curr) >= 3:
                    i, j = random.sample(range(1, len(new_curr)), 2)
                    new_curr[i], new_curr[j] = new_curr[j], new_curr[i]
            
            new_curr = sorted(set(new_curr))
            if not new_curr or 0 not in new_curr or len(new_curr) < 2:
                continue
            
            new_score = compute_score(new_curr)
            delta = new_score - curr_score
            
            # Accept if better or with probability exp(-delta/T)
            if delta > 0 or random.random() < math.exp(-delta / max(T, 0.001)):
                curr = new_curr
                curr_score = new_score
                if curr_score > best_score_sa:
                    best_score_sa = curr_score
                    best_curr = list(curr)
                    try_set(best_curr)
            
            T *= cooling_rate
        
        return best_curr
    
    # Run local search from known sets
    for s in known_sets:
        if time.time() > deadline - 10:
            break
        result = local_search(s)
        try_set(result)
    
    # Run simulated annealing from best sets
    for s in known_sets[:5]:
        if time.time() > deadline - 15:
            break
        result = simulated_annealing(s, max_iters=200)
        try_set(result)
    
    # Genetic algorithm with base-B digit representation
    base = 16  # hex digits
    num_digits = 3  # 3 hex digits = up to 4095
    
    # Initialize population from known good sets
    population = []
    for s in known_sets[:8]:
        if time.time() > deadline - 10:
            break
        pop_entry = list(s)
        population.append(pop_entry)
    
    # Add some random starts with better structure
    for _ in range(6):
        if time.time() > deadline - 10:
            break
        n = random.randint(4, 10)
        max_m = random.randint(20, 80)
        # Generate structured random sets
        inner = []
        current = 1
        for _ in range(n - 1):
            inner.append(current)
            current += random.randint(1, 8)
            if current > max_m:
                break
        population.append(sorted([0] + inner))
    
    # Genetic algorithm loop
    gen = 0
    max_gens = 80
    while gen < max_gens and time.time() < deadline - 20:
        # Evaluate population
        scored = []
        for ind in population:
            ind = sorted(set(ind))
            if not ind or 0 not in ind or len(ind) < 2:
                scored.append((0, ind))
            else:
                s = compute_score(ind)
                scored.append((s, ind))
        
        scored.sort(key=lambda x: -x[0])
        
        # Elitism - keep top 3
        new_pop = [scored[i][1] for i in range(min(3, len(scored)))]
        
        # Crossover and mutation
        while len(new_pop) < len(population):
            if time.time() > deadline - 20:
                break
            # Select parents with tournament selection
            p1 = scored[random.randint(0, min(4, len(scored)-1))][1]
            p2 = scored[random.randint(0, min(4, len(scored)-1))][1]
            
            # Mutate a copy
            child = list(p1)
            if random.random() < 0.5:
                # Add element
                if len(child) < 15:
                    max_c = max(child) if child else 0
                    new_el = random.randint(1, max_c + 12)
                    if new_el not in child:
                        child.append(new_el)
                else:
                    # Replace
                    idx = random.randint(1, len(child)-1)
                    max_c = max(child) if child else 0
                    child[idx] = random.randint(1, max_c + 12)
            elif random.random() < 0.5:
                # Remove element
                if len(child) > 2:
                    idx = random.randint(1, len(child)-1)
                    child.pop(idx)
            else:
                # Replace element
                if len(child) >= 2:
                    idx = random.randint(1, len(child)-1)
                    max_c = max(child) if child else 0
                    child[idx] = random.randint(1, max_c + 12)
            
            child = sorted(set(child))
            if child and 0 in child and len(child) >= 2:
                new_pop.append(child)
            else:
                new_pop.append(p1[:])
        
        population = new_pop
        gen += 1
        
        # Run local search on best every few generations
        if gen % 3 == 0 and time.time() < deadline - 20:
            best_ind = scored[0][1]
            improved = local_search(best_ind, max_iters=80)
            try_set(improved)
        
        # Occasionally run simulated annealing
        if gen % 10 == 0 and time.time() < deadline - 25:
            best_ind = scored[0][1]
            improved = simulated_annealing(best_ind, max_iters=100)
            try_set(improved)
    
    # Final local search on best
    if time.time() < deadline - 5:
        improved = local_search(best_set, max_iters=400)
        try_set(improved)
    
    # Final simulated annealing polish
    if time.time() < deadline - 5:
        improved = simulated_annealing(best_set, max_iters=200)
        try_set(improved)
    
    return list(best_set)
