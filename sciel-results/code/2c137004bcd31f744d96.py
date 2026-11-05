import time
import random
import heapq
import bisect
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 140  # 10s margin from 150s limit
    
    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _all_diffs(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used
    
    def _valid(rows):
        all_d = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            rd = _row_diffs(row)
            if rd & all_d:
                return False
            all_d |= rd
        return True
    
    def _scope(rows):
        return max(max(r) for r in rows)
    
    def greedy_construct(seed_offset=0):
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        
        return rows
    
    def greedy_construct_heuristic(seed_offset=0):
        """Greedy with heuristic: prefer candidates that use 'large' differences (low 1/d sum)
        to preserve small differences for future tight packing."""
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                best_x = None
                best_cost = float('inf')
                best_new_diffs = None
                
                # Search up to a reasonable bound
                search_limit = x + 50
                for candidate in range(x + 1, search_limit + 1):
                    new_diffs = {candidate - a for a in row}
                    if new_diffs & used:
                        continue
                    
                    # Heuristic: cost = sum(1/d) + small penalty for large x
                    cost = sum(1.0 / d for d in new_diffs) + 0.01 * candidate
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_x = candidate
                        best_new_diffs = new_diffs
                
                if best_x is None:
                    # Fallback: just find the first valid x
                    x += 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            best_x = x
                            best_new_diffs = new_diffs
                            break
                        x += 1
                else:
                    x = best_x
                    new_diffs = best_new_diffs
                
                row.append(x)
                used |= new_diffs
            
            rows.append(row)
        
        return rows
    
    def backtrack_row(used, max_val, row, depth, rng, deadline, heuristic_mode='degree'):
        """Build one row via backtracking within max_val.
        heuristic_mode: 'degree' uses 1/d cost, 'random' shuffles, 'first' takes first valid."""
        if time.time() > deadline:
            return None
        if depth == k + 1:
            return row[:]
        
        last = row[-1] if row else 0
        start = last + 1
        
        candidates = []
        for x in range(start, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            
            if heuristic_mode == 'degree':
                # Prefer candidates that use large differences (low 1/d sum)
                cost = sum(1.0 / d for d in new_diffs)
            elif heuristic_mode == 'random':
                cost = rng.random()
            else:  # 'first'
                cost = x
            
            candidates.append((cost, x, new_diffs))
            
            # Limit candidates to avoid explosion
            if len(candidates) > 80:
                break
        
        if not candidates:
            return None
        
        if heuristic_mode == 'degree':
            candidates.sort(key=lambda item: (item[0], item[1]))
        elif heuristic_mode == 'random':
            rng.shuffle(candidates)
        else:
            candidates.sort(key=lambda item: item[1])
        
        for cost, x, new_diffs in candidates:
            row.append(x)
            used_copy = used | new_diffs
            
            result = backtrack_row(used_copy, max_val, row, depth + 1, rng, deadline, heuristic_mode)
            if result is not None:
                return result
            
            row.pop()
        
        return None
    
    def backtrack_construct(max_val, num_rows, rng, deadline, heuristic_mode='degree'):
        """Try to construct all rows within max_val using backtracking."""
        used = set()
        rows = []
        
        for i in range(num_rows):
            if time.time() > deadline:
                return None
            
            row = [0]
            result = backtrack_row(used, max_val, row, 1, rng, deadline, heuristic_mode)
            if result is None:
                return None
            
            rows.append(result)
            used |= _row_diffs(result)
        
        return rows
    
    def local_search_shrink(rows, deadline, rng):
        """Local search: try to reduce the maximum entry by modifying individual elements."""
        rows = [list(r) for r in rows]
        best_scope = _scope(rows)
        used = _all_diffs(rows)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_scope = _scope(rows)
            
            for ri in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    if old_val < current_scope:
                        continue
                    
                    low = row[pos - 1] + 1
                    high = old_val - 1
                    if pos + 1 < len(row):
                        high = min(high, row[pos + 1] - 1)
                    
                    if low > high:
                        continue
                    
                    candidates = list(range(high, low - 1, -1))
                    rng.shuffle(candidates)
                    
                    for new_val in candidates:
                        if time.time() > deadline:
                            break
                        if new_val >= old_val:
                            continue
                        
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for i in range(pos + 1, len(row)):
                            old_diffs.add(row[i] - old_val)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        
                        other_diffs = used - old_diffs
                        conflict = new_diffs & other_diffs
                        if conflict:
                            continue
                        
                        row[pos] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        break
        
        return rows
    
    def genetic_algorithm(rows, deadline, rng, max_generations=200):
        """Genetic algorithm with row-frame crossover and shift mutation."""
        def repair(row, used, deadline):
            """Repair a row by greedily fixing conflicts."""
            row = list(row)
            if len(row) == 0:
                return row
            if row[0] != 0:
                row[0] = 0
            
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    row[i] = row[i-1] + 1
                
                # Check for conflicts and fix
                attempts = 0
                while attempts < 200:
                    new_diffs = set()
                    for j in range(i):
                        new_diffs.add(row[i] - row[j])
                    for j in range(i + 1, len(row)):
                        new_diffs.add(row[j] - row[i])
                    
                    if not (new_diffs & used):
                        break
                    
                    row[i] += 1
                    attempts += 1
                
                # Update used
                for j in range(i):
                    used.add(row[i] - row[j])
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
            
            return row
        
        # Initialize population
        pop_size = 30
        population = []
        for i in range(pop_size):
            if i == 0:
                pop = [list(r) for r in rows]
            else:
                # Create variations
                base = [list(r) for r in rows]
                for r_idx in range(len(base)):
                    for pos in range(1, len(base[r_idx])):
                        if rng.random() < 0.3:
                            shift = rng.randint(-5, 5)
                            old_val = base[r_idx][pos]
                            new_val = old_val + shift
                            if new_val > base[r_idx][pos-1] and (pos == len(base[r_idx])-1 or new_val < base[r_idx][pos+1]):
                                base[r_idx][pos] = new_val
                pop = base
            population.append(pop)
        
        best_individual = [list(r) for r in rows]
        best_scope = _scope(best_individual)
        
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            
            # Evaluate fitness
            def fitness(ind):
                if not _valid(ind):
                    return float('inf')
                return _scope(ind)
            
            # Sort by fitness
            population.sort(key=fitness)
            
            # Update best
            if fitness(population[0]) < best_scope:
                best_scope = fitness(population[0])
                best_individual = [list(r) for r in population[0]]
            
            # Elitism: keep top 2
            new_population = [list(r) for r in population[0]]
            new_population.append([list(r) for r in population[1]])
            
            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() > deadline:
                    break
                
                # Crossover: swap entire row frames between two parents
                p1_idx = rng.randint(0, min(5, len(population)-1))
                p2_idx = rng.randint(0, min(5, len(population)-1))
                p1 = [list(r) for r in population[p1_idx]]
                p2 = [list(r) for r in population[p2_idx]]
                
                child = []
                for i in range(len(p1)):
                    if rng.random() < 0.5:
                        child.append(list(p1[i]))
                    else:
                        child.append(list(p2[i]))
                
                # Mutation: small integer shifts to individual elements
                for r_idx in range(len(child)):
                    for pos in range(1, len(child[r_idx])):
                        if rng.random() < 0.2:
                            shift = rng.randint(-3, 3)
                            old_val = child[r_idx][pos]
                            new_val = old_val + shift
                            if new_val > child[r_idx][pos-1] and (pos == len(child[r_idx])-1 or new_val < child[r_idx][pos+1]):
                                child[r_idx][pos] = new_val
                
                # Repair
                used = set()
                for i in range(len(child)):
                    child[i] = repair(child[i], used, deadline)
                
                new_population.append(child)
            
            population = new_population
        
        return best_individual, best_scope
    
    # Start with greedy constructions
    best_rows = greedy_construct(0)
    best_scope = _scope(best_rows)
    
    # Try heuristic greedy
    for seed in range(10):
        if time.time() > deadline - 40:
            break
        rows = greedy_construct_heuristic(seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
    
    # Try more standard greedy variants
    for seed in range(10):
        if time.time() > deadline - 40:
            break
        rows = greedy_construct(seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
    
    # Apply local search shrink
    rng = random.Random(123)
    if time.time() < deadline - 25:
        polished = local_search_shrink(best_rows, deadline - 20, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Genetic algorithm
    if time.time() < deadline - 20:
        ga_rows, ga_scope = genetic_algorithm(best_rows, deadline - 15, rng, max_generations=150)
        if ga_scope < best_scope:
            best_rows = ga_rows
            best_scope = ga_scope
    
    # Try backtracking at progressively lower scopes with different heuristics
    current_max = best_scope
    lower_bound = max(n_rows * k * (k + 1) // 2, 50)
    
    heuristic_modes = ['degree', 'random', 'first']
    for mode in heuristic_modes:
        if time.time() > deadline - 10:
            break
        
        target_scope = current_max - 1
        while time.time() < deadline - 10 and target_scope >= lower_bound:
            rng = random.Random(int(time.time() * 1000) % 10000)
            result = backtrack_construct(target_scope, n_rows, rng, deadline - 10, mode)
            if result is not None:
                best_rows = result
                best_scope = target_scope
                current_max = target_scope
                break
            else:
                target_scope -= 1
    
    # Final polish
    if time.time() < deadline - 5:
        rng = random.Random(999)
        polished = local_search_shrink(best_rows, deadline - 5, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    return best_rows
