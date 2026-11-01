import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Greedy Initialization
    # We need n_rows rows, each of length k+1, starting at 0.
    # Greedy: pick smallest available integers that don't create duplicate differences.
    
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def count_conflicts(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        c = Counter(all_diffs)
        conflicts = 0
        for d, cnt in c.items():
            if cnt > 1:
                conflicts += cnt - 1
        return conflicts

    def get_scope(rows):
        return max(max(r) for r in rows)

    # 2. Best-Fit Row Constructor
    # Given a set of used differences (from other rows), find the "best" row 
    # that minimizes its own max element while avoiding conflicts as much as possible.
    # Returns (row, conflicts_created, max_element)
    
    def construct_best_row(used_diffs_from_others, target_max=None):
        # target_max is an optional upper bound to force compactness
        # We use a randomized greedy with backtracking-lite (restarts)
        best_row = None
        best_score = float('inf')
        
        # If target_max is not provided, estimate a reasonable bound
        # Based on counting bound or previous best
        if target_max is None:
            # Heuristic: counting bound is around n_rows * (k+1)*k/2
            # Let's just let it grow, but penalize large values
            target_max = 200 
            
        # Try a few random seeds for the greedy process
        for _ in range(20):
            row = [0]
            current_max = 0
            valid = True
            conflicts_in_row = 0
            
            while len(row) < k + 1:
                # Try values from current_max+1 up to some limit
                # To make it "best fit", we prioritize small values but allow skipping
                # if it causes too many conflicts
                
                # Determine search range
                # If we are forced to be small (target_max), we might need to look further
                # but we want to minimize max, so we start small.
                
                found = False
                # Look for the smallest x > current_max such that x - a not in used_diffs_from_others
                # AND x - a not in current row's diffs (internal uniqueness is usually guaranteed by construction if we pick smallest? No, internal diffs must be distinct too)
                
                # Actually, the problem requires ALL diffs in the row to be distinct.
                # So we need x - a not in used_diffs_from_others AND x - a not in {row[j]-row[i] for i<j}.
                
                current_row_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        current_row_diffs.add(row[j]-row[i])
                
                x = current_max + 1
                limit = target_max if target_max else current_max + 50
                
                while x <= limit:
                    # Check against external used diffs
                    conflict_ext = False
                    # Check against internal diffs
                    conflict_int = False
                    
                    # Optimization: precompute row diffs? 
                    # For k=7, row has 8 elements, 28 diffs. Fast enough.
                    
                    for a in row:
                        d = x - a
                        if d in used_diffs_from_others:
                            conflict_ext = True
                            break
                        if d in current_row_diffs:
                            conflict_int = True
                            break
                    
                    if not conflict_ext and not conflict_int:
                        row.append(x)
                        current_row_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        found = True
                        break
                    x += 1
                
                if not found:
                    valid = False
                    break
            
            if valid:
                max_val = row[-1]
                # Score: primarily max_val, secondarily how many "close calls" or just random tie-break
                # Actually, if it's valid against external diffs, conflicts created is 0.
                # If we allow "soft" constraints (which we don't in final output), we'd count them.
                # Here we only accept rows that are valid against external diffs.
                # So we just want the one with smallest max_val.
                if max_val < best_score:
                    best_score = max_val
                    best_row = row[:]
        
        if best_row:
            return best_row
        return None

    # 3. Population-Based Genetic Search
    
    # Initial Population
    pop_size = 30
    population = []
    
    # Seed with greedy
    pop_rows = greedy_init()
    population.append(pop_rows)
    
    # Add some random perturbations to initial population
    for _ in range(pop_size - 1):
        # Take a copy and perturb
        new_pop = [r[:] for r in pop_rows]
        for _ in range(random.randint(2, 5)):
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)
            row = new_pop[ri]
            # Random jump
            lower = row[ei-1] + 1
            upper = row[ei+1] - 1 if ei < k else row[ei] + 10
            if lower < upper:
                new_val = random.randint(lower, upper)
                row[ei] = new_val
                # Ensure strict increase
                # If new_val <= row[ei-1] or new_val >= row[ei+1], shift neighbors? 
                # Simpler: just reject if invalid structure
                valid_struct = True
                for i in range(1, len(row)):
                    if row[i] <= row[i-1]:
                        valid_struct = False
                        break
                if not valid_struct:
                    row[ei] = random.randint(row[ei-1]+1, row[ei-1]+10)
                    # Fix subsequent
                    for j in range(ei+1, len(row)):
                        if row[j] <= row[j-1]:
                            row[j] = row[j-1] + 1
        population.append(new_pop)
    
    # Track best
    best_pop = None
    best_score = float('inf')
    best_conflicts = float('inf')
    
    def evaluate(pop):
        conf = count_conflicts(pop)
        scope = get_scope(pop)
        # Composite score: minimize conflicts first, then scope
        # If conf > 0, it's invalid, so we prioritize reducing conf.
        score = conf * 10000 + scope
        return conf, scope, score
    
    # Initial evaluation
    for pop in population:
        c, s, sc = evaluate(pop)
        if sc < best_score:
            best_score = sc
            best_conflicts = c
            best_pop = [r[:] for r in pop]
            
    # Genetic Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # 1. Selection: Keep top half
        scored_pop = []
        for pop in population:
            c, s, sc = evaluate(pop)
            scored_pop.append((sc, pop))
        
        scored_pop.sort(key=lambda x: x[0])
        
        # Keep top 50%
        survivors = [p for _, p in scored_pop[:pop_size//2]]
        
        # 2. Crossover: Create new population from survivors
        new_population = []
        
        # Keep the best few unchanged (elitism)
        for i in range(5):
            new_population.append([r[:] for r in survivors[i]])
            
        # Fill rest with crossovers and mutations
        while len(new_population) < pop_size:
            # Pick two parents
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            
            # Crossover: Randomly pick rows from p1 or p2
            child = []
            for i in range(n_rows):
                if random.random() < 0.5:
                    child.append(p1[i][:])
                else:
                    child.append(p2[i][:])
            
            # Mutation: Replace 1-2 rows with "Best Fit" rows
            num_mutations = random.randint(1, 2)
            for _ in range(num_mutations):
                ri = random.randint(0, n_rows - 1)
                
                # Calculate used diffs from OTHER rows in the child
                used_diffs_from_others = set()
                for j in range(n_rows):
                    if j == ri:
                        continue
                    used_diffs_from_others.update(get_diffs(child[j]))
                
                # Try to construct a better row for this position
                # We want a row that fits within the current scope or slightly less
                current_max_in_child = max(max(r) for r in child)
                # Try to find a row with max < current_max_in_child
                # Or just a valid row
                new_row = construct_best_row(used_diffs_from_others, target_max=current_max_in_child)
                
                if new_row:
                    child[ri] = new_row
                else:
                    # Fallback: Random small perturbation of existing row
                    row = child[ri]
                    ei = random.randint(1, k)
                    lower = row[ei-1] + 1
                    upper = row[ei+1] - 1 if ei < k else row[ei] + 5
                    if lower < upper:
                        row[ei] = random.randint(lower, upper)
                        # Fix structure
                        for j in range(ei+1, len(row)):
                            if row[j] <= row[j-1]:
                                row[j] = row[j-1] + 1
            
            new_population.append(child)
        
        population = new_population
        
        # 3. Update Best
        for pop in population:
            c, s, sc = evaluate(pop)
            if sc < best_score:
                best_score = sc
                best_conflicts = c
                best_pop = [r[:] for r in pop]
                
        # If we found a valid solution (conflicts == 0), we can tighten the scope target
        if best_conflicts == 0:
             # Optional: If we have time, do a local search on the best_pop to shave off more
             # But the GA loop should handle this via the "Best Fit" constructor trying to minimize max
             pass

    return best_pop
