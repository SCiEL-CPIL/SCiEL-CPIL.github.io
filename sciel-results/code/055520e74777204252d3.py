import time
import random
import sys
from itertools import combinations

def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i):
            diffs.add(row[i] - row[j])
    return diffs

def _check_valid(rows, k):
    """Verify that all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for d in _row_diffs(row):
            if d in all_diffs:
                return False
            all_diffs.add(d)
    return True

def _count_diffs(rows):
    """Count total number of differences."""
    total = 0
    for row in rows:
        total += len(_row_diffs(row))
    return total

def _greedy_construct(n_rows, k, M, rng=None, strategy='random'):
    """
    Greedy construction with randomization.
    Tries to build n_rows rows with max entry <= M.
    """
    if rng is None:
        rng = random.Random()
    
    used_diffs = set()
    rows = []
    
    for r_idx in range(n_rows):
        row = [0]
        row_diffs = set()
        success = False
        
        for attempt in range(100):
            row = [0]
            row_diffs = set()
            success = False
            
            for pos in range(1, k + 1):
                last_val = row[-1]
                
                if strategy == 'random':
                    candidates = list(range(last_val + 1, M + 1))
                    rng.shuffle(candidates)
                elif strategy == 'small':
                    candidates = list(range(last_val + 1, M + 1))
                elif strategy == 'large':
                    candidates = list(range(M, last_val, -1))
                else:
                    candidates = list(range(last_val + 1, M + 1))
                    rng.shuffle(candidates)
                
                if not candidates:
                    break
                
                found = False
                for x in candidates:
                    valid = True
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in row_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    # Pruning: check if enough diffs remain
                    used_count = len(used_diffs) + len(row_diffs)
                    remaining_needed = n_rows * k * (k + 1) // 2 - used_count
                    available = M - used_count
                    if available < remaining_needed:
                        continue
                    
                    row.append(x)
                    row_diffs |= new_diffs
                    found = True
                    break
                
                if not found:
                    break
            
            if success:
                break
        
        if not success:
            return None
        
        used_diffs |= row_diffs
        rows.append(row)
    
    return rows

def _genetic_algorithm(n_rows, k, M, deadline, rng=None):
    """
    Genetic algorithm for constructing difference triangle sets.
    """
    if rng is None:
        rng = random.Random()
    
    population_size = 20
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    def fitness(rows):
        """
        Fitness: penalize difference collisions and large scope.
        Lower is better.
        """
        if rows is None:
            return float('inf')
        
        all_diffs = set()
        collisions = 0
        max_entry = 0
        
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return float('inf')
            for i in range(len(row)):
                if i > 0 and row[i] <= row[i-1]:
                    return float('inf')
            max_entry = max(max_entry, row[-1])
            for d in _row_diffs(row):
                if d in all_diffs:
                    collisions += 1
                all_diffs.add(d)
        
        # Penalize collisions heavily, then scope
        return collisions * 10000 + max_entry
    
    def create_individual():
        """Create a random individual."""
        rows = []
        used_diffs = set()
        
        for _ in range(n_rows):
            row = [0]
            row_diffs = set()
            
            for pos in range(1, k + 1):
                last_val = row[-1]
                candidates = list(range(last_val + 1, M + 1))
                rng.shuffle(candidates)
                
                found = False
                for x in candidates:
                    valid = True
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in row_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    row.append(x)
                    row_diffs |= new_diffs
                    found = True
                    break
                
                if not found:
                    # If we can't find a valid x, use a large value
                    x = M
                    row.append(x)
                    break
            
            rows.append(row)
            used_diffs |= _row_diffs(row)
        
        return rows
    
    def crossover(parent1, parent2):
        """Crossover: recombine rows from parents."""
        child = []
        for i in range(n_rows):
            if rng.random() < 0.5:
                child.append(parent1[i][:])
            else:
                child.append(parent2[i][:])
        return child
    
    def mutate(individual):
        """Mutation: shift elements to resolve conflicts."""
        rows = [row[:] for row in individual]
        rng.shuffle(rows)
        
        for row_idx in range(n_rows):
            row = rows[row_idx]
            for pos in range(1, len(row)):
                if rng.random() < 0.3:
                    # Try to shift this element
                    old_val = row[pos]
                    shift = rng.randint(-3, 3)
                    new_val = old_val + shift
                    
                    if new_val > row[pos-1] and new_val <= M:
                        # Check if this creates conflicts
                        test_row = row[:]
                        test_row[pos] = new_val
                        
                        # Simple validity check for this row
                        valid = True
                        for i in range(len(test_row)):
                            for j in range(i):
                                d = test_row[i] - test_row[j]
                                if d <= 0:
                                    valid = False
                                    break
                        if valid:
                            row[pos] = new_val
        
        return rows
    
    # Initialize population
    population = []
    for _ in range(population_size):
        population.append(create_individual())
    
    best_individual = min(population, key=fitness)
    best_fitness = fitness(best_individual)
    
    # Evolution loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by fitness
        population.sort(key=fitness)
        
        # Keep best 50%
        elite_count = population_size // 2
        new_population = [ind[:] for ind in population[:elite_count]]
        
        # Generate offspring
        while len(new_population) < population_size and time.time() < deadline:
            # Select parents (tournament selection)
            p1 = rng.choice(population[:elite_count])
            p2 = rng.choice(population[:elite_count])
            
            child = crossover(p1, p2)
            child = mutate(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Update best
        for ind in population:
            f = fitness(ind)
            if f < best_fitness:
                best_fitness = f
                best_individual = ind[:]
        
        # If we found a valid solution, return it
        if best_fitness < 10000:  # No collisions
            max_entry = max(max(row) for row in best_individual)
            if max_entry <= M:
                return best_individual
    
    return best_individual

def _local_search(rows, n_rows, k, M, deadline, rng=None):
    """
    Local search to improve a valid or near-valid solution.
    """
    if rng is None:
        rng = random.Random()
    
    if rows is None:
        return None
    
    rows = [row[:] for row in rows]
    
    def fitness(rows):
        all_diffs = set()
        collisions = 0
        max_entry = 0
        
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return float('inf')
            for i in range(len(row)):
                if i > 0 and row[i] <= row[i-1]:
                    return float('inf')
            max_entry = max(max_entry, row[-1])
            for d in _row_diffs(row):
                if d in all_diffs:
                    collisions += 1
                all_diffs.add(d)
        
        return collisions * 10000 + max_entry
    
    current_fitness = fitness(rows)
    best_rows = rows[:]
    best_fitness = current_fitness
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        # Pick a random row and position
        row_idx = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        
        old_val = rows[row_idx][pos]
        
        # Try nearby values
        for shift in range(-5, 6):
            if shift == 0:
                continue
            
            new_val = old_val + shift
            
            if new_val <= rows[row_idx][pos-1] or new_val > M:
                continue
            
            if pos < k:
                if new_val >= rows[row_idx][pos+1]:
                    continue
            
            # Test this change
            test_rows = [r[:] for r in rows]
            test_rows[row_idx][pos] = new_val
            
            new_fitness = fitness(test_rows)
            
            if new_fitness < current_fitness:
                rows = test_rows
                current_fitness = new_fitness
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_rows = rows[:]
                break
    
    if best_fitness < 10000:  # Valid solution
        max_entry = max(max(row) for row in best_rows)
        if max_entry <= M:
            return best_rows
    
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Theoretical lower bound
    M_lower = max(total_diffs_needed, 112)
    M_upper = 135
    
    best_solution = None
    best_M = float('inf')
    
    rng = random.Random(42)
    
    # Try M from lower bound to known best + margin
    for M in range(M_lower, M_upper + 1):
        if time.time() > deadline:
            break
        
        # Strategy 1: Multiple greedy attempts with different strategies
        for strategy in ['random', 'small', 'large']:
            for attempt in range(30):
                if time.time() > deadline:
                    break
                
                solution = _greedy_construct(n_rows, k, M, rng, strategy)
                if solution and _check_valid(solution, k):
                    max_entry = max(max(row) for row in solution)
                    if max_entry <= M:
                        best_solution = solution
                        best_M = max_entry
                        return solution
        
        # Strategy 2: Genetic algorithm
        if time.time() < deadline - 10.0:
            solution = _genetic_algorithm(n_rows, k, M, deadline, rng)
            if solution and _check_valid(solution, k):
                max_entry = max(max(row) for row in solution)
                if max_entry <= M:
                    best_solution = solution
                    best_M = max_entry
                    return solution
        
        # Strategy 3: Local search from a near-valid solution
        if time.time() < deadline - 5.0:
            # Create a near-valid solution
            rows = []
            used_diffs = set()
            for _ in range(n_rows):
                row = [0]
                row_diffs = set()
                for pos in range(1, k + 1):
                    last_val = row[-1]
                    candidates = list(range(last_val + 1, M + 1))
                    rng.shuffle(candidates)
                    
                    found = False
                    for x in candidates:
                        valid = True
                        new_diffs = set()
                        for a in row:
                            d = x - a
                            if d in used_diffs or d in row_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        
                        if not valid:
                            continue
                        
                        row.append(x)
                        row_diffs |= new_diffs
                        found = True
                        break
                    
                    if not found:
                        x = M
                        row.append(x)
                        break
                
                rows.append(row)
                used_diffs |= _row_diffs(row)
            
            # Local search
            solution = _local_search(rows, n_rows, k, M, deadline, rng)
            if solution and _check_valid(solution, k):
                max_entry = max(max(row) for row in solution)
                if max_entry <= M:
                    best_solution = solution
                    best_M = max_entry
                    return solution
    
    # If we found a solution with a larger M, return it
    if best_solution:
        return best_solution
    
    # Fallback: simple greedy without M constraint
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
