"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard: kill before 150s wall-clock limit
    deadline = time.time() + 140.0
    
    # Total number of differences per row
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    def get_row_diffs(row):
        """Return list of all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    def get_all_diffs(rows):
        """Return list of all differences across all rows."""
        diffs = []
        for row in rows:
            diffs.extend(get_row_diffs(row))
        return diffs
    
    def count_violations_from_diffs(diffs):
        """Count how many differences are duplicated."""
        if not diffs:
            return 0
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        return sum(c - 1 for c in counts.values() if c > 1)
    
    def is_valid(rows):
        diffs = get_all_diffs(rows)
        return len(diffs) == len(set(diffs))
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def compute_cost(rows):
        """Cost = violations * 10000 + max_entry"""
        diffs = get_all_diffs(rows)
        violations = count_violations_from_diffs(diffs)
        m = max_entry(rows)
        return violations * 10000 + m
    
    # --- Phase 1: Smart greedy initialization ---
    def greedy_init():
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
    
    best_rows = greedy_init()
    best_max = max_entry(best_rows)
    best_cost = compute_cost(best_rows)
    
    # --- Phase 2: Genetic Algorithm with bitmask-based difference tracking ---
    # Represent each row's difference set as a bitmask for fast collision detection
    # Fitness: minimize (violations, max_entry)
    
    def row_to_mask(row):
        """Convert a row to a bitmask of its differences."""
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                mask |= (1 << (row[j] - row[i]))
        return mask
    
    def masks_to_violations(masks):
        """Count violations from a list of bitmasks (one per row)."""
        # Combine all masks and count set bits vs total expected
        combined = 0
        for m in masks:
            combined |= m
        # Count set bits in combined
        set_bits = bin(combined).count('1')
        # Total expected distinct diffs
        total_expected = n_rows * num_diffs_per_row
        violations = total_expected - set_bits
        return violations
    
    def evaluate_population(pop):
        """Evaluate a population of rows. Returns list of (cost, rows, masks)."""
        results = []
        for rows in pop:
            masks = [row_to_mask(row) for row in rows]
            violations = masks_to_violations(masks)
            m = max_entry(rows)
            cost = violations * 10000 + m
            results.append((cost, rows, masks, violations, m))
        return results
    
    def crossover(parent1, parent2):
        """Crossover: for each row position, take from one parent or the other.
        Prefer taking rows that have fewer local conflicts."""
        child = []
        for i in range(n_rows):
            if random.random() < 0.5:
                child.append(parent1[i][:])
            else:
                child.append(parent2[i][:])
        return child
    
    def mutate_row(row, intensity=1.0):
        """Mutate a single row by perturbing its values."""
        new_row = row[:]
        for i in range(1, len(new_row)):
            if random.random() < 0.3 * intensity:
                # Perturb value
                lower = new_row[i-1] + 1
                upper = new_row[i+1] - 1 if i < len(new_row) - 1 else new_row[i] + 20
                if upper >= lower:
                    new_row[i] = random.randint(lower, upper)
        # Ensure monotonicity
        for i in range(1, len(new_row)):
            if new_row[i] <= new_row[i-1]:
                new_row[i] = new_row[i-1] + 1
        return new_row
    
    def mutate_individual(rows, rate=0.1):
        """Mutate an individual by mutating some of its rows."""
        new_rows = [row[:] for row in rows]
        for i in range(n_rows):
            if random.random() < rate:
                new_rows[i] = mutate_row(new_rows[i])
        return new_rows
    
    # Initialize population
    pop_size = min(50, max(10, 200 // n_rows))
    population = [best_rows[:]]
    for _ in range(pop_size - 1):
        mutated = mutate_individual(best_rows, rate=0.2)
        population.append(mutated)
    
    # Genetic algorithm loop
    ga_start = time.time()
    ga_time_budget = 60.0  # Spend up to 60s on GA
    
    while time.time() < deadline - 20:
        if time.time() - ga_start > ga_time_budget:
            break
        
        # Evaluate population
        evals = evaluate_population(population)
        evals.sort(key=lambda x: x[0])
        
        # Update best
        best_eval = evals[0]
        if best_eval[0] < best_cost:
            best_cost = best_eval[0]
            best_rows = [r[:] for r in best_eval[1]]
            best_max = best_eval[4]
        
        # Selection: keep top half
        elite_count = max(1, pop_size // 3)
        elites = [e[1] for e in evals[:elite_count]]
        
        # Create new population
        new_population = []
        # Elites
        for e in elites:
            new_population.append(e[:])
        
        # Fill rest with crossover + mutation
        while len(new_population) < pop_size:
            # Select two parents (tournament selection)
            p1_idx = random.randint(0, len(evals) - 1)
            p2_idx = random.randint(0, len(evals) - 1)
            # Bias towards better individuals
            if random.random() < 0.7:
                p1_idx = random.randint(0, elite_count)
            if random.random() < 0.7:
                p2_idx = random.randint(0, elite_count)
            
            parent1 = evals[p1_idx][1]
            parent2 = evals[p2_idx][1]
            
            child = crossover(parent1, parent2)
            child = mutate_individual(child, rate=0.15)
            new_population.append(child)
        
        population = new_population
    
    # --- Phase 3: Local search refinement on best individual ---
    current_rows = [r[:] for r in best_rows]
    current_cost = best_cost
    
    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
    
    T = 30.0
    T_min = 0.01
    cooling = 0.99995
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
        
        r, i = random.choice(indices)
        old_val = current_rows[r][i]
        
        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 30
        
        if upper < lower:
            continue
        
        # Choose new value with bias towards smaller values
        if random.random() < 0.4:
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.4)
        else:
            new_val = random.randint(lower, upper)
        
        if new_val == old_val:
            continue
        
        current_rows[r][i] = new_val
        
        # Check monotonicity
        row = current_rows[r]
        valid_mono = True
        for j in range(1, len(row)):
            if row[j] <= row[j - 1]:
                valid_mono = False
                break
        
        if not valid_mono:
            current_rows[r][i] = old_val
            continue
        
        new_cost = compute_cost(current_rows)
        
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            new_max = max_entry(current_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [rr[:] for rr in current_rows]
                best_cost = new_cost
        else:
            current_rows[r][i] = old_val
        
        T *= cooling
    
    # Final validation
    if is_valid(best_rows):
        return best_rows
    else:
        # Try to return the best valid solution found, or the best cost solution
        return best_rows
# EVOLVE-BLOCK-END
