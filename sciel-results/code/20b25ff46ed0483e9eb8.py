"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    # Problem Parameters
    num_vars = n_rows * (k + 1)
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Helper: Calculate all differences for a single row
    def get_row_diffs(row):
        diffs = []
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                diffs.append(row[j] - row[i])
        return diffs

    # Helper: Calculate all differences for the entire set of rows
    def get_all_diffs(rows):
        diffs = []
        for r in rows:
            diffs.extend(get_row_diffs(r))
        return diffs

    # Helper: Check validity and get score
    # Returns (is_valid, score)
    def evaluate(rows):
        # Check row validity (strictly increasing)
        for r in rows:
            if r[0] != 0:
                return False, float('inf')
            for i in range(1, k + 1):
                if r[i] <= r[i-1]:
                    return False, float('inf')
        
        # Check difference uniqueness
        diffs = get_all_diffs(rows)
        if len(set(diffs)) != len(diffs):
            return False, float('inf')
        
        score = max(max(r) for r in rows)
        return True, score

    # Helper: Fitness for GA (lower is better)
    # Penalty for collisions + penalty for scope
    def fitness(rows):
        # Check row validity first
        for r in rows:
            if r[0] != 0:
                return 1e10
            for i in range(1, k + 1):
                if r[i] <= r[i-1]:
                    return 1e10
        
        diffs = get_all_diffs(rows)
        unique_diffs = set(diffs)
        collisions = len(diffs) - len(unique_diffs)
        
        scope = max(max(r) for r in rows)
        
        # Weighting: collisions are fatal, scope is the goal
        # If valid, fitness is just scope.
        # If invalid, fitness is huge + collisions.
        if collisions == 0:
            return scope
        else:
            return 10000 + collisions * 100 + scope

    # 1. Initialization: Generate diverse population
    # Strategy: Mix of greedy starts and random jittered starts
    population = []
    pop_size = 20
    
    # Add a few greedy solutions
    for _ in range(5):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        population.append(rows)
        
    # Add random valid-ish starts (repair if needed)
    while len(population) < pop_size:
        # Try to build a random set
        rows = []
        valid = True
        used = set()
        for _ in range(n_rows):
            row = [0]
            # Random start with some structure
            # Try to pick increasing integers
            # We will just use a random walk that respects increasing order
            # and then repair collisions later in mutation? 
            # For init, let's use a biased random greedy
            current_max = 0
            for _ in range(k):
                # Pick a random increment
                # Bias towards small increments to keep scope low
                inc = random.randint(1, 10)
                next_val = current_max + inc
                new_diffs = {next_val - a for a in row}
                if new_diffs & used:
                    # Collision, try smaller or larger?
                    # Just retry with a different random increment
                    # For init, if we fail too many times, fallback to greedy
                    pass 
                # To ensure we get valid starts for init, let's just use greedy with random jitter
                # Actually, let's just use the greedy solution and mutate it heavily for the rest
                break 
            
            # Fallback: Use the standard greedy row but add random noise to non-zero elements
            # This is risky for validity. 
            # Better: Use the greedy solution, then apply random "shakes" that maintain validity
            # But for init, let's just duplicate the greedy solution with slight modifications
            # Or use a different greedy seed?
            
            # Let's just use the greedy solution for the first few, and then
            # generate random valid rows by backtracking? Too slow.
            
            # Simplest valid random start:
            # Pick k random distinct integers from 1..200, sort them, prepend 0.
            # This will likely have collisions.
            # The fitness function handles collisions.
            
            # Let's generate a random row
            candidates = random.sample(range(1, 250), k)
            candidates.sort()
            row = [0] + candidates
            rows.append(row)
        
        # Check if this random set is valid?
        # If not, it's still in the population, but with high fitness.
        # This allows the GA to explore invalid space.
        population.append(rows)

    # Ensure we have at least one valid solution in population
    # If not, force one
    valid_exists = False
    for ind in population:
        is_val, _ = evaluate(ind)
        if is_val:
            valid_exists = True
            break
            
    if not valid_exists:
        # Add a standard greedy solution
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        population[0] = rows

    best_individual = None
    best_fitness = float('inf')
    
    # Evaluate initial population
    for ind in population:
        f = fitness(ind)
        if f < best_fitness:
            best_fitness = f
            best_individual = [r[:] for r in ind]

    # 2. Genetic Algorithm Loop
    # Parameters
    mutation_rate = 0.3
    elite_size = 2
    
    while time.time() < deadline:
        # Selection: Tournament Selection
        # Fitness is to be minimized
        def select():
            a = random.choice(population)
            b = random.choice(population)
            fa = fitness(a)
            fb = fitness(b)
            if fa < fb:
                return a
            else:
                return b

        # Crossover: Uniform crossover on the flattened list of values
        # But we must ensure rows are valid (increasing).
        # So we do row-wise crossover: For each row, pick from parent A or Parent B.
        # This maintains row validity if parents are valid.
        # If parents are invalid, children might be too, but fitness handles it.
        
        new_population = []
        
        # Elitism: Keep best individuals
        # Sort by fitness
        # Since fitness calculation is expensive, we might cache or just re-calc
        # For small pop, re-calc is fine.
        
        for _ in range(pop_size):
            p1 = select()
            p2 = select()
            
            # Crossover
            child = []
            for i in range(n_rows):
                if random.random() < 0.5:
                    child.append(p1[i][:])
                else:
                    child.append(p2[i][:])
            
            # Mutation
            # Randomly change a value in a row
            # We need to maintain increasing order.
            # Strategy: Pick a row, pick an index, change value.
            # If it breaks increasing order, repair by shifting subsequent values?
            # Or just reject mutation if it breaks order?
            # Better: Mutation picks a value and tries to move it.
            # If it violates order, we clamp it.
            
            for r_idx in range(n_rows):
                if random.random() < mutation_rate:
                    # Pick a position 1..k
                    pos = random.randint(1, k)
                    # Current value
                    old_val = child[r_idx][pos]
                    # New value: random walk
                    step = random.randint(-5, 5)
                    new_val = old_val + step
                    
                    # Check bounds
                    if pos > 0:
                        if new_val <= child[r_idx][pos-1]:
                            new_val = child[r_idx][pos-1] + 1
                    if pos < k:
                        if new_val >= child[r_idx][pos+1]:
                            new_val = child[r_idx][pos+1] - 1
                    
                    if new_val > 0:
                        child[r_idx][pos] = new_val
                        # Repair subsequent values if necessary to maintain strict increase
                        # If we increased a value, subsequent ones might be too small
                        # If we decreased, previous ones might be too large (handled by clamp above? No, clamp only checks immediate neighbors)
                        # Let's do a forward pass repair
                        for j in range(pos + 1, k + 1):
                            if child[r_idx][j] <= child[r_idx][j-1]:
                                child[r_idx][j] = child[r_idx][j-1] + 1
                        # Backward pass repair
                        for j in range(pos - 1, 0, -1):
                            if child[r_idx][j] >= child[r_idx][j+1]:
                                child[r_idx][j] = child[r_idx][j+1] - 1
                                if child[r_idx][j] < 0:
                                    child[r_idx][j] = 0 # Should not happen if pos>0 and we clamp
                                    # Actually if j=0, it's fixed at 0. If j>0, it must be >0.
                                    # If j=1 and it becomes 0, that's invalid.
                                    # So we must ensure j>0 implies val > 0.
                                    # The backward pass might push val 1 to 0.
                                    # So we should not allow decreasing below 1 for j>=1.
                                    # Let's add a floor of 1 for j>=1 in backward pass?
                                    # Or just reject if it becomes 0.
                                    pass

            new_population.append(child)
            
            # Update best
            f = fitness(child)
            if f < best_fitness:
                best_fitness = f
                best_individual = [r[:] for r in child]

        # Next generation: Elites + New population
        # Sort new_population by fitness to keep best?
        # Or just replace non-elites?
        # Let's keep the top 'elite_size' from the current population (before replacement)
        # and fill the rest with new_population.
        
        # We need to know which were the elites in the *current* population.
        # Let's just re-sort the entire new_population + elites?
        # Simpler: 
        # 1. Identify elites from current population (we didn't sort it explicitly, so let's find top 2)
        # Actually, `best_individual` is the global best.
        # Let's just insert the global best into the new population to ensure it's not lost.
        new_population.append([r[:] for r in best_individual])
        
        # If new_population is too big, truncate?
        # Or just take random samples?
        # Let's take the best half of new_population and the elites.
        # To avoid expensive sorting, we'll just random sample from new_population
        # and ensure best_individual is present.
        
        # Actually, a simple replacement strategy:
        # population = new_population[:pop_size]
        # But we must ensure best_individual is in there.
        # Since we appended it, it's at the end.
        # Let's just shuffle and take pop_size.
        random.shuffle(new_population)
        population = new_population[:pop_size]
        
        # Ensure best_individual is still in population?
        # It might have been cut off.
        # Let's force it in.
        if best_individual not in population:
            # Replace a random one
            idx = random.randint(0, pop_size - 1)
            population[idx] = [r[:] for r in best_individual]

    # Final check: return the best valid solution
    # If best_individual is invalid (fitness > 10000), we should try to repair it or return the best valid one found.
    # But best_fitness is tracked. If best_fitness < 10000, it's valid.
    # If not, we have a problem.
    # Let's do a final greedy repair on best_individual if it's invalid.
    
    is_val, score = evaluate(best_individual)
    if not is_val:
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_individual = rows
        score = max(max(r) for r in best_individual)
        
    return best_individual
# EVOLVE-BLOCK-END
