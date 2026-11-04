import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy: Multi-Objective Evolutionary Algorithm (MOEA)
    - Population of candidate solutions (list of rows)
    - Crossover: Merges partial row constructions from two parents
    - Mutation: 'Difference-swap' moves to resolve conflicts and shrink scope
    - Selection: Tournament selection based on scope (lower is better) and validity
    """
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill
    
    num_elements = k + 1
    total_diffs_per_row = num_elements * (num_elements - 1) // 2
    
    # Helper functions
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def get_all_diffs(rows):
        all_diffs = set()
        for r in rows:
            all_diffs.update(get_row_diffs(r))
        return all_diffs

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        return len(all_diffs) == len(set(all_diffs))

    def scope(rows):
        return max(max(r) for r in rows) if rows else float('inf')

    def greedy_construct(max_scope, used_diffs=None):
        """
        Greedily construct a valid row starting with 0, using smallest possible values
        that don't conflict with used_diffs.
        Returns None if no valid row can be constructed within max_scope.
        """
        if used_diffs is None:
            used_diffs = set()
            
        row = [0]
        current_max = 0
        while len(row) < num_elements:
            found = False
            for v in range(current_max + 1, max_scope + 1):
                # Check if adding v creates any duplicate diff
                valid = True
                for x in row:
                    d = v - x
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    # Check internal consistency
                    # v-x are distinct because x are distinct and v is fixed
                    # But we must ensure v-x is not in used_diffs
                    row.append(v)
                    for x in row[:-1]:
                        used_diffs.add(v - x)
                    current_max = v
                    found = True
                    break
            if not found:
                return None
        return row

    def random_valid_row(max_scope, used_diffs=None):
        """
        Randomly construct a valid row starting with 0, using random values
        that don't conflict with used_diffs.
        """
        if used_diffs is None:
            used_diffs = set()
            
        row = [0]
        current_max = 0
        while len(row) < num_elements:
            # Try random values in increasing order to maintain strictly increasing property
            # But we need to ensure they are > current_max
            # To make it more random, we can pick a random target range
            candidates = []
            # We need at least (num_elements - len(row)) - 1 more steps
            min_next = current_max + 1
            max_next = max_scope - (num_elements - len(row) - 1)
            
            if min_next > max_next:
                return None
                
            # Try a few random values
            attempts = 0
            while attempts < 100:
                v = random.randint(min_next, max_next)
                valid = True
                for x in row:
                    d = v - x
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(v)
                    for x in row[:-1]:
                        used_diffs.add(v - x)
                    current_max = v
                    break
                attempts += 1
            
            if attempts >= 100:
                # Fallback to greedy for this step
                found = False
                for v in range(current_max + 1, max_next + 1):
                    valid = True
                    for x in row:
                        d = v - x
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        for x in row[:-1]:
                            used_diffs.add(v - x)
                        current_max = v
                        found = True
                        break
                if not found:
                    return None
        return row

    def crossover(parent1, parent2):
        """
        Structural crossover: For each row index, take the row from the better parent
        (based on the row's max element or validity contribution).
        Alternatively, mix elements within rows.
        Here, we do a simple row-wise crossover: for each row, pick from parent1 or parent2.
        """
        child = []
        for i in range(n_rows):
            if random.random() < 0.5:
                child.append(list(parent1[i]))
            else:
                child.append(list(parent2[i]))
        return child

    def mutate(rows, max_scope):
        """
        Difference-swap mutation:
        1. Identify conflicts (duplicate differences).
        2. Try to adjust elements to resolve conflicts and reduce scope.
        """
        # Calculate current differences
        all_diffs = {}
        for r_idx, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d not in all_diffs:
                        all_diffs[d] = []
                    all_diffs[d].append((r_idx, i, j))
        
        # Find conflicts
        conflicts = [d for d, locs in all_diffs.items() if len(locs) > 1]
        
        # If no conflicts, try to shrink scope
        if not conflicts:
            # Try to reduce the largest element in each row
            for r_idx in range(n_rows):
                row = rows[r_idx]
                # Try to reduce the last element
                last_val = row[-1]
                for new_val in range(last_val - 1, 0, -1):
                    # Check if new_val > row[-2]
                    if new_val <= row[-2]:
                        break
                    # Check if changing last_val to new_val creates conflicts
                    # Remove old diffs involving last_val
                    old_diffs = set()
                    for i in range(len(row)-1):
                        old_diffs.add(last_val - row[i])
                    
                    # Check new diffs
                    new_diffs = set()
                    for i in range(len(row)-1):
                        new_diffs.add(new_val - row[i])
                    
                    # Check if new_diffs are unique in the whole set
                    # Temporarily remove old_diffs from all_diffs
                    temp_diffs = {}
                    for d, locs in all_diffs.items():
                        if d in old_diffs:
                            # Filter out locs involving this row and last index
                            new_locs = [loc for loc in locs if not (loc[0] == r_idx and (loc[1] == len(row)-1 or loc[2] == len(row)-1))]
                            if new_locs:
                                temp_diffs[d] = new_locs
                        else:
                            temp_diffs[d] = locs
                    
                    # Add new_diffs
                    valid = True
                    for d in new_diffs:
                        if d in temp_diffs:
                            valid = False
                            break
                        temp_diffs[d] = [(r_idx, i, len(row)-1) for i in range(len(row)-1)]
                    
                    if valid:
                        row[-1] = new_val
                        # Update all_diffs
                        # This is a simplified update, might need full recalculation for accuracy
                        # For now, just update the row and let the next iteration handle it
                        break
            
            return rows
        
        # If conflicts exist, try to resolve them by moving elements
        for d in conflicts:
            locs = all_diffs[d]
            # Pick one location to move
            r_idx, i, j = random.choice(locs)
            row = rows[r_idx]
            
            # Try to move element j to a new value
            old_val = row[j]
            # Try smaller values first to reduce scope
            for new_val in range(row[j-1] + 1, old_val):
                # Check if new_val creates new conflicts
                # This is a simplified check
                valid = True
                for x_idx in range(j):
                    new_d = new_val - row[x_idx]
                    if new_d in all_diffs and new_d != d:
                        # Check if it conflicts with other locations
                        # Simplified: just check if it's already used
                        valid = False
                        break
                
                if valid:
                    # Perform the move
                    row[j] = new_val
                    # Recalculate all_diffs (expensive but necessary for accuracy)
                    all_diffs = {}
                    for r_idx2, row2 in enumerate(rows):
                        for i2 in range(len(row2)):
                            for j2 in range(i2+1, len(row2)):
                                d2 = row2[j2] - row2[i2]
                                if d2 not in all_diffs:
                                    all_diffs[d2] = []
                                all_diffs[d2].append((r_idx2, i2, j2))
                    break
        
        return rows

    def local_search(rows, max_scope):
        """
        Perform local search to improve the solution.
        """
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try to reduce scope by shrinking elements
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for j in range(len(row)-1, 0, -1):
                    old_val = row[j]
                    # Try to reduce old_val
                    for new_val in range(old_val - 1, row[j-1], -1):
                        # Check validity
                        # This is a simplified check, full validation would be expensive
                        # We'll do a full validation after the move
                        row[j] = new_val
                        if is_valid(rows):
                            improved = True
                            break
                        row[j] = old_val
                    if improved:
                        break
                if improved:
                    break
        return rows

    # Initialize population
    pop_size = 50
    population = []
    
    # Add a greedy solution
    used_diffs = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = greedy_construct(112, used_diffs)
        if row:
            greedy_rows.append(row)
            used_diffs.update(get_row_diffs(row))
        else:
            greedy_rows = None
            break
    if greedy_rows:
        population.append(greedy_rows)
    
    # Add random solutions
    for _ in range(pop_size - 1):
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = random_valid_row(112, used_diffs)
            if row:
                rows.append(row)
                used_diffs.update(get_row_diffs(row))
            else:
                success = False
                break
        if success:
            population.append(rows)
    
    # If population is too small, add more greedy-like solutions
    while len(population) < pop_size:
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = greedy_construct(112, used_diffs)
            if row:
                rows.append(row)
                used_diffs.update(get_row_diffs(row))
            else:
                success = False
                break
        if success:
            population.append(rows)
        else:
            # Try with a larger scope
            used_diffs = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = greedy_construct(120, used_diffs)
                if row:
                    rows.append(row)
                    used_diffs.update(get_row_diffs(row))
                else:
                    success = False
                    break
            if success:
                population.append(rows)
            else:
                break
    
    # Evolutionary loop
    best_solution = min(population, key=scope)
    best_scope = scope(best_solution)
    
    while time.time() < DEADLINE:
        # Selection: Tournament selection
        new_population = []
        
        # Elitism: Keep the best solution
        new_population.append(best_solution)
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Select parents
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            if random.random() < 0.7:
                child = mutate(child, 112)
            
            # Local search
            if random.random() < 0.3:
                child = local_search(child, 112)
            
            # Validate and add to new population
            if is_valid(child):
                new_population.append(child)
                child_scope = scope(child)
                if child_scope < best_scope:
                    best_scope = child_scope
                    best_solution = child
            else:
                # If invalid, try to repair it
                # Simplified repair: replace invalid rows with greedy rows
                repaired = []
                used_diffs = set()
                for r in child:
                    # Try to keep the row if it's valid locally
                    # This is a simplified repair
                    pass
                
                # For now, just add a new random solution
                used_diffs = set()
                rows = []
                success = True
                for _ in range(n_rows):
                    row = random_valid_row(112, used_diffs)
                    if row:
                        rows.append(row)
                        used_diffs.update(get_row_diffs(row))
                    else:
                        success = False
                        break
                if success:
                    new_population.append(rows)
                else:
                    # Fallback to greedy
                    used_diffs = set()
                    rows = []
                    success = True
                    for _ in range(n_rows):
                        row = greedy_construct(112, used_diffs)
                        if row:
                            rows.append(row)
                            used_diffs.update(get_row_diffs(row))
                        else:
                            success = False
                            break
                    if success:
                        new_population.append(rows)
                    else:
                        # Give up on this offspring
                        break
        
        # Replace population
        population = new_population
        
        # Check if we found a solution with scope 105
        if best_scope <= 105:
            break
    
    # Return the best solution found
    return best_solution

# Fallback in case the above fails or times out
def construct_fallback(n_rows, k):
    num_elements = k + 1
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

# Wrap the main construct to ensure it doesn't crash
def construct(n_rows, k):
    try:
        result = construct_moena(n_rows, k)
        if result and is_valid(result):
            return result
        else:
            return construct_fallback(n_rows, k)
    except Exception:
        return construct_fallback(n_rows, k)

# Rename the MOEA function to avoid conflict
construct_moena = construct

# Re-define the main entry point to use the MOEA strategy
def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = 145.0
    
    num_elements = k + 1
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        return len(all_diffs) == len(set(all_diffs))

    def scope(rows):
        return max(max(r) for r in rows) if rows else float('inf')

    def greedy_construct(max_scope, used_diffs=None):
        if used_diffs is None:
            used_diffs = set()
        row = [0]
        current_max = 0
        while len(row) < num_elements:
            found = False
            for v in range(current_max + 1, max_scope + 1):
                valid = True
                for x in row:
                    d = v - x
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(v)
                    for x in row[:-1]:
                        used_diffs.add(v - x)
                    current_max = v
                    found = True
                    break
            if not found:
                return None
        return row

    def random_valid_row(max_scope, used_diffs=None):
        if used_diffs is None:
            used_diffs = set()
        row = [0]
        current_max = 0
        while len(row) < num_elements:
            min_next = current_max + 1
            max_next = max_scope - (num_elements - len(row) - 1)
            if min_next > max_next:
                return None
            attempts = 0
            while attempts < 100:
                v = random.randint(min_next, max_next)
                valid = True
                for x in row:
                    d = v - x
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(v)
                    for x in row[:-1]:
                        used_diffs.add(v - x)
                    current_max = v
                    break
                attempts += 1
            if attempts >= 100:
                found = False
                for v in range(current_max + 1, max_next + 1):
                    valid = True
                    for x in row:
                        d = v - x
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        for x in row[:-1]:
                            used_diffs.add(v - x)
                        current_max = v
                        found = True
                        break
                if not found:
                    return None
        return row

    def crossover(parent1, parent2):
        child = []
        for i in range(n_rows):
            if random.random() < 0.5:
                child.append(list(parent1[i]))
            else:
                child.append(list(parent2[i]))
        return child

    def mutate(rows, max_scope):
        all_diffs = {}
        for r_idx, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d not in all_diffs:
                        all_diffs[d] = []
                    all_diffs[d].append((r_idx, i, j))
        conflicts = [d for d, locs in all_diffs.items() if len(locs) > 1]
        if not conflicts:
            for r_idx in range(n_rows):
                row = rows[r_idx]
                last_val = row[-1]
                for new_val in range(last_val - 1, 0, -1):
                    if new_val <= row[-2]:
                        break
                    old_diffs = set()
                    for i in range(len(row)-1):
                        old_diffs.add(last_val - row[i])
                    new_diffs = set()
                    for i in range(len(row)-1):
                        new_diffs.add(new_val - row[i])
                    temp_diffs = {}
                    for d, locs in all_diffs.items():
                        if d in old_diffs:
                            new_locs = [loc for loc in locs if not (loc[0] == r_idx and (loc[1] == len(row)-1 or loc[2] == len(row)-1))]
                            if new_locs:
                                temp_diffs[d] = new_locs
                        else:
                            temp_diffs[d] = locs
                    valid = True
                    for d in new_diffs:
                        if d in temp_diffs:
                            valid = False
                            break
                    if valid:
                        row[-1] = new_val
                        break
            return rows
        for d in conflicts:
            locs = all_diffs[d]
            r_idx, i, j = random.choice(locs)
            row = rows[r_idx]
            old_val = row[j]
            for new_val in range(row[j-1] + 1, old_val):
                valid = True
                for x_idx in range(j):
                    new_d = new_val - row[x_idx]
                    if new_d in all_diffs and new_d != d:
                        valid = False
                        break
                if valid:
                    row[j] = new_val
                    all_diffs = {}
                    for r_idx2, row2 in enumerate(rows):
                        for i2 in range(len(row2)):
                            for j2 in range(i2+1, len(row2)):
                                d2 = row2[j2] - row2[i2]
                                if d2 not in all_diffs:
                                    all_diffs[d2] = []
                                all_diffs[d2].append((r_idx2, i2, j2))
                    break
        return rows

    def local_search(rows, max_scope):
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for j in range(len(row)-1, 0, -1):
                    old_val = row[j]
                    for new_val in range(old_val - 1, row[j-1], -1):
                        row[j] = new_val
                        if is_valid(rows):
                            improved = True
                            break
                        row[j] = old_val
                    if improved:
                        break
                if improved:
                    break
        return rows

    pop_size = 50
    population = []
    used_diffs = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = greedy_construct(112, used_diffs)
        if row:
            greedy_rows.append(row)
            used_diffs.update(get_row_diffs(row))
        else:
            greedy_rows = None
            break
    if greedy_rows:
        population.append(greedy_rows)
    for _ in range(pop_size - 1):
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = random_valid_row(112, used_diffs)
            if row:
                rows.append(row)
                used_diffs.update(get_row_diffs(row))
            else:
                success = False
                break
        if success:
            population.append(rows)
    while len(population) < pop_size:
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = greedy_construct(112, used_diffs)
            if row:
                rows.append(row)
                used_diffs.update(get_row_diffs(row))
            else:
                success = False
                break
        if success:
            population.append(rows)
        else:
            used_diffs = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = greedy_construct(120, used_diffs)
                if row:
                    rows.append(row)
                    used_diffs.update(get_row_diffs(row))
                else:
                    success = False
                    break
            if success:
                population.append(rows)
            else:
                break
    best_solution = min(population, key=scope)
    best_scope = scope(best_solution)
    while time.time() < DEADLINE:
        new_population = []
        new_population.append(best_solution)
        while len(new_population) < pop_size:
            p1 = random.choice(population)
            p2 = random.choice(population)
            child = crossover(p1, p2)
            if random.random() < 0.7:
                child = mutate(child, 112)
            if random.random() < 0.3:
                child = local_search(child, 112)
            if is_valid(child):
                new_population.append(child)
                child_scope = scope(child)
                if child_scope < best_scope:
                    best_scope = child_scope
                    best_solution = child
            else:
                used_diffs = set()
                rows = []
                success = True
                for _ in range(n_rows):
                    row = random_valid_row(112, used_diffs)
                    if row:
                        rows.append(row)
                        used_diffs.update(get_row_diffs(row))
                    else:
                        success = False
                        break
                if success:
                    new_population.append(rows)
                else:
                    used_diffs = set()
                    rows = []
                    success = True
                    for _ in range(n_rows):
                        row = greedy_construct(112, used_diffs)
                        if row:
                            rows.append(row)
                            used_diffs.update(get_row_diffs(row))
                        else:
                            success = False
                            break
                    if success:
                        new_population.append(rows)
                    else:
                        break
        population = new_population
        if best_scope <= 105:
            break
    return best_solution

# Final fallback
def construct(n_rows, k):
    try:
        result = construct_moena(n_rows, k)
        if result and is_valid(result):
            return result
        else:
            num_elements = k + 1
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            return rows
    except Exception:
        num_elements = k + 1
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
