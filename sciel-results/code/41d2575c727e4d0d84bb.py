import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set using randomized backtracking with constraint propagation.
    
    Strategy:
    1. We need n_rows rows, each of size k+1, starting with 0.
    2. All differences a_j - a_i (i < j) within all rows must be distinct.
    3. We want to minimize the maximum entry (the scope).
    
    Instead of greedy + local search, we use a depth-first search (DFS) with backtracking.
    We build the set row by row. For each row, we choose the elements sequentially.
    To make the search efficient:
    - We maintain a set of 'used' differences.
    - We use a heuristic to pick the next element: we try smaller values first to keep the scope small, 
      but we also use randomization to explore different branches.
    - We use a time limit to switch to a "best effort" mode or restart.
    
    However, pure backtracking can be slow. 
    A hybrid approach:
    1. Try to find a solution with a specific scope M.
    2. If successful, try a smaller M.
    3. If time runs out, return the best valid solution found (smallest M).
    
    Actually, the task asks for the set, and the score is the largest entry.
    We don't know the optimal M in advance.
    
    Let's refine the strategy:
    We will perform a randomized DFS that tries to construct the rows.
    We will bias the search towards smaller numbers.
    We will keep track of the best solution (smallest max entry) found so far.
    We will use "conflict-directed backjumping" or simply standard backtracking with randomization.
    
    Given the small k (7) and n_rows (4), the search space is manageable if we prune well.
    
    Key Insight: The number of differences per row is (k+1)*k/2. For k=7, that's 8*7/2 = 28 diffs per row.
    Total diffs = 4 * 28 = 112.
    The differences must be distinct integers.
    The smallest possible max difference is roughly related to the scope.
    
    Let's implement a randomized backtracking solver that tries to minimize the max value.
    We'll use a "best-first" or "depth-first" with pruning.
    
    Since we have a 120s limit, we can afford a significant amount of search.
    
    Algorithm:
    1. Initialize best_score = infinity.
    2. Use a randomized DFS.
       - State: current rows, used_diffs, current_row_being_built.
       - At each step, pick a candidate value for the next element in the current row.
       - Candidates are generated in a randomized order, biased towards smaller values.
       - If a candidate leads to a conflict, backtrack.
       - If we complete all rows, update best_score.
       - If we exceed a certain depth or time, backtrack.
       
    To improve efficiency:
    - We can use a "fail-first" heuristic: choose the next variable (position in the row) that has the fewest valid values. But since we build sequentially, the variable is fixed.
    - Instead, we can randomize the order of rows? No, rows are symmetric.
    - We can randomize the starting point? No, starts at 0.
    
    Let's try a different angle: **Simulated Annealing on the entire set of rows**.
    The current program does hill climbing on single elements. 
    A more powerful move is to **swap entire rows** or **re-generate a row** using a constrained solver.
    
    Actually, let's look at Inspiration 1 which scored 1.4979. It used `randomized_greedy`.
    The current program is stuck at 1.0.
    
    Let's try a **Genetic Algorithm** or **Memetic Algorithm**.
    - Population of solutions (sets of rows).
    - Fitness: -max_entry (we want to minimize).
    - Crossover: Take rows from different parents.
    - Mutation: Perturb a row using local search.
    - Selection: Keep the best.
    
    This is a different design family (Evolutionary) vs (Local Search).
    
    Let's implement a simple GA.
    
    1. Initialize population with diverse solutions (randomized greedy with different seeds/strategies).
    2. Evaluate fitness.
    3. While time < deadline:
       a. Select parents (tournament selection).
       b. Crossover: Combine rows from parents.
       c. Mutation: Apply local search (hill climbing) to the child to improve it.
       d. Replace worst individuals in population.
    
    This combines global search (GA) with local refinement (Hill Climbing).
    The key is that GA can jump out of local minima by combining good rows from different solutions.
    
    Implementation Details:
    - Representation: List of rows.
    - Fitness: -max_entry.
    - Crossover: Uniform crossover on rows. Pick a random subset of rows from Parent A and the rest from Parent B.
    - Mutation: 
      - Pick a random row.
      - Try to improve it using a local search that allows "worse" moves temporarily? 
      - Or simply re-generate the row using a constrained greedy that avoids used diffs from *other* rows.
    
    Let's implement the "Re-generate Row" mutation.
    Given a set of rows, pick one row to replace.
    Compute the set of used diffs from the *other* rows.
    Try to find a new row that:
      - Starts with 0.
      - Has k+1 elements.
      - Uses no diffs from the "used_by_others" set.
      - Has a small max value.
    
    This is a constrained construction problem. We can use a randomized backtracking for this single row.
    
    This approach is robust because it doesn't rely on the initial greedy being good; it continuously tries to replace rows with better ones.
    """
    
    start_time = time.time()
    deadline = start_time + 115
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            for d in get_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    def objective(rows):
        return max(row[-1] for row in rows)

    # 1. Randomized Greedy Construction for a single row given forbidden diffs
    def construct_row(forbidden_diffs, limit=None, seed=None):
        """
        Constructs a valid row of size k+1 starting with 0.
        Avoids diffs in forbidden_diffs.
        Tries to keep values small.
        Returns None if no row found within reasonable effort.
        """
        if seed is not None:
            random.seed(seed)
        
        # We use a randomized backtracking search for the row
        # State: current_row, current_diffs
        # We want to find a row with max < limit (if provided)
        
        # To speed up, we use a recursive helper
        def backtrack(current_row, current_diffs):
            if len(current_row) == k + 1:
                return current_row[:]
            
            last = current_row[-1]
            # Determine range for next element
            # We want to try smaller values first, but randomize to avoid bias
            # Limit is important. If limit is None, we can go up to some heuristic max
            max_possible = limit if limit is not None else 200 # Heuristic cap
            
            # Candidates: last+1 to max_possible
            # We will shuffle a range of candidates to try
            # To be efficient, we try a window of candidates
            candidates = list(range(last + 1, max_possible + 1))
            random.shuffle(candidates)
            
            # Prioritize smaller candidates? 
            # Let's sort by a mix of value and randomness
            # Actually, just shuffling is fine for exploration.
            # But for exploitation, we want small values.
            # Let's try the first 20 smallest valid candidates first, then random.
            
            # Optimization: Check conflicts early
            for val in candidates:
                new_diffs = {val - a for a in current_row}
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & current_diffs:
                    continue
                
                # Recurse
                res = backtrack(current_row + [val], current_diffs | new_diffs)
                if res is not None:
                    return res
            
            return None
        
        return backtrack([0], set())

    # 2. Initial Population
    # We create diverse solutions using randomized greedy
    
    def greedy_construct(n_rows, k, seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            # Use the constrained row constructor with no forbidden diffs initially?
            # No, we must avoid diffs used by previous rows.
            row = construct_row(used, seed=seed)
            if row is None:
                # Fallback to simple greedy if backtracking fails (shouldn't happen for small k)
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
            rows.append(row)
            used |= get_diffs(row)
        return rows

    population = []
    # Create 50 diverse individuals
    for i in range(50):
        pop = greedy_construct(n_rows, k, seed=i)
        population.append(pop)
    
    # Sort by objective
    population.sort(key=objective)
    best_individual = population[0]
    best_score = objective(best_individual)
    
    # 3. Evolutionary Loop
    pop_size = 20
    # Keep the top 20 individuals
    
    def evolve():
        nonlocal best_individual, best_score
        while time.time() < deadline:
            # Selection
            # Tournament selection
            def select():
                candidates = random.sample(population, 3)
                return min(candidates, key=objective)
            
            # Crossover
            parent1 = select()
            parent2 = select()
            
            # Uniform crossover on rows
            child_rows = []
            for i in range(n_rows):
                if random.random() < 0.5:
                    child_rows.append(parent1[i][:])
                else:
                    child_rows.append(parent2[i][:])
            
            # Mutation: Improve the child using local search
            # We try to replace each row with a better one
            for i in range(n_rows):
                # Compute forbidden diffs from other rows
                other_rows = [child_rows[j] for j in range(n_rows) if j != i]
                forbidden = set()
                for r in other_rows:
                    forbidden |= get_diffs(r)
                
                # Try to find a better row for position i
                # Current row max: child_rows[i][-1]
                # We want a row with max < current max
                # Or just a valid row, and we'll compare later
                
                # Try a few times to find a row with smaller max
                current_max = child_rows[i][-1]
                found_better = False
                for attempt in range(10):
                    # Limit to current_max - 1 to force improvement
                    # If we can't find one, try without limit? No, that might make it worse.
                    # Let's try limit = current_max - 1
                    new_row = construct_row(forbidden, limit=current_max - 1, seed=random.randint(0, 10000))
                    if new_row is not None:
                        # Check if it's actually better (it should be, due to limit)
                        # But we must ensure it's valid against forbidden (constructor does this)
                        child_rows[i] = new_row
                        found_better = True
                        break
                
                # If we didn't find a better row, maybe try to find *any* valid row to diversify?
                # No, we want to minimize. If we can't improve, keep the old one.
                # But we might be stuck. 
                # Let's try a "kick": replace with a random valid row (even if worse) to escape?
                # For now, stick to improvement.
            
            # Evaluate child
            if not is_valid(child_rows):
                # Should not happen if construction is correct, but just in case
                continue
                
            child_score = objective(child_rows)
            
            # Insert into population
            population.append(child_rows)
            population.sort(key=objective)
            
            # Trim population
            if len(population) > pop_size:
                population = population[:pop_size]
            
            # Update best
            if child_score < best_score:
                best_score = child_score
                best_individual = [r[:] for r in child_rows]
                
            # Periodically re-seed population with fresh greedy solutions to maintain diversity
            if len(population) < 5: # This shouldn't happen, but just in case
                pass
            
            # Every 100 generations, add a new random greedy individual
            if len(population) > 0 and random.random() < 0.1:
                new_ind = greedy_construct(n_rows, k, seed=random.randint(0, 100000))
                population.append(new_ind)
                population.sort(key=objective)
                if len(population) > pop_size:
                    population = population[:pop_size]

    evolve()
    
    # Final check
    if is_valid(best_individual):
        return best_individual
    else:
        # Fallback
        return greedy_construct(n_rows, k)

