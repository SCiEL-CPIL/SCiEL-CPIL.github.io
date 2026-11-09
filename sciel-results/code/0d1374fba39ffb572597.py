import time
import sys
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Greedy fallback: always available
    def greedy_solve():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            for _ in range(k):
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        return rows

    greedy_rows = greedy_solve()
    greedy_M = max(max(r) for r in greedy_rows)
    best_solution = greedy_rows
    best_M = greedy_M

    # New Strategy: Backtracking on Difference Values
    # We search for a packing of n_rows rows such that all differences are distinct.
    # We view this as assigning rows to "slots".
    # Instead of picking a row first, we pick a "critical" difference value that must be covered.
    # Then we try to complete a row that uses that difference.
    
    # To make this tractable, we iterate over M (max entry).
    # For a fixed M, we have a universe of differences D = {1, ..., M}.
    # We need to partition D (or a subset of it) into n_rows sets, each being the diff-set of a valid row.
    
    # Heuristic:
    # 1. Start with an empty set of used differences.
    # 2. Identify the smallest unused difference 'd_min'.
    # 3. Generate all valid rows (within [0, M]) that contain 'd_min' as a difference.
    # 4. For each such row, check if its diff-set is disjoint from 'used'.
    # 5. Recursively add the row and repeat.
    # 6. If we reach n_rows, success.
    
    # Optimization:
    # - Precompute all valid rows for a given M? 
    #   C(M, k) is too large for M~110, k=5 (C(110,5) ~ 4e7). 
    #   However, we don't need all. We only need rows that contain a specific difference d.
    #   The number of pairs (i, j) in a row of length k+1 within [0, M] such that row[j]-row[i]=d is limited.
    #   Actually, generating rows containing a specific difference d is much cheaper.
    
    def generate_rows_with_diff(M, d, existing_rows=None):
        """
        Generate all valid rows [0, x1, ..., xk] with 0 < x1 < ... < xk <= M
        such that d is in the difference set of the row.
        """
        # A row contains difference d if there exist i < j such that row[j] - row[i] = d.
        # Since row[0]=0, if d is a difference, it could be:
        # 1. x_j - 0 = d => x_j = d
        # 2. x_j - x_i = d => x_j = x_i + d
        
        # We can generate all rows of length k from [1..M] and filter?
        # No, that's expensive.
        
        # Let's use a recursive generation that forces the presence of d.
        # This is complex. Alternative:
        # For a specific M, precompute a list of "candidate rows" that are "good".
        # But M varies.
        
        # Let's try a different angle for the backtracking:
        # Instead of fixing M and searching rows, we build rows one by one.
        # But we use "Look-ahead" or "Constraint Propagation" on the difference set.
        
        # Actually, the most effective structural change for small n_rows (7) and k (5) is:
        # Use a SAT-like solver or a specialized backtracking that orders variables by degree of constraint.
        
        # Let's implement a backtracking that picks the "hardest" row to place next.
        # State: (used_diffs, rows_placed)
        # Move: Pick a row R such that R's diff-set is disjoint from used_diffs.
        # To pick R, we don't guess randomly. We pick the smallest unused difference d.
        # Any valid solution MUST cover d with some row.
        # So we enumerate all valid rows R that cover d.
        # For each R, if R is disjoint from used_diffs, we recurse.
        
        # This is a "dancing links" style or exact cover approach.
        
        pass # Placeholder, will implement inside try_backtrack

    def try_backtrack(M, time_budget):
        """
        Try to find n_rows rows with max entry <= M using backtracking on difference coverage.
        """
        end_time = start_time + time_budget
        
        # Precompute all valid rows for this M?
        # C(M, k) for M=110, k=5 is ~40 million. Too many to precompute.
        # But we can generate them on the fly for a specific difference d.
        
        # Helper: Generate all rows of length k+1 starting with 0, elements in 1..M, 
        # such that d is in the diff set.
        def get_rows_containing_d(d):
            rows = set()
            # Case 1: d is a difference from 0. So d is in the row.
            # Case 2: d is a difference between two non-zero elements.
            
            # We can generate all combinations of k elements from 1..M? No.
            # We can generate rows by picking the positions of elements.
            
            # Efficient generation:
            # Iterate over all pairs (i, j) in the row indices (0 <= i < j <= k) such that the difference is d.
            # Let the row be [0, x1, x2, ..., xk].
            # If i=0, j=some index p, then x_p = d.
            # If i>0, then x_j - x_i = d.
            
            # This is still complex to enumerate without duplicates.
            
            # Alternative: 
            # Since M is small (~105-120), and k=5, the number of valid rows is not THAT huge if we prune.
            # But C(110, 5) is large.
            
            # Let's try a different structural approach:
            # Local Search / Simulated Annealing on the set of rows.
            # Represent state as a list of n_rows rows.
            # Objective: Minimize max_entry + penalty(conflicts).
            # Moves: 
            # 1. Replace one row with a random valid row.
            # 2. Swap two elements in a row? No, must keep sorted.
            # 3. Perturb a row: change one element.
            
            # But the current program already does randomized set packing.
            # The backtracking on "smallest unused difference" is a known powerful heuristic for Exact Cover.
            
            # Let's implement the "Smallest Unused Difference" backtracking properly.
            # To make it fast, we need to generate candidate rows for a given d quickly.
            
            # Optimization: Precompute for a given M, a map: d -> list of rows containing d.
            # How to build this map efficiently?
            # Iterate all rows? No.
            
            # Let's assume we can generate rows containing d on the fly.
            # For a fixed d, the number of rows containing d is much smaller than total rows.
            
            def gen_rows_with_diff(M, d):
                # Returns a list of rows (tuples) that contain difference d.
                # Row is (x1, ..., xk) with 0 < x1 < ... < xk <= M.
                # Full row is (0, x1, ..., xk).
                
                # We need to find all subsets S of {1..M} of size k such that d in Diff(S ∪ {0}).
                
                # This is still hard to enumerate directly.
                
                # Let's fall back to a smarter randomized search if backtracking is too slow to implement correctly in time.
                # However, the prompt asks for a STRUCTURAL change.
                # 
                # Let's try: **Iterative Deepening A* (IDA*)** on the rows?
                # Or **Genetic Algorithm**?
                
                # Actually, for n=7, k=5, the space is small enough for a well-pruned backtracking.
                # The key is the generation of rows.
                
                # Let's precompute ALL valid rows for M up to 120?
                # C(120, 5) = 190 million. Too many.
                
                # What if we restrict the search space by "shape"?
                
                # Let's try a different structural change:
                # **Min-Conflict Heuristic** with **WalkSAT** style moves.
                # Start with a random valid packing (or greedy).
                # If there are conflicts (duplicate diffs), pick a conflict.
                # Move a row to resolve the conflict.
                # This is local search.
                
                # But the current program is already randomized greedy.
                
                # Let's implement the **Exact Cover Backtracking** with a twist:
                # We don't precompute all rows. We generate them lazily.
                # To generate rows containing d:
                # We know d = y - x where x, y are in the row (including 0).
                # If x=0, y=d. So d must be in the row.
                # If x>0, then x and x+d are in the row.
                
                # We can iterate over all possible "anchors" for d.
                # Anchor type 1: 0 and d are in the row.
                # Anchor type 2: x and x+d are in the row, for some x in 1..M-d.
                
                # For each anchor, we need to fill the remaining k-2 (or k-1) elements.
                # This is a constrained combination problem.
                
                # Given time constraints, implementing a perfect lazy generator is risky.
                
                # Let's try a **Hybrid**: 
                # 1. Use the current greedy to get a baseline.
                # 2. Use a **Simulated Annealing** that operates on the difference set directly?
                #    No, the row structure is rigid.
                
                # Let's go with **Backtracking with Memoization on (used_diffs, rows_left)**?
                # State space is huge.
                
                # Let's try **Randomized Restart with Smarter Initialization**.
                # The current program samples random rows. 
                # Structural change: **Sample rows from a Gaussian distribution of differences**?
                
                # Actually, the most promising structural change for this specific small instance (7,5) is:
                # **Pre-compute a library of "good" rows** using a separate offline-like process within the 120s?
                # No, we must do it online.
                
                # Let's implement a **Depth-First Search (DFS)** that builds rows one by one, 
                # but uses **Forward Checking** to prune earlier.
                # Current program does randomized greedy.
                # DFS with backtracking is structurally different (systematic vs random).
                
                # To make DFS feasible, we need good variable ordering.
                # Order rows by "difficulty" (fewest valid completions).
                
                # Let's implement DFS with Forward Checking.
                
                return None # Placeholder

        # Since implementing the lazy row generator for Exact Cover is complex and risky in this context,
        # and the current program is already a form of randomized search,
        # I will implement a **Systematic Backtracking** that is faster than random sampling
        # by using **Constraint Propagation** on the difference set.
        
        # We will search for rows in order of "density" of differences.
        
        # Actually, let's look at the "Inspiration 3" note: "sequential, depth-first backtracking... is extremely brittle".
        # So we should avoid simple DFS.
        
        # Let's try **Min-Cost Max-Flow**?
        # Model as a bipartite graph? Rows to Differences?
        # Each row "uses" a set of differences.
        # We want to select n_rows rows such that no difference is used twice.
        # This is Set Packing. NP-Hard.
        
        # For small n_rows=7, we can use **Branch and Bound**.
        # Bound: If the number of available differences remaining is less than n_rows * diffs_per_row, prune.
        # This is weak.
        
        # Let's try a **Genetic Algorithm** with a fitness function based on:
        # 1. Validity (no conflicts)
        # 2. Maximality (spread of differences)
        # 3. Minimality (max entry)
        
        # This is a different optimization family (Evolutionary vs Stochastic Local Search/Greedy).
        
        def genetic_search(M, time_budget):
            end_time = start_time + time_budget
            pop_size = 100
            generations = 0
            
            # Initial population: Random valid rows?
            # We need rows that are individually valid (always true) but we need the SET to be valid.
            # Fitness of a set: -1 if conflicts, else -max_entry.
            
            # Representation: List of n_rows rows.
            # Row: list of k+1 ints.
            
            def fitness(individual):
                used = set()
                max_val = 0
                for row in individual:
                    if row[-1] > M:
                        return -1000000
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            d = row[j] - row[i]
                            if d in used:
                                return -1000000 + (len(used) / total_diffs) # Penalty for conflicts
                            used.add(d)
                    max_val = max(max_val, row[-1])
                return -max_val # Lower is better, so negative max_val is higher fitness? 
                # Let's use Score = -max_val. Higher is better.
                
            def generate_random_row(M):
                elems = sorted(random.sample(range(1, M+1), k))
                return [0] + elems

            # Initialize population
            population = []
            for _ in range(pop_size):
                ind = [generate_random_row(M) for _ in range(n_rows)]
                population.append(ind)
                
            best_ind = max(population, key=fitness)
            best_fit = fitness(best_ind)
            
            while time.time() < end_time:
                # Selection
                parents = sorted(population, key=fitness, reverse=True)[:pop_size//2]
                
                # Crossover and Mutation
                new_population = []
                while len(new_population) < pop_size:
                    p1 = random.choice(parents)
                    p2 = random.choice(parents)
                    
                    # Crossover: Take rows from p1 and p2
                    child = []
                    for i in range(n_rows):
                        if random.random() < 0.5:
                            child.append(p1[i][:])
                        else:
                            child.append(p2[i][:])
                            
                    # Mutation: Perturb a row
                    if random.random() < 0.5:
                        idx = random.randint(0, n_rows-1)
                        # Change one element in the row
                        row = child[idx]
                        pos = random.randint(1, k) # 1..k (0 is fixed)
                        old_val = row[pos]
                        # Try to find a new value that keeps row sorted and within M
                        # Simple mutation: replace with random value and sort
                        new_val = random.randint(1, M)
                        row[pos] = new_val
                        row.sort()
                        # Ensure no duplicates (strictly increasing)
                        # If duplicates, remove one? Or reject mutation?
                        # Let's just reject if not strictly increasing.
                        if any(row[i] == row[i+1] for i in range(len(row)-1)):
                            # Revert
                            row[pos] = old_val
                            row.sort()
                        # Ensure starts with 0
                        if row[0] != 0:
                            row[0] = 0
                            row = sorted(set(row)) # Might reduce length, need to handle
                            if len(row) < k+1:
                                # Fill with randoms
                                while len(row) < k+1:
                                    row.append(random.randint(1, M))
                                row = sorted(set(row))[:k+1]
                                if row[0] != 0:
                                    row[0] = 0
                                    row = sorted(row)
                    
                    new_population.append(child)
                
                population = new_population
                current_best = max(population, key=fitness)
                current_fit = fitness(current_best)
                
                if current_fit > best_fit:
                    best_fit = current_fit
                    best_ind = current_best
                    
                # If we found a valid solution (no conflicts), check if it's better than global best
                if best_fit > -1000000: # Valid
                    actual_M = max(max(r) for r in best_ind)
                    if actual_M < best_M:
                        return best_ind

            return None

        # Try M values
        lower_bound = max(total_diffs, k)
        upper_try = min(greedy_M - 1, lower_bound + 20)
        
        remaining = deadline - time.time() - 5.0
        if remaining <= 0:
            return best_solution
            
        time_per_M = remaining / max(1, upper_try - lower_bound + 1)
        
        for M in range(lower_bound, upper_try + 1):
            if time.time() > deadline - 5.0:
                break
            alloc = min(time_per_M, deadline - time.time() - 3.0)
            if alloc < 1.0:
                break
                
            sol = genetic_search(M, alloc)
            if sol is not None:
                if is_valid(sol):
                    actual_M = max(max(r) for r in sol)
                    if actual_M < best_M:
                        best_solution = sol
                        best_M = actual_M
                        break

    # Run the new strategy
    try_backtrack = None # We decided against complex backtracking, using Genetic instead
    # Actually, let's stick to the Genetic Search as the structural change.
    
    # Integrate Genetic Search into the main flow
    # We already defined genetic_search inside try_backtrack which is not called.
    # Let's move it out or call it directly.
    
    # Re-define genetic_search outside try_backtrack for clarity in the final code structure below.
    # For now, the code above has it nested. I will clean it up in the final output.
    
    return best_solution

# The code above is a draft. I will now write the clean version.
