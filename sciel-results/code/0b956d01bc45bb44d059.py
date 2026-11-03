import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    best_max = float('inf')
    best_rows = None

    # --- Greedy Baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(max(r) for r in best_rows)

    # --- Helper: get all pairwise differences of a row ---
    def get_diffs(row):
        diffs = []
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                diffs.append(row[j] - row[i])
        return diffs

    # --- Genetic Algorithm in difference-space ---
    # Each individual is represented by per-row difference sequences.
    # A row with elements [0, a1, a2, ..., ak] has differences:
    #   d1 = a1, d2 = a2-a1, d3 = a3-a2, ..., dk = ak-a_{k-1}
    # All differences must be positive integers.
    # The full set of pairwise differences is determined by the row.
    
    def row_from_diffs(diffs):
        """Reconstruct row from consecutive differences."""
        row = [0]
        for d in diffs:
            row.append(row[-1] + d)
        return row

    def diffs_from_row(row):
        """Get consecutive differences from a row."""
        return [row[i+1] - row[i] for i in range(len(row)-1)]

    def pairwise_diffs_from_row(row):
        """Get all pairwise differences."""
        L = len(row)
        diffs = []
        for i in range(L):
            for j in range(i+1, L):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_fitness(individual, diff_counts):
        """
        individual: list of k consecutive-difference sequences (one per row)
        diff_counts: dict of diff -> count across all rows
        Returns (collisions, max_val)
        """
        collisions = 0
        max_val = 0
        for i, d in diff_counts.items():
            if d > 1:
                collisions += d - 1
        for diffs in individual:
            row = row_from_diffs(diffs)
            if row[-1] > max_val:
                max_val = row[-1]
        return collisions, max_val

    def build_diff_counts(individual):
        """Build global diff counts from an individual."""
        diff_counts = {}
        for diffs in individual:
            row = row_from_diffs(diffs)
            for d in pairwise_diffs_from_row(row):
                diff_counts[d] = diff_counts.get(d, 0) + 1
        return diff_counts

    def ga_optimize(initial_individual, time_limit, target_bound):
        """
        Genetic algorithm operating in difference-space.
        initial_individual: list of k diff-sequences
        Returns best (rows, max_val) found.
        """
        pop_size = 30
        n = n_rows
        L = row_len
        
        # Initialize population
        population = [initial_individual[:]]
        for _ in range(pop_size - 1):
            ind = [d[:] for d in initial_individual]
            # Random perturb
            for _ in range(random.randint(2, 5)):
                r_idx = random.randint(0, n - 1)
                pos = random.randint(0, k - 1)
                old = ind[r_idx][pos]
                new = max(1, old + random.randint(-3, 3))
                ind[r_idx][pos] = new
            population.append(ind)
        
        # Compute initial fitness
        best_ind = None
        best_collisions = float('inf')
        best_max = float('inf')
        
        for ind in population:
            dc = build_diff_counts(ind)
            coll, mx = compute_fitness(ind, dc)
            if coll < best_collisions or (coll == best_collisions and mx < best_max):
                best_collisions = coll
                best_max = mx
                best_ind = [d[:] for d in ind]
        
        start_time = time.time()
        gen = 0
        T = 100.0
        
        while time.time() - start_time < time_limit:
            gen += 1
            # Evaluate current population
            scored = []
            for ind in population:
                dc = build_diff_counts(ind)
                coll, mx = compute_fitness(ind, dc)
                scored.append((coll, mx, ind))
            
            scored.sort(key=lambda x: (x[0], x[1]))
            
            # Update best
            if scored[0][0] < best_collisions or (scored[0][0] == best_collisions and scored[0][1] < best_max):
                best_collisions = scored[0][0]
                best_max = scored[0][1]
                best_ind = [d[:] for d in scored[0][2]]
            
            # If we found a valid solution, try to improve
            if best_collisions == 0:
                # Focus on reducing max_val
                T *= 0.95
                if T < 1.0:
                    T = 1.0
            
            # Selection: tournament
            def select():
                a = random.randint(0, min(5, len(scored)-1))
                b = random.randint(0, min(5, len(scored)-1))
                return scored[a][2] if scored[a][0] < scored[b][0] or (scored[a][0] == scored[b][0] and scored[a][1] <= scored[b][1]) else scored[b][2]
            
            # Generate next generation
            new_pop = []
            
            # Elitism: keep top few
            for i in range(min(5, len(scored))):
                new_pop.append([d[:] for d in scored[i][2]])
            
            # Crossover: swap difference segments between two parents
            while len(new_pop) < pop_size and time.time() - start_time < time_limit - 0.5:
                p1 = select()
                p2 = select()
                
                child = [d[:] for d in p1]
                
                # For each row, potentially swap a contiguous segment of differences
                for r_idx in range(n):
                    if random.random() < 0.5:
                        # Swap entire row
                        child[r_idx] = p2[r_idx][:]
                    elif random.random() < 0.3:
                        # Swap a contiguous segment
                        if k >= 2:
                            seg_start = random.randint(0, k - 2)
                            seg_end = random.randint(seg_start, k - 1)
                            child[r_idx][seg_start:seg_end+1] = p2[r_idx][seg_start:seg_end+1]
                
                # Mutation: perturb individual differences
                for _ in range(random.randint(1, 3)):
                    r_idx = random.randint(0, n - 1)
                    pos = random.randint(0, k - 1)
                    old = child[r_idx][pos]
                    # Bias toward smaller values to reduce scope
                    if random.random() < 0.6:
                        delta = random.randint(-2, 1)
                    else:
                        delta = random.randint(-3, 3)
                    new_val = max(1, old + delta)
                    child[r_idx][pos] = new_val
                
                new_pop.append(child)
            
            population = new_pop
            
            # Occasionally re-inject best solution with small perturbation
            if best_ind is not None and random.random() < 0.1:
                perturbed = [d[:] for d in best_ind]
                r_idx = random.randint(0, n - 1)
                pos = random.randint(0, k - 1)
                perturbed[r_idx][pos] = max(1, perturbed[r_idx][pos] + random.randint(-2, 2))
                if len(population) > 1:
                    population[-1] = perturbed
        
        # Convert best individual to rows
        best_rows = [row_from_diffs(d) for d in best_ind]
        actual_max = max(r[-1] for r in best_rows)
        return best_rows, actual_max

    # --- SA for final polishing ---
    def sa_optimize(initial_rows, time_limit, bound):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len

        row_diffs = [get_diffs(r) for r in rows]
        
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        def compute_cost():
            collisions = 0
            for d, c in diff_counts.items():
                if c > 1:
                    collisions += c - 1
            max_val = max(r[-1] for r in rows)
            return collisions * 10000 + max_val

        current_cost = compute_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = max(r[-1] for r in best_local_rows)

        T = 50.0
        cooling = 0.9995
        start_time = time.time()

        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)
            
            row = rows[r_idx]
            old_val = row[i_idx]
            
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else bound + 10
            upper = min(upper, bound + 20)
            
            if lower > upper:
                continue
                
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue
            
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs
            
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_cost = compute_cost()
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = max(r[-1] for r in best_local_rows)
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if random.random() < 0.5:
                        current_cost = new_cost
                    else:
                        row[i_idx] = old_val
                        for d in new_diffs:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs
                else:
                    if delta < 1000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    
                    if random.random() < p:
                        current_cost = new_cost
                    else:
                        row[i_idx] = old_val
                        for d in new_diffs:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs
            
            T *= cooling
            if T < 0.1:
                T = 0.1

        return best_local_rows, best_local_max, best_local_cost

    # Main Loop
    while time.time() < deadline - 5.0:
        # Convert best_rows to difference representation
        best_individual = [diffs_from_row(r) for r in best_rows]
        
        # Run GA
        ga_time = min(8.0, deadline - time.time() - 5.0)
        if ga_time > 1.0:
            ga_rows, ga_max = ga_optimize(best_individual, ga_time, best_max)
            
            # Verify GA result
            all_diffs = []
            for r in ga_rows:
                for d in get_diffs(r):
                    all_diffs.append(d)
            if len(all_diffs) == len(set(all_diffs)):
                if ga_max < best_max:
                    best_rows = ga_rows
                    best_max = ga_max
        
        # Polish with SA
        sa_time = min(5.0, deadline - time.time() - 2.0)
        if sa_time > 0.5:
            init_rows = [r[:] for r in best_rows]
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                row = init_rows[r_idx]
                lower = row[i_idx - 1] + 1
                upper = row[i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 5
                if lower <= upper:
                    init_rows[r_idx][i_idx] = random.randint(lower, upper)
            
            local_rows, local_max, local_cost = sa_optimize(init_rows, sa_time, best_max)
            
            if local_cost < 10000:
                if local_max < best_max:
                    best_rows = local_rows
                    best_max = local_max

    return best_rows
