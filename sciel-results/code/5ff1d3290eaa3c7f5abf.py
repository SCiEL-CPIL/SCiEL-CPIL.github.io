import time
import random
import math
import sys

def construct(n_rows, k):
    row_len = k + 1
    # Deadline: 115 seconds to be safe against the 120s limit
    deadline = time.time() + 115.0
    
    # Helper functions
    
    def row_from_diffs(diffs):
        """Reconstruct row from consecutive differences."""
        row = [0]
        for d in diffs:
            row.append(row[-1] + d)
        return row

    def diffs_from_row(row):
        """Get consecutive differences from a row."""
        return [row[i+1] - row[i] for i in range(len(row)-1)]

    def get_pairwise_diffs(row):
        """Get all pairwise differences for a row."""
        L = len(row)
        diffs = []
        for i in range(L):
            for j in range(i+1, L):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_stats(individual):
        """
        individual: list of n_rows lists of k differences.
        Returns (collisions, max_scope, all_diffs_list)
        """
        diff_counts = {}
        max_scope = 0
        all_diffs = []
        for diffs in individual:
            row = row_from_diffs(diffs)
            if row[-1] > max_scope:
                max_scope = row[-1]
            pdiffs = get_pairwise_diffs(row)
            all_diffs.extend(pdiffs)
            for d in pdiffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        collisions = 0
        for c in diff_counts.values():
            if c > 1:
                collisions += c - 1
        return collisions, max_scope, all_diffs

    def is_valid(individual):
        collisions, _, _ = compute_stats(individual)
        return collisions == 0

    # 1. Greedy Baseline in Value Space
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
    baseline_diffs = [diffs_from_row(r) for r in baseline_rows]
    best_cols, best_scope, _ = compute_stats(baseline_diffs)
    
    best_individual = [d[:] for d in baseline_diffs]
    best_valid_scope = best_scope if best_cols == 0 else float('inf')
    best_valid_ind = best_individual if best_cols == 0 else None

    # 2. Genetic Algorithm in Difference Space
    def run_ga(initial_ind, time_limit):
        pop_size = 40
        n = n_rows
        k_diffs = k
        
        # Initialize population
        population = [initial_ind[:]]
        for _ in range(pop_size - 1):
            # Perturb initial
            ind = [d[:] for d in initial_ind]
            for _ in range(random.randint(2, 5)):
                r_idx = random.randint(0, n - 1)
                pos = random.randint(0, k_diffs - 1)
                old = ind[r_idx][pos]
                # Bias towards smaller values
                if random.random() < 0.7:
                    delta = random.randint(-2, 1)
                else:
                    delta = random.randint(-3, 3)
                new_val = max(1, old + delta)
                ind[r_idx][pos] = new_val
            population.append(ind)
        
        # Evaluate initial
        scored = []
        for ind in population:
            c, s, _ = compute_stats(ind)
            scored.append((c, s, ind))
        
        scored.sort(key=lambda x: (x[0], x[1]))
        curr_best_c, curr_best_s, curr_best_ind = scored[0]
        
        start_time = time.time()
        gen = 0
        stagnation_counter = 0
        max_stagnation = 50
        
        while time.time() - start_time < time_limit:
            gen += 1
            
            # Update global best
            if curr_best_c < best_cols or (curr_best_c == best_cols and curr_best_s < best_scope):
                best_cols = curr_best_c
                best_scope = curr_best_s
                best_individual = [d[:] for d in curr_best_ind]
                
                if best_cols == 0:
                    if best_scope < best_valid_scope:
                        best_valid_scope = best_scope
                        best_valid_ind = [d[:] for d in curr_best_ind]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            elif curr_best_c == best_cols and curr_best_s == best_scope:
                stagnation_counter += 1
            else:
                # If we didn't improve, but are exploring, reset stagnation if we found valid?
                # Actually, if best_cols > 0, we want to reduce collisions.
                pass

            # Selection: Tournament
            def select():
                a = random.randint(0, min(5, len(scored)-1))
                b = random.randint(0, min(5, len(scored)-1))
                sa, sb = scored[a], scored[b]
                if sa[0] < sb[0]: return sa[2]
                if sa[0] > sb[0]: return sb[2]
                if sa[1] <= sb[1]: return sa[2]
                return sb[2]
            
            new_pop = []
            
            # Elitism
            for i in range(min(5, len(scored))):
                new_pop.append([d[:] for d in scored[i][2]])
            
            # Crossover and Mutation
            while len(new_pop) < pop_size and time.time() - start_time < time_limit - 1.0:
                p1 = select()
                p2 = select()
                
                child = [d[:] for d in p1]
                
                # Crossover: Swap rows or segments
                for r_idx in range(n):
                    if random.random() < 0.4:
                        # Swap entire row
                        child[r_idx] = p2[r_idx][:]
                    elif random.random() < 0.3 and k_diffs >= 2:
                        # Swap segment
                        seg_start = random.randint(0, k_diffs - 2)
                        seg_end = random.randint(seg_start, k_diffs - 1)
                        child[r_idx][seg_start:seg_end+1] = p2[r_idx][seg_start:seg_end+1]
                
                # Mutation
                # Number of mutations proportional to population pressure?
                n_mut = random.randint(1, 3)
                for _ in range(n_mut):
                    r_idx = random.randint(0, n - 1)
                    pos = random.randint(0, k_diffs - 1)
                    old = child[r_idx][pos]
                    
                    # Adaptive mutation step
                    if best_cols == 0:
                        # Focus on reducing scope: try smaller diffs
                        step = random.randint(-3, 1)
                    else:
                        # Focus on reducing collisions: try varied steps
                        step = random.randint(-4, 4)
                    
                    new_val = max(1, old + step)
                    child[r_idx][pos] = new_val
                
                new_pop.append(child)
            
            population = new_pop
            
            # Evaluate new population
            scored = []
            for ind in population:
                c, s, _ = compute_stats(ind)
                scored.append((c, s, ind))
            
            scored.sort(key=lambda x: (x[0], x[1]))
            
            # Check for convergence/stagnation
            if scored[0][0] == curr_best_c and scored[0][1] == curr_best_s:
                stagnation_counter += 1
            else:
                curr_best_c, curr_best_s, curr_best_ind = scored[0]
                # Reset stagnation if we found something new, even if not better than global?
                # Let's just track if global best improved.
                pass
            
            if stagnation_counter > max_stagnation:
                # Inject random diversity
                for _ in range(pop_size // 2):
                    ind = [d[:] for d in best_individual]
                    for _ in range(random.randint(5, 10)):
                        r_idx = random.randint(0, n - 1)
                        pos = random.randint(0, k_diffs - 1)
                        old = ind[r_idx][pos]
                        new_val = max(1, old + random.randint(-5, 5))
                        ind[r_idx][pos] = new_val
                    # Replace worst half
                    scored.sort(key=lambda x: (x[0], x[1]))
                    population[-1] = ind
                    scored[-1] = (compute_stats(ind)[0], compute_stats(ind)[1], ind)
                    scored.sort(key=lambda x: (x[0], x[1]))
                stagnation_counter = 0

        return best_individual, best_cols, best_scope

    # 3. Local Search for Scope Reduction (only if valid)
    def local_search_reduce_scope(ind, time_limit):
        """
        Given a valid individual (0 collisions), try to reduce the maximum scope.
        """
        if not is_valid(ind):
            return ind, float('inf')
        
        curr_ind = [d[:] for d in ind]
        curr_cols, curr_scope, _ = compute_stats(curr_ind)
        best_ind = [d[:] for d in curr_ind]
        best_scope = curr_scope
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Find the row with the largest scope
            rows = [row_from_diffs(d) for d in curr_ind]
            max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            max_row = rows[max_idx]
            
            improved = False
            
            # Try to reduce the last element of the max row by adjusting its differences
            # We want to make the row "tighter"
            
            # Strategy: Try to reduce any difference in the max row by 1, check validity.
            # If valid, keep it. If not, try next.
            # This is slow. Better: Randomly pick a diff in max row, reduce it, check.
            
            # To speed up, we can try to reduce the largest differences first.
            diffs = curr_ind[max_idx]
            # Sort indices by diff value descending
            idxs = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)
            
            for pos in idxs:
                old_val = diffs[pos]
                if old_val <= 1:
                    continue
                
                # Try reducing by 1
                new_val = old_val - 1
                curr_ind[max_idx][pos] = new_val
                
                # Check validity efficiently?
                # Full check is O(N*K^2). With N=7, K=5, it's fast enough.
                c, s, _ = compute_stats(curr_ind)
                
                if c == 0:
                    # Valid!
                    if s < best_scope:
                        best_scope = s
                        best_ind = [d[:] for d in curr_ind]
                    improved = True
                    # Don't revert, keep the improvement and continue searching from here
                    break
                else:
                    # Revert
                    curr_ind[max_idx][pos] = old_val
            
            if not improved:
                # Try random small changes in other rows to allow the max row to shrink?
                # Or just break if we can't improve max row directly.
                # Let's try a few random perturbations in the max row
                found_any = False
                for _ in range(10):
                    pos = random.randint(0, k-1)
                    old_val = curr_ind[max_idx][pos]
                    if old_val <= 1: continue
                    # Try +/- 1 or 2
                    delta = random.choice([-1, -2, 1, 2])
                    new_val = max(1, old_val + delta)
                    if new_val == old_val: continue
                    
                    curr_ind[max_idx][pos] = new_val
                    c, s, _ = compute_stats(curr_ind)
                    
                    if c == 0:
                        if s < best_scope:
                            best_scope = s
                            best_ind = [d[:] for d in curr_ind]
                        found_any = True
                        break
                    else:
                        curr_ind[max_idx][pos] = old_val
                
                if not found_any:
                    # Try perturbing other rows to "make room" for a smaller max row?
                    # This is complex. For now, if we can't improve max row, 
                    # maybe we should stop or try a different row?
                    # Since we always pick the max row, if we can't shrink it, 
                    # we might be stuck. 
                    # Let's try to perturb the max row more aggressively.
                    pos = random.randint(0, k-1)
                    old_val = curr_ind[max_idx][pos]
                    new_val = max(1, old_val + random.randint(-3, 3))
                    if new_val != old_val:
                        curr_ind[max_idx][pos] = new_val
                        c, s, _ = compute_stats(curr_ind)
                        if c == 0:
                            if s < best_scope:
                                best_scope = s
                                best_ind = [d[:] for d in curr_ind]
                            # Accept even if scope didn't decrease? 
                            # If scope increased, we might want to revert to keep best.
                            # But if it's valid, it's a new state.
                            # Let's only accept if it doesn't increase scope? 
                            # Actually, moving to a valid state with same scope might allow further reduction later.
                            # Let's accept valid moves.
                        else:
                            curr_ind[max_idx][pos] = old_val
                            # If we are stuck, break out of local search for this round
                            # and let GA handle it.
                            break

        return best_ind, best_scope

    # Main Loop
    # Run GA for a portion of time, then Local Search
    ga_time = min(60.0, (deadline - time.time()) * 0.7)
    
    if ga_time > 5.0:
        best_ind, best_c, best_s = run_ga(baseline_diffs, ga_time)
        
        # Update global bests
        if best_c == 0:
            if best_s < best_valid_scope:
                best_valid_scope = best_s
                best_valid_ind = [d[:] for d in best_ind]
        else:
            # If GA didn't find valid, use the best invalid as starting point for next phase?
            # Or just use baseline.
            pass

    # If we have a valid solution, try to improve it with Local Search
    if best_valid_ind is not None:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 2.0:
            ls_ind, ls_scope = local_search_reduce_scope(best_valid_ind, remaining_time)
            if ls_scope < best_valid_scope:
                best_valid_scope = ls_scope
                best_valid_ind = ls_ind

    # Final Fallback: If no valid solution found, return the one with fewest collisions?
    # The task requires valid rows. "Valid when all positive differences... are distinct".
    # If we can't find a valid one, we should return the best we have? 
    # The evaluator says "an invalid result... scores 0". 
    # So we MUST return a valid result. 
    # The greedy baseline is always valid.
    
    if best_valid_ind is None:
        # Should not happen with greedy baseline, but just in case
        best_valid_ind = baseline_diffs
        best_valid_scope = max(row_from_diffs(d)[-1] for d in best_valid_ind)
        
    # Convert best valid individual back to rows
    final_rows = [row_from_diffs(d) for d in best_valid_ind]
    
    return final_rows

if __name__ == "__main__":
    # Quick test
    # t0 = time.time()
    # res = construct(7, 5)
    # t1 = time.time()
    # print(f"Time: {t1-t0}")
    # print(f"Max: {max(r[-1] for r in res)}")
    # # Check validity
    # diffs = []
    # for r in res:
    #     for i in range(len(r)):
    #         for j in range(i+1, len(r)):
    #             diffs.append(r[j]-r[i])
    # print(f"Valid: {len(diffs) == len(set(diffs))}")
    pass
