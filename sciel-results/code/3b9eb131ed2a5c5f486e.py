"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct difference triangle set: n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, subject to all pairwise differences within each row being
    distinct across all rows.
    """
    deadline = time.time() + 140  # 140s out of 150s budget
    
    # Total number of differences per row: C(k+1, 2) = k*(k+1)/2
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Counting bound: need at least total_diffs distinct positive integers
    # The smallest possible max entry satisfies: max_entry * (max_entry + 1) / 2 >= total_diffs
    # So max_entry >= ceil((-1 + sqrt(1 + 8*total_diffs)) / 2)
    import math
    lower_bound = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    
    def greedy_construct():
        """Original greedy: smallest next mark with no collision."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = [x - a for a in row]
                if not any(d in used for d in new_diffs):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows
    
    def get_max_entry(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def randomized_greedy(seed):
        """Greedy with randomization in tie-breaking and occasional larger jumps."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a range of candidates starting from x+1
                found = False
                # With some probability, skip ahead a bit
                start_offset = 0
                if rng.random() < 0.3:
                    start_offset = rng.randint(1, 5)
                
                for trial in range(20):
                    candidate = x + 1 + start_offset + trial
                    new_diffs = [candidate - a for a in row]
                    if not any(d in used for d in new_diffs):
                        row.append(candidate)
                        used.update(new_diffs)
                        x = candidate
                        found = True
                        break
                if not found:
                    # Fallback: linear scan
                    candidate = x + 1
                    while True:
                        new_diffs = [candidate - a for a in row]
                        if not any(d in used for d in new_diffs):
                            row.append(candidate)
                            used.update(new_diffs)
                            x = candidate
                            break
                        candidate += 1
            rows.append(row)
        return rows
    
    def simulated_annealing(initial_rows, seed, max_iters=50000):
        """
        Simulated annealing that can navigate through invalid states.
        State: list of rows. Move: change one entry in one row.
        Energy: number of collisions + lambda * (max_entry - lower_bound)^2
        """
        rng = random.Random(seed)
        rows = [list(r) for r in initial_rows]
        
        def compute_energy(rows):
            """Compute energy: collisions + penalty for large max entry."""
            diff_count = {}
            max_e = 0
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        diff_count[d] = diff_count.get(d, 0) + 1
                        if row[j] > max_e:
                            max_e = row[j]
            collisions = sum(c - 1 for c in diff_count.values() if c > 1)
            scope_penalty = max(0, max_e - lower_bound)
            return collisions * 100 + scope_penalty * scope_penalty, max_e, collisions
        
        best_energy, best_max, _ = compute_energy(rows)
        best_rows = [list(r) for r in rows]
        
        T = 50.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random row and position to modify
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # don't change position 0 (always 0)
            
            old_val = rows[r_idx][pos]
            # Propose a new value: random walk
            step = rng.randint(-3, 3)
            new_val = old_val + step
            
            # Ensure row remains strictly increasing
            if pos > 0 and new_val <= rows[r_idx][pos - 1]:
                continue
            if pos < k and new_val >= rows[r_idx][pos + 1]:
                continue
            if new_val <= 0:
                continue
            
            rows[r_idx][pos] = new_val
            new_energy, new_max, new_collisions = compute_energy(rows)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_max = new_max
                best_rows = [list(r) for r in rows]
            
            # Accept or reject
            delta_e = new_energy - compute_energy([list(r) for r in rows])
            # Recompute properly
            old_rows = [list(r) for r in rows]
            old_rows[r_idx][pos] = old_val
            old_energy, _, _ = compute_energy(old_rows)
            delta_e = new_energy - old_energy
            
            if delta_e < 0 or rng.random() < pow(2.718281828, -delta_e / max(T, 1e-10)):
                # Accept
                pass
            else:
                # Reject
                rows[r_idx][pos] = old_val
            
            T *= cooling
        
        return best_rows, best_max, best_energy
    
    def local_search(rows, seed, max_iters=20000):
        """Local search that only accepts valid states, trying to reduce max entry."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        
        def compute_diffs(rows):
            diffs = []
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        diffs.append(row[j] - row[i])
            return diffs
        
        def is_valid_fast(diffs):
            return len(set(diffs)) == len(diffs)
        
        current_diffs = compute_diffs(rows)
        current_max = get_max_entry(rows)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to reduce a large entry
            # Find the row with the max entry
            max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
            max_pos = max(range(k + 1), key=lambda j: rows[max_row_idx][j])
            
            # Try to decrease this entry
            old_val = rows[max_row_idx][max_pos]
            improved = False
            for new_val in range(old_val - 1, max(rows[max_row_idx][max_pos - 1] + 1 if max_pos > 0 else 0, old_val - 10), -1):
                if max_pos > 0 and new_val <= rows[max_row_idx][max_pos - 1]:
                    break
                if max_pos < k and new_val >= rows[max_row_idx][max_pos + 1]:
                    continue
                rows[max_row_idx][max_pos] = new_val
                new_diffs = compute_diffs(rows)
                if is_valid_fast(new_diffs):
                    new_max = get_max_entry(rows)
                    if new_max < current_max:
                        current_max = new_max
                        current_diffs = new_diffs
                        improved = True
                        break
                rows[max_row_idx][max_pos] = old_val
            
            if not improved:
                # Try random moves that maintain validity
                r_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, k)
                old_val = rows[r_idx][pos]
                # Try small changes
                for step in [rng.randint(-2, 2) for _ in range(10)]:
                    new_val = old_val + step
                    if pos > 0 and new_val <= rows[r_idx][pos - 1]:
                        continue
                    if pos < k and new_val >= rows[r_idx][pos + 1]:
                        continue
                    if new_val <= 0:
                        continue
                    rows[r_idx][pos] = new_val
                    new_diffs = compute_diffs(rows)
                    if is_valid_fast(new_diffs):
                        new_max = get_max_entry(rows)
                        if new_max <= current_max:
                            current_diffs = new_diffs
                            if new_max < current_max:
                                current_max = new_max
                        break
                    rows[r_idx][pos] = old_val
        
        return rows, current_max
    
    # Phase 1: Get initial solutions
    best_rows = greedy_construct()
    best_max = get_max_entry(best_rows)
    
    # Try multiple randomized greedy seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        rows = randomized_greedy(seed)
        if is_valid(rows):
            m = get_max_entry(rows)
            if m < best_max:
                best_max = m
                best_rows = rows
    
    # Phase 2: Local search from best solution
    if time.time() < deadline:
        improved_rows, improved_max = local_search(best_rows, seed=42, max_iters=30000)
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows
    
    # Phase 3: Simulated annealing (can go through invalid states)
    if time.time() < deadline - 5:
        sa_rows, sa_max, sa_energy = simulated_annealing(best_rows, seed=123, max_iters=40000)
        if is_valid(sa_rows):
            if sa_max < best_max:
                best_max = sa_max
                best_rows = sa_rows
    
    # Final local search polish
    if time.time() < deadline - 2:
        final_rows, final_max = local_search(best_rows, seed=999, max_iters=20000)
        if final_max < best_max:
            best_max = final_max
            best_rows = final_rows
    
    # Ensure validity of final result
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct()
    
    return best_rows
# EVOLVE-BLOCK-END
