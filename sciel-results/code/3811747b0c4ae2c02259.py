"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # 10s safety margin
    
    # Total number of differences needed: n_rows * k(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Theoretical lower bound: need at least total_diffs distinct positive integers
    # so max >= ceil((-1 + sqrt(1 + 8*total_diffs))/2)
    lb = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_result = None
    best_scope = float('inf')
    
    # Strategy 1: Greedy with backtracking (limited depth)
    # Strategy 2: Local search / hill climbing
    # Strategy 3: Simulated annealing on the gap representation
    
    # Start with a greedy solution
    def greedy_construct():
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
        return rows
    
    # Try greedy first
    try:
        rows = greedy_construct()
        scope = max(max(r) for r in rows)
        best_result = rows
        best_scope = scope
    except:
        pass
    
    # If we have time, try to improve with local search
    if best_result is not None and time.time() < DEADLINE:
        # Convert to gap representation for local search
        # Each row has k gaps: g_i = a_{i+1} - a_i
        # The differences are cumulative sums of consecutive gaps
        
        def row_to_gaps(row):
            return [row[i+1] - row[i] for i in range(k)]
        
        def gaps_to_row(gaps):
            row = [0]
            for g in gaps:
                row.append(row[-1] + g)
            return row
        
        def get_all_diffs(row):
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
            return diffs
        
        def is_valid(rows):
            all_diffs = set()
            for row in rows:
                row_diffs = get_all_diffs(row)
                if row_diffs & all_diffs:
                    return False
                all_diffs |= row_diffs
            return True
        
        def scope_of(rows):
            return max(max(r) for r in rows)
        
        # Local search: try to reduce scope by adjusting gaps
        current_rows = [list(r) for r in best_result]
        current_scope = best_scope
        
        # Try random restarts with different random seeds
        rng = random.Random(42)
        
        # Simulated annealing on gap vectors
        def anneal(initial_rows, max_iters=50000, deadline=None):
            nonlocal current_rows, current_scope
            
            if deadline is None:
                deadline = DEADLINE
            
            # Represent as list of gap lists
            gap_lists = [row_to_gaps(r) for r in initial_rows]
            
            # Current energy = scope
            cur_scope = scope_of(initial_rows)
            best_local = [list(r) for r in initial_rows]
            best_local_scope = cur_scope
            
            T = 10.0
            T_min = 0.1
            cooling = 0.999
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random row and a random gap to modify
                ri = rng.randint(0, n_rows - 1)
                gi = rng.randint(0, k - 1)
                
                old_gap = gap_lists[ri][gi]
                # Try changing the gap by a small amount
                delta = rng.choice([-2, -1, 1, 2, 3, -3])
                new_gap = old_gap + delta
                if new_gap <= 0:
                    continue
                
                # Apply change
                gap_lists[ri][gi] = new_gap
                new_rows = [gaps_to_row(gl) for gl in gap_lists]
                
                # Check validity
                if not is_valid(new_rows):
                    # Revert
                    gap_lists[ri][gi] = old_gap
                    continue
                
                new_scope = scope_of(new_rows)
                delta_e = new_scope - cur_scope
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
                    cur_scope = new_scope
                    if new_scope < best_local_scope:
                        best_local_scope = new_scope
                        best_local = [list(r) for r in new_rows]
                else:
                    # Revert
                    gap_lists[ri][gi] = old_gap
                
                T = max(T * cooling, T_min)
            
            return best_local, best_local_scope
        
        # Try a few annealing runs
        for attempt in range(5):
            if time.time() > DEADLINE:
                break
            
            # Start from current best or a fresh greedy
            if attempt == 0:
                start_rows = [list(r) for r in best_result]
            else:
                # Perturb the best solution
                perturbed = [list(r) for r in best_result]
                for _ in range(k):
                    ri = rng.randint(0, n_rows - 1)
                    gi = rng.randint(0, k - 1)
                    old = row_to_gaps(perturbed[ri])[gi]
                    new = max(1, old + rng.randint(-2, 2))
                    gaps = row_to_gaps(perturbed[ri])
                    gaps[gi] = new
                    perturbed[ri] = gaps_to_row(gaps)
                if not is_valid(perturbed):
                    perturbed = [list(r) for r in best_result]
                start_rows = perturbed
            
            local_best, local_scope = anneal(start_rows, max_iters=20000, deadline=DEADLINE)
            
            if local_scope < best_scope:
                best_scope = local_scope
                best_result = local_best
                current_rows = [list(r) for r in local_best]
                current_scope = best_scope
    
    # Final validation
    if best_result is not None:
        all_diffs = set()
        valid = True
        for row in best_result:
            if len(row) != k + 1:
                valid = False
                break
            if row[0] != 0:
                valid = False
                break
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        valid = False
                        break
                    all_diffs.add(d)
                if not valid:
                    break
            if not valid:
                break
            if not all(row[i] < row[i+1] for i in range(len(row)-1)):
                valid = False
                break
        if not valid:
            # Fallback to greedy
            try:
                best_result = greedy_construct()
            except:
                pass
    
    return best_result if best_result is not None else greedy_construct()
# EVOLVE-BLOCK-END
