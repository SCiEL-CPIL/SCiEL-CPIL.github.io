import time
import random
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within each row are globally distinct.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # For (7,5): 7 rows, 6 entries each, 15 diffs per row, 105 total diffs
    # Counting bound: need at least 105 distinct positive differences
    # The best known is 112 (Shehadeh et al. 2025)
    
    # Phase 1: Greedy construction to get an initial valid solution
    def greedy_construct():
        rows = []
        used_diffs = set()
        for r in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                x = 1
                while True:
                    new_diffs = set()
                    ok = True
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in new_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows
    
    best_rows = greedy_construct()
    best_val = max(max(r) for r in best_rows)
    
    # Phase 2: Simulated annealing with row-level operations
    # We use a more sophisticated approach:
    # - Represent each row by its k+1 elements
    # - Moves: change one element, swap two rows' elements, or rebuild a row
    
    def row_diffs(row):
        d = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d.add(row[j] - row[i])
        return d
    
    def global_max(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            d = row_diffs(row)
            if d & used:
                return False
            used |= d
        return True
    
    # Verify initial solution
    if not is_valid(best_rows):
        return best_rows  # Should not happen, but safety
    
    # Phase 3: Local search / simulated annealing
    # Key insight: focus on reducing the maximum entry
    # Strategy: identify rows with large entries and try to reduce them
    
    current_rows = [r[:] for r in best_rows]
    current_val = global_max(current_rows)
    
    # Precompute diffs for each row
    current_diffs = [row_diffs(r) for r in current_rows]
    all_diffs = set()
    for d in current_diffs:
        all_diffs |= d
    
    T = 30.0
    T_min = 0.01
    cool_rate = 0.99999
    
    no_improve_count = 0
    max_no_improve = 30000
    
    while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
        # Choose a row to modify - prioritize rows with large entries
        row_maxes = [max(r) for r in current_rows]
        # Weight selection by how far above average the row's max is
        avg_max = sum(row_maxes) / n_rows
        weights = [max(1, (row_maxes[i] - avg_max) + 1) for i in range(n_rows)]
        total_w = sum(weights)
        r = random.random() * total_w
        cumulative = 0
        ri = 0
        for i in range(n_rows):
            cumulative += weights[i]
            if r <= cumulative:
                ri = i
                break
        
        # Choose type of move
        move_type = random.random()
        
        if move_type < 0.5:
            # Single element change: try to reduce an element
            pi = random.randint(1, k)  # Don't change the first element (must be 0)
            old_val = current_rows[ri][pi]
            
            # Try values below old_val
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else old_val
            
            if old_val <= min_allowed:
                no_improve_count += 1
                T = max(T_min, T * cool_rate)
                continue
            
            # Generate candidate values
            candidates = []
            # Focus on reducing
            for delta in range(1, min(30, old_val - min_allowed + 1)):
                candidates.append(old_val - delta)
            # Some random exploration
            for _ in range(3):
                v = random.randint(min_allowed, max_allowed)
                if v != old_val:
                    candidates.append(v)
            
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                # Check internal validity
                new_d = row_diffs(cand_row)
                if len(new_d) != len(cand_row) * (len(cand_row) - 1) // 2:
                    continue
                
                # Check cross-row conflicts
                other_diffs = all_diffs - current_diffs[ri]
                if new_d & other_diffs:
                    continue
                
                # Compute new global max
                new_row_max = max(cand_row)
                other_maxes = [max(current_rows[j]) for j in range(n_rows) if j != ri]
                new_global = max([new_row_max] + other_maxes)
                
                delta_val = new_global - current_val
                
                # Acceptance criterion
                accept = False
                if delta_val < 0:
                    accept = True
                elif delta_val == 0:
                    if random.random() < 0.05:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta_val / T)
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_diffs -= current_diffs[ri]
                    all_diffs |= new_d
                    current_rows[ri] = cand_row
                    current_diffs[ri] = new_d
                    current_val = new_global
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve_count += 1
        
        elif move_type < 0.8:
            # Row rebuild: try to reconstruct a row with smaller max
            ri = random.randint(0, n_rows - 1)
            other_diffs = all_diffs - current_diffs[ri]
            
            # Try to build a new row with smaller max
            target_max = max(current_rows[ri]) - random.randint(1, 15)
            if target_max < k:  # Minimum possible is roughly k*(k+1)/2 for tight packing
                target_max = k * (k + 1) // 2
            
            new_row = None
            # Greedy with randomization
            for attempt in range(5):
                row = [0]
                success = True
                for pos in range(1, k + 1):
                    found = False
                    start = row[-1] + 1
                    end = min(target_max, start + 20)
                    candidates = list(range(start, end + 1))
                    random.shuffle(candidates)
                    for x in candidates:
                        new_d = set()
                        ok = True
                        for a in row:
                            d = x - a
                            if d in other_diffs or d in new_d:
                                ok = False
                                break
                            new_d.add(d)
                        if ok:
                            row.append(x)
                            found = True
                            break
                    if not found:
                        success = False
                        break
                if success:
                    new_row = row
                    break
            
            if new_row is not None:
                new_d = row_diffs(new_row)
                new_row_max = max(new_row)
                other_maxes = [max(current_rows[j]) for j in range(n_rows) if j != ri]
                new_global = max([new_row_max] + other_maxes)
                
                delta_val = new_global - current_val
                
                accept = False
                if delta_val < 0:
                    accept = True
                elif T > T_min:
                    prob = math.exp(-delta_val / T)
                    if random.random() < prob:
                        accept = True
                
                if accept:
                    all_diffs -= current_diffs[ri]
                    all_diffs |= new_d
                    current_rows[ri] = new_row
                    current_diffs[ri] = new_d
                    current_val = new_global
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        else:
            # Swap elements between two rows
            ri = random.randint(0, n_rows - 1)
            rj = random.randint(0, n_rows - 1)
            if ri == rj:
                no_improve_count += 1
                T = max(T_min, T * cool_rate)
                continue
            
            pi = random.randint(1, k)
            pj = random.randint(1, k)
            
            old_ri_pi = current_rows[ri][pi]
            old_rj_pj = current_rows[rj][pj]
            
            # Swap
            current_rows[ri][pi], current_rows[rj][pj] = old_rj_pj, old_ri_pi
            
            # Check validity of both rows
            valid_ri = True
            d_ri = set()
            row_ri = current_rows[ri]
            for i in range(len(row_ri)):
                for j in range(i + 1, len(row_ri)):
                    d = row_ri[j] - row_ri[i]
                    if d in d_ri:
                        valid_ri = False
                        break
                    d_ri.add(d)
                if not valid_ri:
                    break
            # Check strict increasing
            for i in range(1, len(row_ri)):
                if row_ri[i] <= row_ri[i-1]:
                    valid_ri = False
                    break
            
            valid_rj = True
            d_rj = set()
            row_rj = current_rows[rj]
            for i in range(len(row_rj)):
                for j in range(i + 1, len(row_rj)):
                    d = row_rj[j] - row_rj[i]
                    if d in d_rj:
                        valid_rj = False
                        break
                    d_rj.add(d)
                if not valid_rj:
                    break
            for i in range(1, len(row_rj)):
                if row_rj[i] <= row_rj[i-1]:
                    valid_rj = False
                    break
            
            if not (valid_ri and valid_rj):
                # Undo
                current_rows[ri][pi], current_rows[rj][pj] = old_ri_pi, old_rj_pj
                no_improve_count += 1
                T = max(T_min, T * cool_rate)
                continue
            
            # Check cross conflicts
            other_diffs = all_diffs - current_diffs[ri] - current_diffs[rj]
            if (d_ri & d_rj) or (d_ri & other_diffs) or (d_rj & other_diffs):
                # Undo
                current_rows[ri][pi], current_rows[rj][pj] = old_ri_pi, old_rj_pj
                no_improve_count += 1
                T = max(T_min, T * cool_rate)
                continue
            
            # Compute new global max
            new_global = max(max(current_rows[ri]), max(current_rows[rj]),
                           max(max(current_rows[j]) for j in range(n_rows) if j != ri and j != rj))
            
            delta_val = new_global - current_val
            
            accept = False
            if delta_val < 0:
                accept = True
            elif T > T_min:
                prob = math.exp(-delta_val / T)
                if random.random() < prob:
                    accept = True
            
            if accept:
                all_diffs -= current_diffs[ri]
                all_diffs -= current_diffs[rj]
                all_diffs |= d_ri
                all_diffs |= d_rj
                current_diffs[ri] = d_ri
                current_diffs[rj] = d_rj
                current_val = new_global
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Undo
                current_rows[ri][pi], current_rows[rj][pj] = old_ri_pi, old_rj_pj
                no_improve_count += 1
        
        T = max(T_min, T * cool_rate)
    
    # Final validation
    if not is_valid(best_rows):
        # Fallback to greedy
        return greedy_construct()
    
    return best_rows
