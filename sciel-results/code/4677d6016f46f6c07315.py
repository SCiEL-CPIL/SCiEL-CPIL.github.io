import time
import random

def construct(n_rows, k):
    deadline = time.time() + 140  # Safety margin for 150s limit
    best_rows = None
    best_scope = float('inf')
    
    # Helper to calculate scope
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(r) for r in rows)

    # Helper to build used set
    def build_used(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used.add(r[j] - r[i])
        return used

    # Randomized Greedy Construction
    def randomized_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                candidates = []
                x = row[-1] + 1
                # Limit search window to avoid infinite loops if stuck, though unlikely
                # We just need a few candidates to choose from
                while len(candidates) < 10:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    x += 1
                    if x > 10000: # Fallback if something goes wrong
                        break
                if not candidates:
                    return None # Failed
                # Pick randomly from the smallest candidates to keep scope low
                # but allow some randomness
                choice = random.choice(candidates)
                row.append(choice)
                used.update({choice - a for a in row[:-1]})
            rows.append(row)
        return rows

    # Local Optimization
    def local_optimize(rows):
        rows = [r[:] for r in rows]
        used = build_used(rows)
        improved = True
        # Limit iterations to avoid getting stuck in a specific optimization loop
        while improved and time.time() < deadline:
            improved = False
            for r_idx in range(len(rows)):
                r = rows[r_idx]
                # Pre-calculate row diffs for quick removal? 
                # Or just recalculate the specific diffs involved.
                for j in range(len(r) - 1, 0, -1):
                    old_val = r[j]
                    min_val = r[j-1] + 1
                    max_val = (r[j+1] - 1) if j < len(r) - 1 else old_val - 1
                    
                    if max_val < min_val:
                        continue
                        
                    # Try to decrease old_val
                    for new_val in range(max_val, min_val - 1, -1):
                        # Calculate new diffs involving j
                        new_diffs = set()
                        for i in range(j):
                            new_diffs.add(new_val - r[i])
                        for i in range(j + 1, len(r)):
                            new_diffs.add(r[i] - new_val)
                        
                        # Old diffs involving j
                        old_diffs = set()
                        for i in range(j):
                            old_diffs.add(old_val - r[i])
                        for i in range(j + 1, len(r)):
                            old_diffs.add(r[i] - old_val)
                        
                        # Check if new_diffs conflict with (used - old_diffs)
                        # used currently contains old_diffs.
                        # So we check if new_diffs & (used - old_diffs) is empty.
                        # Equivalent to: (new_diffs & used) - old_diffs is empty?
                        # No. new_diffs must not be in used, except for the ones we are removing.
                        # So: new_diffs & (used - old_diffs) == empty.
                        
                        conflict = new_diffs & (used - old_diffs)
                        if not conflict:
                            # Valid move
                            used -= old_diffs
                            used |= new_diffs
                            r[j] = new_val
                            improved = True
                            break # Break new_val loop, try next j or row
                    if improved:
                        break # Break j loop, restart improved loop or next row?
                        # Restarting improved loop is safer to find global improvements
                        # But breaking to next row is faster. Let's break to next row.
                if improved:
                    break # Break r_idx loop, restart while loop

        return rows

    # Main Loop
    while time.time() < deadline:
        rows = randomized_greedy()
        if rows is None:
            continue
            
        # Local Optimize
        opt_rows = local_optimize(rows)
        scope = get_scope(opt_rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = opt_rows
            
        # If we hit the theoretical lower bound, we can stop
        # Counting bound for n_rows=4, k=7 is 112.
        # If n_rows and k are different, we don't know the bound.
        # But 112 is for the specific task instance mentioned.
        # The evaluator passes n_rows and k.
        # We shouldn't hardcode 112 unless we are sure.
        # However, if we find a very small scope, it's likely optimal.
        
    if best_rows is None:
        # Fallback: simple greedy
        return construct_fallback(n_rows, k)
        
    return best_rows

def construct_fallback(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
