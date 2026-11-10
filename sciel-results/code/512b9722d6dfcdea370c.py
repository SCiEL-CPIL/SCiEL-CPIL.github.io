import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    
    Strategy: Simulated Annealing
    1. Build a large valid baseline using greedy.
    2. Use SA to shrink the scope by randomly moving elements to smaller positions.
    """
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    def build_baseline(n_rows, k, cap=10000):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                if x > cap:
                    return None
            rows.append(row)
        return rows

    baseline = build_baseline(n_rows, k)
    if baseline is None:
        # Fallback
        return [[0] + list(range(1, k + 1)) for _ in range(n_rows)]

    best_rows = [r[:] for r in baseline]
    best_score = max(max(r) for r in best_rows)
    current_rows = [r[:] for r in baseline]
    current_score = best_score

    def compute_diffs(rows):
        global_diffs = set()
        row_diffs = []
        for r in range(len(rows)):
            rd = set()
            row = rows[r]
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    rd.add(d)
            row_diffs.append(rd)
            global_diffs |= rd
        return global_diffs, row_diffs

    global_diffs, row_diffs = compute_diffs(current_rows)
    
    rng = random.Random(42)
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    # Helper to revert a move
    def revert_move(rows, r, p, old_val, new_val, old_diffs, new_diffs, global_diffs, row_diffs):
        # Remove new_diffs from global
        for d in new_diffs:
            global_diffs.discard(d)
        # Add old_diffs to global
        for d in old_diffs:
            global_diffs.add(d)
        # Update row_diffs
        row_diffs[r] = (row_diffs[r] - new_diffs) | old_diffs
        # Revert row value
        rows[r][p] = old_val

    steps = 0
    no_improve_count = 0
    last_best_score = best_score
    
    while time.time() < deadline:
        r = rng.randint(0, n_rows - 1)
        p = rng.randint(1, k)
        
        row = current_rows[r]
        old_val = row[p]
        
        # Determine range
        low = row[p-1] + 1
        if p < k:
            high = row[p+1] - 1
        else:
            # For last element, allow moving up a bit, but mostly down
            high = old_val + 5
            
        if low > high:
            continue
            
        # Bias towards smaller values to encourage shrinking
        # Sample uniformly, but if we want to shrink, we might want to pick from [low, old_val] more often
        # Let's just pick uniformly from [low, high]
        new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
            
        # Compute old_diffs and new_diffs
        old_diffs = set()
        for i in range(p):
            old_diffs.add(old_val - row[i])
        for j in range(p + 1, k + 1):
            old_diffs.add(row[j] - old_val)
            
        new_diffs = set()
        for i in range(p):
            new_diffs.add(new_val - row[i])
        for j in range(p + 1, k + 1):
            new_diffs.add(row[j] - new_val)
            
        # Check validity
        # 1. Internal row conflict
        other_row_diffs = row_diffs[r] - old_diffs
        if new_diffs & other_row_diffs:
            continue
            
        # 2. Global conflict
        # Check if new_diffs intersects with (global_diffs - old_diffs)
        # We do this by temporarily removing old_diffs from a copy? No, that's slow.
        # We check: if any d in new_diffs is in global_diffs and d is NOT in old_diffs.
        # Wait, if d is in old_diffs, it's being removed. So it's available.
        # So we need: (new_diffs & global_diffs) - old_diffs == empty.
        
        conflict = False
        for d in new_diffs:
            if d in global_diffs and d not in old_diffs:
                conflict = True
                break
        
        if conflict:
            continue
            
        # Valid move. Apply tentatively.
        # Remove old_diffs from global
        for d in old_diffs:
            global_diffs.discard(d)
        # Add new_diffs to global
        for d in new_diffs:
            global_diffs.add(d)
        # Update row_diffs
        row_diffs[r] = (row_diffs[r] - old_diffs) | new_diffs
        # Update row
        row[p] = new_val
        
        # Calculate new score
        new_score = max(max(rw) for rw in current_rows)
        
        # Acceptance
        if new_score <= current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows = [rw[:] for rw in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            delta = new_score - current_score
            prob = pow(2.718281828, -delta / T)
            if rng.random() < prob:
                current_score = new_score
                no_improve_count += 1
            else:
                # Revert
                revert_move(current_rows, r, p, old_val, new_val, old_diffs, new_diffs, global_diffs, row_diffs)
                no_improve_count += 1
                
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Restart if stuck
        if no_improve_count > 1000:
            T = 10.0
            no_improve_count = 0
            # Optionally re-randomize? No, just reheat.
            
        steps += 1

    return best_rows
# EVOLVE-BLOCK-END
