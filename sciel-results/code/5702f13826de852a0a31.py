import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    best_rows = None
    best_max = float('inf')
    
    # Helper to check validity and get max
    def get_max(rows):
        return max(max(row) for row in rows)
        
    # Helper to check if a row can be extended with x
    def is_valid_extension(row, x, used_diffs):
        # x must be greater than row[-1]
        if x <= row[-1]:
            return False
        new_diffs = {x - a for a in row}
        if new_diffs & used_diffs:
            return False
        return True

    # Randomized Greedy Construction
    def randomized_greedy():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Find valid candidates
                candidates = []
                x = row[-1] + 1
                # Limit search space to avoid infinite loops if stuck, 
                # though for this problem size it shouldn't get stuck easily
                # We can also randomize the selection
                found_any = False
                while True:
                    if is_valid_extension(row, x, used_diffs):
                        candidates.append(x)
                        # Stop collecting after a few candidates to keep it fast
                        if len(candidates) > 5:
                            break
                    x += 1
                    if x > 200: # Arbitrary cap, should be much lower
                        break
                    found_any = True
                    
                if not candidates:
                    return None # Failed to construct
                
                # Random selection: bias towards smaller
                # Pick the smallest with high prob, else random from candidates
                if random.random() < 0.7:
                    chosen = candidates[0]
                else:
                    chosen = random.choice(candidates)
                
                row.append(chosen)
                used_diffs |= {chosen - a for a in row[:-1]}
            rows.append(row)
        return rows

    # Greedy Reduction: Try to shrink rows
    def reduce_rows(rows):
        # Work on a copy
        rows = [r[:] for r in rows]
        used_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        # Try to reduce the largest element in each row
        # Sort rows by max value descending
        row_indices = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for idx in row_indices:
                row = rows[idx]
                # Try to reduce the last element first, then others?
                # Actually, reducing any element requires re-checking all diffs in that row
                # Let's try to reduce the last element (max) of the most expensive rows
                for pos in range(k, 0, -1):
                    # Try to reduce row[pos]
                    old_val = row[pos]
                    # Remove diffs involving row[pos]
                    diffs_to_remove = set()
                    for j in range(pos):
                        diffs_to_remove.add(row[pos] - row[j])
                    # Also diffs with later elements? No, row is increasing, so pos is not the last necessarily?
                    # Wait, row is increasing. row[pos] is not necessarily the max if pos < k.
                    # But we want to minimize the global max. The global max is in some row's last element.
                    # Let's focus on the last element of the rows with the largest max.
                    
                    # This simple reduction is tricky because reducing an element changes diffs with ALL other elements in the row.
                    # Let's just try to reduce the last element of the row with the current global max.
                    pass 
        # This local search is complex to implement correctly and efficiently in this short time.
        # Let's stick to multiple randomized restarts.
        return rows

    # Main Loop
    iterations = 0
    while time.time() < deadline:
        rows = randomized_greedy()
        if rows:
            current_max = get_max(rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = rows
                # Optional: If we find a very good one, try to polish it?
                # For now, just keep restarting.
        iterations += 1
        
    if best_rows is None:
        # Fallback to deterministic greedy
        return construct_fallback(n_rows, k)
        
    return best_rows

def construct_fallback(n_rows, k):
    # Same as seed
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
