import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def greedy_construct(seed_val):
        """Construct rows using greedy with a compact strategy."""
        random.seed(seed_val)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                # Try to find the smallest x > last such that all (x - a) for a in row are not in used
                # and x - a are distinct among themselves (which they are if x > a and a distinct)
                # Actually, we just need x - a not in used for any a in row.
                
                # Start searching from last + 1
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    x += 1
                
                if not found:
                    return None, float('inf')
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope

    # 1. Get a valid starting point
    # Try a few seeds to see if we can get a slightly better initial scope, 
    # though greedy is deterministic given the "smallest valid" rule.
    # The "smallest valid" greedy is often quite good.
    initial_rows, initial_scope = greedy_construct(0)
    
    if initial_rows is None:
        # Fallback: extremely sparse construction
        used = set()
        initial_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            initial_rows.append(row)
        initial_scope = max(max(r) for r in initial_rows)

    best_rows = [list(r) for r in initial_rows]
    best_scope = initial_scope
    current_rows = [list(r) for r in initial_rows]
    current_used = get_used_diffs(current_rows)

    # 2. Local Search: Iterative Compaction
    # Strategy: Pick a random row, pick a random element (not 0), try to move it to a smaller value.
    # If valid, accept. This reduces the scope if the moved element was the max, or just improves density.
    
    def try_move_element(rows, used_set, r_idx, p_idx, new_val):
        """
        Attempt to move element at rows[r_idx][p_idx] to new_val.
        Returns True if successful and updates rows/used_set, False otherwise.
        """
        old_val = rows[r_idx][p_idx]
        if old_val == new_val:
            return True
            
        # Check if new_val is already in the row
        if new_val in rows[r_idx]:
            return False
        
        # Check conflicts with other elements in the same row
        row = rows[r_idx]
        new_diffs = []
        for i, a in enumerate(row):
            if i == p_idx:
                continue
            d = abs(new_val - a)
            if d in used_set:
                # Check if this difference is ONLY created by the pair we are modifying
                # Actually, if d is in used_set, it means some pair (u,v) has u-v=d.
                # If (u,v) is (old_val, a) or (a, old_val), then it's "reused" which is fine?
                # No, differences must be distinct OVER ALL ROWS.
                # So if d is in used_set, and it was created by (old_val, a), then after removing old_val,
                # d is no longer in used_set (if we remove it first).
                # But here we are checking BEFORE removing.
                # So we need to be careful.
                
                # Let's remove old_val's diffs first to make the check clean.
                pass 
        
        # Better approach: Remove old_val, check new_val, add new_val.
        
        # 1. Remove old_val from row and used_set
        # We need to know the index. p_idx is given.
        # But rows[r_idx] is a list. We pop p_idx.
        
        # Let's implement a safer swap/move logic.
        
        # Remove diffs involving old_val
        for a in row:
            if a != old_val:
                d = abs(a - old_val)
                if d in used_set:
                    # Decrement count? We are using a set, so we just discard.
                    # But wait, if another pair in the SAME row has the same diff, discarding is wrong.
                    # In a valid set, all diffs are unique. So d appears exactly once.
                    # So discard is safe.
                    used_set.discard(d)
        
        # Temporarily remove old_val from row
        rows[r_idx].pop(p_idx)
        
        # 2. Check if new_val is valid
        valid = True
        new_diffs = []
        for a in rows[r_idx]:
            d = abs(a - new_val)
            if d in used_set:
                valid = False
                break
            new_diffs.append(d)
            
        if valid:
            # 3. Insert new_val into correct position to keep sorted
            # Find insertion point
            pos = 0
            while pos < len(rows[r_idx]) and rows[r_idx][pos] < new_val:
                pos += 1
            rows[r_idx].insert(pos, new_val)
            
            # 4. Add new diffs
            for d in new_diffs:
                used_set.add(d)
                
            return True
        else:
            # Revert: put old_val back
            pos = 0
            while pos < len(rows[r_idx]) and rows[r_idx][pos] < old_val:
                pos += 1
            rows[r_idx].insert(pos, old_val)
            
            # Re-add old diffs
            for a in rows[r_idx]:
                if a != old_val:
                    d = abs(a - old_val)
                    used_set.add(d)
                    
            return False

    # Main Local Search Loop
    while time.time() < deadline - 2:
        improved = False
        
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick a random element to move (excluding 0)
        # Bias towards larger elements to reduce scope?
        # Or just random.
        p_idx = random.randint(1, len(row) - 1)
        
        old_val = row[p_idx]
        
        # Try to find a smaller valid value for this element
        # Search range: [1, old_val - 1]
        # We want to minimize the max, so moving large elements down is key.
        
        # Optimization: If this element is not the max of the row, moving it down might not reduce global scope,
        # but it frees up differences for others.
        
        found_better = False
        best_new_val = None
        
        # Try values in a random order or descending order?
        # Descending order from old_val - 1 down to 1 is good for finding "just smaller" valid spots.
        # But it can be slow.
        # Let's try a few random candidates in [1, old_val - 1].
        
        if old_val > 1:
            # Try a few random candidates
            candidates = random.sample(range(1, old_val), min(50, old_val - 1))
            # Sort them descending? No, random is fine for diversity.
            # But we want to find ANY valid smaller value.
            
            for cand in candidates:
                if try_move_element(current_rows, current_used, r_idx, p_idx, cand):
                    found_better = True
                    best_new_val = cand
                    break
            
            # If random failed, try a linear scan from old_val - 1 downwards for a limited range
            if not found_better:
                for cand in range(old_val - 1, max(old_val - 50, 0), -1):
                    if try_move_element(current_rows, current_used, r_idx, p_idx, cand):
                        found_better = True
                        best_new_val = cand
                        break

        if found_better:
            # Calculate new scope
            new_scope = max(max(r) for r in current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
            improved = True
        else:
            # If we couldn't improve, maybe try a different move type?
            # For now, just loop.
            pass

    # Final check
    if best_rows is None:
        best_rows = initial_rows
        
    return best_rows
