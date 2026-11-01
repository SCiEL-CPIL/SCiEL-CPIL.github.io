import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    row_len = k + 1
    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # We need a max scope to allocate the difference tracking array.
    # The best known is 134. We can try to go lower, but we need a buffer.
    # If we start with a high scope and minimize, we need to know the upper bound for the array.
    # Let's set a dynamic upper bound. If we find a solution with scope S, we know S is valid.
    # For the difference array, we can use a size of 200 or 300 to be safe for initial greedy.
    MAX_SCOPE_ESTIMATE = 300
    
    def get_scope(rows):
        return max(max(r) for r in rows)
        
    def validate(rows):
        used = [False] * (MAX_SCOPE_ESTIMATE + 1)
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d > MAX_SCOPE_ESTIMATE:
                        return False # Should not happen if we manage scope
                    if used[d]:
                        return False
                    used[d] = True
        return True

    def greedy_construct(target_scope, used_diffs):
        """
        Constructs one row given current used_diffs and a target_scope.
        Returns None if it fails to construct a valid row within target_scope.
        """
        row = [0]
        # We need to pick k more numbers.
        # To make it harder/easier, we can randomize the order of selection? 
        # No, strictly increasing is required.
        # We pick the next number > prev that doesn't conflict.
        # To allow for backtracking/variability in ILS, we can pick from a set of candidates.
        
        # Simple greedy: pick smallest valid.
        # To add randomness for ILS restarts, we can pick randomly from the first N valid candidates?
        # Or just use random restarts with different seeds.
        
        for pos in range(1, row_len):
            prev_val = row[-1]
            found = False
            # Try values from prev_val + 1 to target_scope
            # Optimization: check from smallest to largest to keep scope tight?
            # Yes, smallest first is generally better for packing.
            for v in range(prev_val + 1, target_scope + 1):
                valid = True
                for j in range(len(row)):
                    d = v - row[j]
                    if d > MAX_SCOPE_ESTIMATE:
                        valid = False
                        break
                    if used_diffs[d]:
                        valid = False
                        break
                if valid:
                    row.append(v)
                    # Mark diffs as used
                    for j in range(len(row) - 1):
                        d = v - row[j]
                        used_diffs[d] = True
                    found = True
                    break
            if not found:
                return None
        return row

    def local_search(rows, used_diffs, max_iterations=5000):
        """
        Attempts to reduce the scope by moving elements to smaller values.
        Returns improved rows and used_diffs, or original if no improvement.
        """
        current_scope = get_scope(rows)
        # We try to reduce the maximum element.
        # Strategy: Pick a random row, pick a random element (not 0), try to move it to a smaller value.
        # If we can move it to a smaller value without conflict, do it.
        
        improved = False
        for _ in range(max_iterations):
            # Pick random row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            
            # Pick random index to try to reduce (1 to k)
            # Bias towards larger indices or larger values?
            # Let's pick a random index i > 0
            i = random.randint(1, row_len - 1)
            
            old_val = row[i]
            
            # Try to find a new_val < old_val that is valid
            # New val must be > row[i-1]
            # And new_val - row[j] must not be in used_diffs (excluding the diffs we are removing)
            
            # Remove old diffs
            old_diffs = []
            for j in range(len(row)):
                if j == i: continue
                d = abs(old_val - row[j])
                old_diffs.append(d)
                used_diffs[d] = False
            
            # Search for new_val
            # Range: row[i-1] + 1 to old_val - 1
            lower = row[i - 1] + 1
            upper = old_val - 1
            
            if lower <= upper:
                # Try values in random order? Or ascending?
                # Ascending might find the tightest packing.
                # But we want to reduce scope, so any valid smaller value helps.
                # Let's try a few random values first, then scan?
                # Scanning is deterministic. Random helps escape local minima in ILS.
                
                # Let's try scanning from lower to upper, but stop at first valid.
                # This tends to pack tight.
                for new_val in range(lower, upper + 1):
                    valid = True
                    new_diffs = []
                    for j in range(len(row)):
                        if j == i: continue
                        d = abs(new_val - row[j])
                        if d > MAX_SCOPE_ESTIMATE:
                            valid = False
                            break
                        if used_diffs[d]:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        # Commit change
                        row[i] = new_val
                        for d in new_diffs:
                            used_diffs[d] = True
                        improved = True
                        break
                # If no valid found, re-add old diffs
                if not improved:
                    for d in old_diffs:
                        used_diffs[d] = True
            else:
                # No room to reduce, re-add old diffs
                for d in old_diffs:
                    used_diffs[d] = True
            
            # If we improved, check time
            if time.time() > deadline:
                break
        
        return rows, used_diffs, improved

    def solve():
        best_rows = None
        best_scope = float('inf')
        
        # We will iterate with increasing "pressure" or just random restarts.
        # To ensure we find a solution, we start with a high scope for greedy, then minimize.
        
        # Initial high scope to guarantee greedy success
        initial_scope = 200 
        
        while time.time() < deadline:
            # Reset used_diffs
            used_diffs = [False] * (MAX_SCOPE_ESTIMATE + 1)
            rows = []
            
            # Construct rows greedily
            success = True
            for _ in range(n_rows):
                # To ensure we don't get stuck, we might need to retry greedy with different random choices?
                # The current greedy is deterministic (smallest valid). 
                # To add randomness, we can shuffle the order of rows? 
                # Or pick randomly from valid candidates in greedy.
                
                # Let's modify greedy to pick randomly from the first 5 valid candidates?
                # For now, let's stick to deterministic greedy but randomize the *order* of rows?
                # No, rows are independent except for used_diffs.
                
                # To get variety, let's add a small random jitter to the greedy choice.
                # Instead of always picking the smallest valid v, pick randomly from valids in [prev+1, prev+10]?
                
                row = None
                # Try to construct with a slightly relaxed constraint or randomization
                # Simple randomization: pick a random valid candidate from the first 20 valid ones.
                
                # Re-implement greedy with randomization
                r = [0]
                for pos in range(1, row_len):
                    prev_val = r[-1]
                    candidates = []
                    for v in range(prev_val + 1, initial_scope + 1):
                        valid = True
                        for j in range(len(r)):
                            d = v - r[j]
                            if d > MAX_SCOPE_ESTIMATE:
                                valid = False
                                break
                            if used_diffs[d]:
                                valid = False
                                break
                        if valid:
                            candidates.append(v)
                            if len(candidates) >= 20: # Limit candidates for speed
                                break
                        if len(candidates) > 0 and v - prev_val > 50: # Stop if gap is large
                            break
                            
                    if not candidates:
                        success = False
                        break
                    
                    # Pick randomly from candidates
                    v = random.choice(candidates)
                    r.append(v)
                    for j in range(len(r) - 1):
                        d = v - r[j]
                        used_diffs[d] = True
                if success:
                    rows.append(r)
                
            if not success:
                # Greedy failed, try again
                continue
            
            current_scope = get_scope(rows)
            
            # Local Search to minimize
            rows, used_diffs, _ = local_search(rows, used_diffs, max_iterations=1000)
            
            current_scope = get_scope(rows)
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
                
                # If we reached the theoretical lower bound, stop
                if best_scope <= total_diffs:
                    break
            
            # Check time
            if time.time() > deadline - 5:
                break

        if best_rows:
            return best_rows
        
        # Fallback: Return a valid but potentially large solution if we found one in greedy but failed LS?
        # Or just construct one with high scope.
        used_diffs = [False] * (MAX_SCOPE_ESTIMATE + 1)
        rows = []
        for _ in range(n_rows):
            r = [0]
            for pos in range(1, row_len):
                prev_val = r[-1]
                while True:
                    v = prev_val + 1
                    valid = True
                    for j in range(len(r)):
                        d = v - r[j]
                        if d > MAX_SCOPE_ESTIMATE:
                            # Expand array?
                            # For fallback, just return empty or raise?
                            # Let's assume 300 is enough.
                            valid = False
                            break
                        if used_diffs[d]:
                            valid = False
                            break
                    if valid:
                        r.append(v)
                        for j in range(len(r) - 1):
                            d = v - r[j]
                            used_diffs[d] = True
                        break
                    prev_val += 1
            rows.append(r)
        return rows

    return solve()

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
