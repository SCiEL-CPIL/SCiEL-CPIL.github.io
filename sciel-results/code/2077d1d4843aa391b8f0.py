import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Problem parameters
    # Each row has k+1 elements: a_0=0, a_1, ..., a_k
    # Total differences per row: D = k*(k+1)//2
    # Total differences in system: T = n_rows * D
    D = k * (k + 1) // 2
    T = n_rows * D
    
    # Lower bound for M (largest entry)
    # We need T distinct differences in [1, M]. So M >= T.
    # Also, the largest entry is at least k (since it's increasing by at least 1).
    # But T is usually much larger.
    lower_bound = T

    def validate(rows):
        """Check if the set of rows is valid and return the max entry."""
        used_diffs = set()
        max_val = 0
        for row in rows:
            # Check increasing
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    return -1
            max_val = max(max_val, row[-1])
            # Check differences
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return -1
                    used_diffs.add(d)
        return max_val

    def greedy_construct():
        """
        Construct a valid solution using a greedy approach.
        This provides an initial upper bound.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                val = row[-1] + 1
                while True:
                    conflict = False
                    for i in range(len(row)):
                        d = val - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                    if not conflict:
                        break
                    val += 1
                row.append(val)
                for i in range(len(row) - 1):
                    used_diffs.add(val - row[i])
            rows.append(row)
        return rows

    # Initial solution
    best_rows = greedy_construct()
    best_score = validate(best_rows)
    
    if best_score == lower_bound:
        return best_rows

    # Backtracking Search with Forward Checking
    # State: 
    #   rows: list of completed rows
    #   used_diffs: set of all differences used so far
    #   current_row: list of elements in the current row being constructed
    #   pos: index in current_row (next element to place)
    
    # We will try to construct rows one by one.
    # To minimize the largest entry, we should try to keep the current row small.
    # However, we also need to leave enough "room" for future rows.
    # A simple heuristic is to try values starting from the smallest possible.
    
    # To escape local minima and find better global solutions, we can randomize the choice of values.
    # Or we can use a "random restart" strategy where we change the order of trying values.
    
    # Let's implement a DFS with a time limit.
    # We'll also add some randomness to the choice of the next value.
    
    def backtrack_search(used_diffs, rows, current_row, pos, limit):
        """
        Recursively construct the current row and subsequent rows.
        Returns (success, rows, used_diffs) or (False, None, None)
        """
        if time.time() > limit:
            return False, None, None
            
        if pos == k + 1:
            # Current row is complete
            new_rows = rows + [current_row]
            if len(new_rows) == n_rows:
                return True, new_rows, used_diffs
            else:
                # Start next row
                return backtrack_search(used_diffs, new_rows, [0], 1, limit)
        
        # Determine the range of possible values for current_row[pos]
        min_val = current_row[-1] + 1
        # We don't have an easy upper bound for a single element without looking ahead,
        # but we can cap it at the current best_score or some heuristic.
        # Let's cap it at best_score to ensure we don't make things worse immediately.
        max_val = best_score
        
        # Generate candidate values
        # To add randomness, we can shuffle the candidates or pick a random start point.
        # Let's try a random start point within a small window, then expand.
        # Or just shuffle the range? Shuffling a large range is expensive.
        # Let's try a random offset.
        
        candidates = []
        # Try values in a random order?
        # For efficiency, let's try a few random values first, then scan.
        
        # Strategy: Try a few random values in [min_val, max_val], then scan from min_val.
        # This helps escape local minima.
        
        # But for now, let's stick to scanning from min_val to max_val, but with a twist:
        # We can randomize the "step" or the "start" slightly.
        
        # Actually, a better randomization is to try a random value first.
        if random.random() < 0.5:
            # Try a random value
            rand_val = random.randint(min_val, max_val)
            # Check if rand_val is valid
            valid = True
            for i in range(len(current_row)):
                d = rand_val - current_row[i]
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                new_current_row = current_row + [rand_val]
                new_used_diffs = set(used_diffs)
                for i in range(len(new_current_row) - 1):
                    new_used_diffs.add(rand_val - new_current_row[i])
                success, res_rows, res_diffs = backtrack_search(new_used_diffs, rows, new_current_row, pos + 1, limit)
                if success:
                    return True, res_rows, res_diffs
        
        # Scan from min_val to max_val
        for val in range(min_val, max_val + 1):
            if time.time() > limit:
                return False, None, None
                
            # Check if val is valid
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in used_diffs:
                    valid = False
                    break
            
            if valid:
                new_current_row = current_row + [val]
                new_used_diffs = set(used_diffs)
                for i in range(len(new_current_row) - 1):
                    new_used_diffs.add(val - new_current_row[i])
                
                success, res_rows, res_diffs = backtrack_search(new_used_diffs, rows, new_current_row, pos + 1, limit)
                if success:
                    return True, res_rows, res_diffs
                    
        return False, None, None

    # We will run the backtracking search with a time limit for each restart.
    # We will try to improve upon the best_score found by the greedy method.
    
    # Since the search space is huge, we need to be smart about the time limit.
    # Let's allocate a portion of the time to each restart.
    
    num_restarts = 20
    time_per_restart = (deadline - time.time() - 5.0) / num_restarts if num_restarts > 0 else 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # For each restart, we can try to find a solution with a max entry <= best_score - 1.
        # If we find one, we update best_score and best_rows.
        
        # To make the search more effective, we can randomize the initial state or the order of values.
        # Let's just run the backtracking search with a time limit.
        
        # We'll use a separate used_diffs set for each restart to avoid interference.
        # But we start from scratch.
        
        limit = time.time() + time_per_restart
        
        success, res_rows, res_diffs = backtrack_search(set(), [], [0], 1, limit)
        
        if success:
            score = validate(res_rows)
            if score < best_score:
                best_score = score
                best_rows = res_rows
                if best_score == lower_bound:
                    break
        else:
            # If we didn't find a solution, we can still try to salvage a partial solution?
            # No, the interface requires a complete valid set.
            # So we stick with the best_rows we have.
            pass

    return best_rows
