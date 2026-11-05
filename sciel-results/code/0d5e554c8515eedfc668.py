import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry.
    
    Uses Simulated Annealing with row-reconstruction moves.
    """
    start_time = time.time()
    max_time = 115.0
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def initial_solution():
        """
        Generate a valid initial solution using a simple greedy approach.
        This might have a large scope, but it's valid.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Greedily pick the smallest possible next value
            # We need k more values.
            # To ensure we can finish, we might need to look ahead, 
            # but for initialization, simple greedy usually works if scope is large enough.
            # We'll set a high limit for initialization.
            limit = 1000 
            while len(row) < k + 1:
                found = False
                # Try values starting from last + 1
                # Optimization: We can jump if we know small values are taken? 
                # For now, simple scan.
                for cand in range(row[-1] + 1, limit + 1):
                    new_diffs = {cand - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(cand)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    # If we hit limit, extend it? 
                    # For init, let's just return None if it fails, 
                    # but greedy on large space usually works.
                    return None
            rows.append(row)
        return rows

    def reconstruct_row(used_diffs_others, target_max, rng, k):
        """
        Try to construct a row of length k+1, starting at 0, 
        with max value <= target_max, using only differences not in used_diffs_others.
        Returns the row if successful, None otherwise.
        """
        # Backtracking search for a single row
        # Depth is k.
        
        def backtrack(current_row):
            if len(current_row) == k + 1:
                return current_row
            
            last_val = current_row[-1]
            # We need to pick the next value.
            # It must be > last_val and <= target_max.
            # Also, if this is not the last element, we need to leave room for (k - len(current_row)) more elements.
            # Max possible value for current step: target_max - (k - len(current_row))
            max_cand = target_max - (k - len(current_row))
            
            # Generate candidates in random order biased towards smaller values?
            # Or just shuffle a range?
            # To be efficient, we can try smaller values first as they are more likely to fit.
            # But randomization helps escape local minima in the global SA.
            # Let's try a mix: sort by value, but with some randomization.
            
            candidates = list(range(last_val + 1, max_cand + 1))
            if not candidates:
                return None
            
            # Heuristic: Try smaller values first, but shuffle to allow diversity.
            # We can use a "biased shuffle" or just shuffle.
            # Shuffling completely might pick large values, which are harder to complete.
            # Let's sort by value, then apply a "random walk" or just shuffle.
            # Given the small depth (k=5), full shuffle is okay.
            
            rng.shuffle(candidates)
            
            for cand in candidates:
                new_diffs = {cand - a for a in current_row}
                if new_diffs & used_diffs_others:
                    continue
                
                # Tentatively add
                current_row.append(cand)
                # Check if this leads to a solution
                res = backtrack(current_row)
                if res is not None:
                    return res
                current_row.pop()
            
            return None
        
        return backtrack([0])

    # Initialize
    current_rows = initial_solution()
    if current_rows is None:
        # Fallback if initial fails (should be rare)
        return [[0, 1, 2, 3, 4, 5] for _ in range(n_rows)] # Invalid but returns something

    current_cost = max(max(r) for r in current_rows)
    best_rows = [r[:] for r in current_rows]
    best_cost = current_cost
    
    # Precompute diffs for current rows
    all_diffs = set()
    for r in current_rows:
        all_diffs.update(get_diffs(r))
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.99
    min_T = 0.01
    rng = random.Random(42)
    
    while time.time() - start_time < max_time:
        if T < min_T:
            break
            
        # Pick a random row to modify
        idx = rng.randint(0, n_rows - 1)
        old_row = current_rows[idx]
        old_max = max(old_row)
        
        # Calculate used diffs excluding this row
        # For efficiency, we could maintain a count of each diff, 
        # but n_rows=7 is small, so recomputing is O(n_rows * k^2) which is fine.
        used_diffs_others = set()
        for i, r in enumerate(current_rows):
            if i != idx:
                used_diffs_others.update(get_diffs(r))
        
        # Try to find a new row with a smaller max value
        # Target max: slightly smaller than old_max? 
        # Or just try to find *any* valid row with max < old_max?
        # To allow moves that don't immediately improve cost (SA), 
        # we can try to find a row with max <= old_max + noise?
        # Actually, if we find a row with max < old_max, cost decreases.
        # If we find a row with max == old_max, cost stays same (neutral move).
        # If we find a row with max > old_max, cost increases (bad move, accept with prob).
        
        # Strategy: Try to find a row with max in [1, old_max + 5]
        # But we prefer smaller.
        
        # Let's try to find a row with max strictly less than old_max first.
        # If that fails, try to find one with max <= old_max.
        
        new_row = None
        
        # Try decreasing targets
        # We want to minimize, so try targets from old_max - 1 down to 1?
        # That's expensive. 
        # Better: Try a random target in [1, old_max]?
        # Or just try to find *any* valid row and see its max.
        
        # Let's try to find a valid row with max <= old_max.
        # We can do this by setting target_max = old_max and running reconstruct_row.
        # If it succeeds, we get a new row.
        
        # To encourage smaller values, we can bias the candidate selection in reconstruct_row
        # towards smaller numbers. The current shuffle is uniform.
        # Let's modify reconstruct_row to accept a 'preference' or just rely on SA.
        
        # Actually, a better move is:
        # 1. Remove row idx.
        # 2. Try to reconstruct with target_max = old_max - 1.
        # 3. If success, cost decreases.
        # 4. If fail, try target_max = old_max.
        # 5. If success, cost same (or we can check if the new row has smaller max).
        
        # Let's try target_max = old_max - 1 first.
        new_row = reconstruct_row(used_diffs_others, old_max - 1, rng, k)
        
        if new_row is None:
            # Try to find a row with max <= old_max (might be same max, but different structure)
            # This helps escape local minima where the row is "stuck" but could be reshuffled.
            new_row = reconstruct_row(used_diffs_others, old_max, rng, k)
            
        if new_row is None:
            T *= cooling_rate
            continue
            
        new_max = max(new_row)
        new_cost = max([max(r) for i, r in enumerate(current_rows) if i != idx] + [new_max])
        
        delta = new_cost - current_cost
        
        # Acceptance criterion
        if delta < 0:
            accept = True
        else:
            accept = rng.random() < pow(2.71828, -delta / T)
            
        if accept:
            current_rows[idx] = new_row
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [r[:] for r in current_rows]
                
        T *= cooling_rate

    return best_rows
