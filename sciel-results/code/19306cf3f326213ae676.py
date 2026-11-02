import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    
    Approach:
    1. Generate a pool of candidate rows (Sidon sets) with small max values.
    2. Use Greedy Set Packing to select n_rows disjoint rows from the pool.
    3. Use Local Search to improve the selection (swap rows to reduce max).
    """
    start_time = time.time()
    time_limit = 110
    
    best_rows = None
    best_max_val = float('inf')
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def generate_candidate():
        """
        Generates a random valid row (Sidon set) of size k+1 starting at 0.
        Uses a greedy randomized approach to keep values small.
        """
        row = [0]
        used_diffs = set()
        for _ in range(k):
            # Try to find the next element x > row[-1]
            # We bias towards smaller x to keep max low
            x = row[-1] + 1
            # Randomized window to escape local minima in generation
            # but we want small values, so we scan up
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    # Valid
                    row.append(x)
                    used_diffs.update(new_diffs)
                    break
                x += 1
                # Safety break if x gets too large (shouldn't happen for small k)
                if x > 5000:
                    return None
        return row

    def greedy_set_packing(candidates, n_needed):
        """
        Greedily selects n_needed disjoint rows from candidates.
        Candidates are assumed to be sorted by max value.
        Returns list of selected rows or None if not enough.
        """
        selected = []
        used_diffs = set()
        
        for cand in candidates:
            if len(selected) == n_needed:
                break
            cand_diffs = get_diffs(cand)
            if not (cand_diffs & used_diffs):
                selected.append(cand)
                used_diffs.update(cand_diffs)
                
        if len(selected) == n_needed:
            return selected
        return None

    def local_search(rows, candidate_pool, max_iters=500):
        """
        Attempts to improve the set of rows by swapping one row with another
        from the pool that fits the remaining difference slots and has a smaller max.
        """
        rows = [r[:] for r in rows]
        # Precompute diffs for current rows
        row_diffs = [get_diffs(r) for r in rows]
        used_diffs = set()
        for d in row_diffs:
            used_diffs.update(d)
            
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Find the row with the largest max value
            max_idx = 0
            max_val = max(r[-1] for r in rows)
            for i, r in enumerate(rows):
                if r[-1] == max_val:
                    max_idx = i
                    break
            
            # Try to replace rows[max_idx] with a candidate from the pool
            # that has a smaller max value and fits into the remaining slots
            current_row = rows[max_idx]
            current_diffs = row_diffs[max_idx]
            
            # Diffs available for the new row: All diffs NOT used by OTHER rows
            # Let other_diffs = used_diffs - current_diffs
            other_diffs = used_diffs - current_diffs
            
            # We want a candidate c such that:
            # 1. c[-1] < current_row[-1] (improvement)
            # 2. get_diffs(c) is disjoint from other_diffs
            
            # Sort candidates by max value to find the best improvement quickly
            # We can filter the pool once
            best_candidate = None
            best_candidate_diffs = None
            
            # To speed this up, we might want to pre-filter candidates
            # But the pool size is manageable (e.g., 1000-5000)
            for c in candidate_pool:
                if c[-1] >= current_row[-1]:
                    continue
                c_diffs = get_diffs(c)
                if not (c_diffs & other_diffs):
                    # This candidate fits and is smaller
                    if best_candidate is None or c[-1] < best_candidate[-1]:
                        best_candidate = c
                        best_candidate_diffs = c_diffs
                        # Since we iterate pool (not sorted), we can't break immediately
                        # unless we sort the pool first.
                        # Let's assume pool is sorted by max value.
                        # If pool is sorted, the first valid one found is the best.
                        break
            
            if best_candidate is not None:
                # Perform swap
                # Update used_diffs
                used_diffs -= current_diffs
                used_diffs |= best_candidate_diffs
                
                # Update rows and row_diffs
                rows[max_idx] = best_candidate
                row_diffs[max_idx] = best_candidate_diffs
                
                improved = True
                # Continue searching for further improvements
            else:
                # Try other rows with max value?
                # Or break to avoid getting stuck?
                # For now, break if the largest row can't be improved.
                # A more robust search would try all rows.
                break
                
        return rows

    # Main Loop
    # 1. Generate a large pool of candidates
    # We need to ensure we have enough candidates to form a valid set.
    # The number of candidates needed depends on n_rows and k.
    # For n=7, k=5, we need 7 disjoint sets.
    # Generating ~2000-5000 candidates should be sufficient.
    
    pool_size = 2000
    candidates = []
    
    # Generate initial pool
    while len(candidates) < pool_size and time.time() - start_time < time_limit / 2:
        cand = generate_candidate()
        if cand is not None:
            candidates.append(cand)
            
    # Sort candidates by max value (last element)
    candidates.sort(key=lambda x: x[-1])
    
    # 2. Try to find a solution
    while time.time() - start_time < time_limit:
        # Greedy Set Packing
        rows = greedy_set_packing(candidates, n_rows)
        
        if rows is not None:
            current_max = max(r[-1] for r in rows)
            if current_max < best_max_val:
                best_max_val = current_max
                best_rows = [r[:] for r in rows]
                
                # If we found a very good solution, we might want to stop
                # But let's try to improve it with local search
                rows = local_search(rows, candidates, max_iters=100)
                current_max = max(r[-1] for r in rows)
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_rows = [r[:] for r in rows]
        
        # If we didn't find a solution, or we want to try again with a slightly different pool
        # We can add more candidates
        if time.time() - start_time < time_limit - 5:
            # Generate more candidates
            for _ in range(100):
                cand = generate_candidate()
                if cand is not None:
                    candidates.append(cand)
            # Re-sort
            candidates.sort(key=lambda x: x[-1])
            
        # If we have a valid solution, we can stop after some time
        # to avoid wasting time if we are stuck at a local optimum
        # But the local search above handles some of that.
        
        # Check if we hit the theoretical lower bound?
        # Counting bound: n * k(k+1)/2 <= M(M+1)/2 ?
        # Actually, the differences are 1..M.
        # Total diffs needed = n * k(k+1)/2.
        # So M(M+1)/2 >= n * k(k+1)/2
        # M(M+1) >= n * k(k+1)
        # For n=7, k=5: 7 * 30 = 210.
        # M(M+1) >= 210 -> 14*15=210. So M >= 14?
        # Wait, the counting bound in the prompt is 105.
        # Why 105?
        # "simple counting bound is 105".
        # Maybe the bound is on the largest *difference*?
        # If the largest difference is D, then D >= total_diffs / n ? No.
        # The differences are distinct over all rows.
        # Total distinct differences = n * C(k+1, 2).
        # These differences must be in {1, ..., M}.
        # So M >= n * C(k+1, 2).
        # For n=7, k=5: 7 * 15 = 105.
        # So M >= 105.
        
        if best_max_val <= 105:
            break

    return best_rows
