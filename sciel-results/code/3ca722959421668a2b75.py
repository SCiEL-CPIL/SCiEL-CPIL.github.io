import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Problem parameters
    # Each row has k+1 elements: a_0=0, a_1, ..., a_k
    # Total differences per row: D = k*(k+1)//2
    D = k * (k + 1) // 2
    T = n_rows * D
    
    # Lower bound for M (largest entry)
    lower_bound = T
    
    def get_diffs(row):
        """Return set of differences for a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_row(max_val, used_diffs=None, randomize=False):
        """
        Generate a single valid row with max entry <= max_val.
        If used_diffs is provided, the row must not conflict with it.
        Returns row or None.
        """
        if used_diffs is None:
            used_diffs = set()
        
        row = [0]
        for _ in range(k):
            found = False
            # Determine search range
            # We want to keep values small, so we search upwards from last+1
            # but we can also try to "fill in" gaps if we allow non-greedy? 
            # For now, standard greedy: pick smallest valid next value.
            # To add diversity, we can skip some values randomly.
            
            start_val = row[-1] + 1
            # Upper bound for this step to keep total max_val in check
            # Rough heuristic: remaining steps need at least (k - len(row)) * 1
            # So val <= max_val - (k - len(row))
            end_val = max_val - (k - len(row))
            
            if start_val > end_val:
                return None

            # Randomization strategy: 
            # With prob 0.2, skip the first few candidates to allow different structures
            skip_count = 0
            if randomize and random.random() < 0.3:
                skip_count = random.randint(1, 3)
            
            for v in range(start_val + skip_count, end_val + 1):
                # Check conflicts with current row
                conflict_row = False
                for i in range(len(row)):
                    d = v - row[i]
                    if d in used_diffs or d in get_diffs(row): # d in get_diffs(row) is redundant if we update used_diffs, but we are checking against current row's internal diffs too? 
                        # Actually, we just need to check against 'used_diffs' which should contain all diffs of previous rows AND current row so far.
                        # Let's maintain a local set for current row diffs + global used_diffs
                        pass
                
                # Efficient check:
                ok = True
                for i in range(len(row)):
                    d = v - row[i]
                    if d in used_diffs:
                        ok = False
                        break
                if not ok:
                    continue
                
                # If ok, this is a candidate. 
                # If randomize, we might accept it or keep looking? 
                # Standard greedy takes the first one. 
                # To get diversity, we can take a random valid one from a small window.
                # For simplicity in generation, let's stick to first valid to ensure speed, 
                # but vary the 'skip_count' or start point.
                
                row.append(v)
                for i in range(len(row) - 1):
                    used_diffs.add(v - row[i])
                found = True
                break
            
            if not found:
                return None
        return row

    def generate_candidate_pool(target_max, count):
        """Generate a pool of candidate rows with max entry <= target_max."""
        candidates = []
        # We need to ensure we get enough valid candidates.
        # Greedy is fast.
        for _ in range(count):
            # Try to generate a row
            # We can vary the 'max_val' constraint slightly to get different shapes
            # Or just use target_max
            row = greedy_row(target_max, None, randomize=True)
            if row:
                # Check if it's a duplicate
                # Simple check: tuple(row)
                if row not in candidates:
                    candidates.append(row)
        return candidates

    # Strategy:
    # 1. Estimate a good upper bound M.
    # 2. Generate a pool of candidate rows with max entry <= M.
    # 3. Build a conflict graph.
    # 4. Use a greedy/set-packing algorithm to select n_rows non-conflicting rows.
    # 5. If we can't select n_rows, increase M and repeat.
    # 6. Once a valid set is found, try to improve it by swapping rows for "better" (smaller max) rows.

    # Initial M estimate
    # Lower bound is T. Best known is 134 for (4,7) where T=4*28=112.
    # So M is likely T + some overhead.
    # Let's start with M = T + 20 and increase if needed.
    
    current_M = lower_bound + 10
    best_solution = None
    best_max_val = float('inf')
    
    while time.time() < deadline - 10.0:
        # Generate candidates
        # How many? Maybe 100-500 depending on time.
        num_candidates = 200
        candidates = generate_candidate_pool(current_M, num_candidates)
        
        if len(candidates) < n_rows:
            current_M += 5
            continue
            
        # Precompute diffs for candidates
        cand_diffs = [get_diffs(row) for row in candidates]
        cand_maxes = [row[-1] for row in candidates]
        
        # Build conflict matrix (or adjacency list)
        # n_candidates is small (200), so O(N^2) is fine.
        # conflict[i][j] = True if cand_diffs[i] intersects cand_diffs[j]
        n_cand = len(candidates)
        conflicts = [[False] * n_cand for _ in range(n_cand)]
        
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                # Check intersection
                # Since D is small (28 for k=7), sets are small
                if cand_diffs[i].intersection(cand_diffs[j]):
                    conflicts[i][j] = True
                    conflicts[j][i] = True
        
        # Now, find a set of n_rows non-conflicting candidates minimizing the max value.
        # This is a weighted independent set problem.
        # Heuristic: Greedy selection.
        # Sort candidates by max value (ascending).
        # Pick the smallest one that doesn't conflict with already picked ones.
        
        indices = list(range(n_cand))
        indices.sort(key=lambda i: cand_maxes[i])
        
        selected = []
        selected_diffs = set()
        
        for i in indices:
            if len(selected) == n_rows:
                break
            # Check if i conflicts with any selected
            conflict_found = False
            for s_idx in selected:
                if conflicts[i][s_idx]:
                    conflict_found = True
                    break
            if not conflict_found:
                selected.append(i)
                selected_diffs.update(cand_diffs[i])
        
        if len(selected) == n_rows:
            # We found a valid set!
            max_val = max(cand_maxes[i] for i in selected)
            if max_val < best_max_val:
                best_max_val = max_val
                best_solution = [candidates[i][:] for i in selected]
                
            # If we found a solution close to lower bound, we might be done
            if max_val == lower_bound:
                return best_solution
                
            # Try to improve by local search on the selected set
            # Swap one selected row with another candidate that has a smaller max value
            # and doesn't conflict with the rest.
            
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Sort selected by max value descending (worst first)
                sel_sorted = sorted(selected, key=lambda i: cand_maxes[i], reverse=True)
                
                for s_idx in sel_sorted:
                    # Try to replace s_idx with a candidate j not in selected
                    # such that cand_maxes[j] < cand_maxes[s_idx]
                    # and j doesn't conflict with (selected - {s_idx})
                    
                    current_conflicts = [i for i in selected if i != s_idx]
                    
                    # Candidates to try: those with smaller max value
                    # We can just iterate through all candidates sorted by max value
                    for j in indices: # indices is sorted by max value asc
                        if cand_maxes[j] >= cand_maxes[s_idx]:
                            break # Since indices is sorted asc, if j is too big, rest are too
                        if j in selected:
                            continue
                        # Check if j conflicts with any in current_conflicts
                        ok = True
                        for c_idx in current_conflicts:
                            if conflicts[j][c_idx]:
                                ok = False
                                break
                        if ok:
                            # Swap
                            selected.remove(s_idx)
                            selected.append(j)
                            best_max_val = max(cand_maxes[i] for i in selected)
                            best_solution = [candidates[i][:] for i in selected]
                            improved = True
                            break # Restart loop with new selection
                    if improved:
                        break
        else:
            # Couldn't find a valid set with current M
            current_M += 5
            
    if best_solution:
        return best_solution
    else:
        # Fallback to a simple valid construction (greedy without optimization)
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
