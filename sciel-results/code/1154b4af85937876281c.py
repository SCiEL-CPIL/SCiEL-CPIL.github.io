import time
import random
import heapq
import bisect
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 145  # 5s margin from 150s limit
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * n_diffs_per_row
    
    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _all_diffs(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used
    
    def _valid(rows):
        all_d = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            rd = _row_diffs(row)
            if rd & all_d:
                return False
            all_d |= rd
        return True
    
    def _scope(rows):
        return max(max(r) for r in rows)
    
    def _check_realizable(diff_set, max_val):
        """Check if a set of n_diffs_per_row differences can be embedded into
        k+1 strictly increasing integers starting at 0 within [0, max_val].
        
        Uses bipartite matching: assign each difference to a pair (i,j) with i<j,
        such that the resulting positions are consistent (a_j - a_i = d).
        
        We use a simpler check: try to reconstruct the row from the differences.
        A row of k+1 elements has exactly k*(k+1)/2 differences.
        Given the differences, we can try to reconstruct by noting that:
        - The smallest difference must be a consecutive pair
        - We build the row incrementally
        
        Actually, let's use a direct reconstruction approach:
        For a valid difference set D of size k*(k+1)/2, there must exist
        positions p_0=0 < p_1 < ... < p_k such that {p_j - p_i : i<j} = D.
        
        We can check this by trying all possible assignments of which differences
        correspond to which pairs. But that's expensive.
        
        Alternative: use the fact that if we know the row, the differences are determined.
        So we check: does there exist a row with these exact differences?
        
        For small k (k=5), we can try a backtracking reconstruction.
        """
        if len(diff_set) != n_diffs_per_row:
            return None
        
        # Try to reconstruct the row from the difference set
        # The row must start at 0, so all differences are positive
        # p_0 = 0
        # p_1 = min(D) (the smallest difference must be p_1 - p_0)
        # Actually not necessarily - the smallest difference could be p_j - p_i for any i<j
        
        # Better approach: enumerate possible rows
        # p_0 = 0
        # p_1 > 0, p_2 > p_1, ..., p_k > p_{k-1}
        # All pairwise differences must be in D and cover D exactly
        
        # For k=5, we have 6 positions. Let's use backtracking.
        diffs_sorted = sorted(diff_set)
        
        # p_0 = 0
        # p_1 must be in D (since p_1 - p_0 = p_1)
        # p_2 - p_0 and p_2 - p_1 must be in D
        # etc.
        
        def _reconstruct_row(diffs_remaining, current_row, max_val, depth):
            if depth == k + 1:
                # Check that all diffs are used
                if len(diffs_remaining) == 0:
                    return current_row[:]
                return None
            
            last = current_row[-1]
            # Try each remaining diff as the next step
            # The next position p_depth = last + d where d is in diffs_remaining
            # But we also need p_depth - p_i for all i < depth to be in diffs_remaining
            
            # Sort candidates by value for efficiency
            candidates = []
            for d in diffs_remaining:
                new_pos = last + d
                if new_pos > max_val:
                    continue
                # Check that new_pos - p_i is in diffs_remaining for all i < depth
                valid = True
                for i in range(depth):
                    diff_to_prev = new_pos - current_row[i]
                    if diff_to_prev == d and i == depth - 1:
                        continue  # This is the step itself
                    if diff_to_prev not in diffs_remaining:
                        valid = False
                        break
                if valid:
                    candidates.append(d)
            
            for d in candidates:
                new_pos = last + d
                # Remove the diffs that would be used
                used_diffs = set()
                for i in range(depth):
                    used_diffs.add(new_pos - current_row[i])
                
                # Check all used_diffs are in diffs_remaining
                if used_diffs - diffs_remaining:
                    continue
                
                new_remaining = diffs_remaining - used_diffs
                result = _reconstruct_row(new_remaining, current_row + [new_pos], max_val, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        result = _reconstruct_row(set(diff_set), [0], max_val, 1)
        return result
    
    def _diff_set_to_row_fallback(diff_set, max_val):
        """Fallback: try to find a row with these differences using element-level search."""
        # This is a slower but more robust approach
        used = set()
        row = [0]
        for _ in range(k):
            found = False
            for x in range(row[-1] + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if new_diffs <= diff_set:
                    # Check if adding these diffs keeps us on track
                    # We need exactly diff_set at the end
                    row.append(x)
                    found = True
                    break
            if not found:
                return None
        # Verify
        final_diffs = _row_diffs(row)
        if final_diffs == diff_set:
            return row
        return None
    
    def _generate_diff_sets_for_scope(max_scope, n_rows, rng, deadline):
        """Generate n_rows disjoint difference sets, each of size n_diffs_per_row,
        all within [1, max_scope], such that each set is realizable.
        
        Uses backtracking over the difference space."""
        all_diffs = set(range(1, max_scope + 1))
        used_diffs = set()
        row_diff_sets = []
        rows = []
        
        def _assign_row_diffs(remaining_diffs, row_idx, deadline):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Choose a subset of remaining_diffs of size n_diffs_per_row
            # that is realizable
            # Heuristic: prefer smaller differences (compactness)
            # and differences that cluster together (easier to realize)
            
            # Get available diffs sorted
            avail = sorted(remaining_diffs)
            
            # Try different subsets - use a scoring heuristic
            # Score a candidate set by:
            # 1. Sum of elements (prefer smaller)
            # 2. Whether it's realizable
            
            # Generate candidate subsets
            # For efficiency, try greedy first, then random
            attempts = 0
            max_attempts = 500 if row_idx < 2 else 200
            
            # Greedy: pick smallest diffs that form a realizable set
            # A realizable set must contain a chain: d1, d1+d2, d1+d2+d3, ...
            # where the partial sums give the row positions
            
            # Try multiple strategies
            strategies = ['greedy_small', 'greedy_cluster', 'random']
            rng.shuffle(strategies)
            
            for strategy in strategies:
                if attempts >= max_attempts:
                    break
                
                if strategy == 'greedy_small':
                    # Try to build a row by picking small differences
                    # Start with the smallest available diff
                    if not avail:
                        continue
                    # Try different starting points
                    for start_idx in range(min(10, len(avail))):
                        if attempts >= max_attempts:
                            break
                        attempts += 1
                        # Try to build a row starting with avail[start_idx]
                        candidate_set = set()
                        row_positions = [0]
                        last = 0
                        success = True
                        for step in range(k):
                            # Find a diff d such that last + d - p_i is in remaining for all i
                            found_step = False
                            for d in avail:
                                if d in candidate_set:
                                    continue
                                new_pos = last + d
                                if new_pos > max_scope:
                                    continue
                                # Check all new differences
                                valid = True
                                for i in range(len(row_positions)):
                                    diff = new_pos - row_positions[i]
                                    if diff not in remaining_diffs and diff != d:
                                        # diff must be in remaining or already in candidate_set
                                        if diff not in candidate_set:
                                            valid = False
                                            break
                                if valid:
                                    # Add this step
                                    for i in range(len(row_positions)):
                                        diff = new_pos - row_positions[i]
                                        if diff not in candidate_set:
                                            candidate_set.add(diff)
                                    row_positions.append(new_pos)
                                    last = new_pos
                                    found_step = True
                                    break
                            if not found_step:
                                success = False
                                break
                        
                        if success and len(candidate_set) == n_diffs_per_row:
                            # Check realizability
                            row = _check_realizable(candidate_set, max_scope)
                            if row is not None:
                                row_diff_sets.append(candidate_set)
                                rows.append(row)
                                used_diffs |= candidate_set
                                if _assign_row_diffs(remaining_diffs - candidate_set, row_idx + 1, deadline):
                                    return True
                                row_diff_sets.pop()
                                rows.pop()
                                used_diffs -= candidate_set
                                return False  # Backtrack
                elif strategy == 'random':
                    # Pick random subsets
                    for _ in range(max_attempts - attempts):
                        if attempts >= max_attempts:
                            break
                        attempts += 1
                        if len(avail) < n_diffs_per_row:
                            break
                        # Random sample
                        subset = set(rng.sample(avail, n_diffs_per_row))
                        row = _check_realizable(subset, max_scope)
                        if row is not None:
                            row_diff_sets.append(subset)
                            rows.append(row)
                            used_diffs |= subset
                            if _assign_row_diffs(remaining_diffs - subset, row_idx + 1, deadline):
                                return True
                            row_diff_sets.pop()
                            rows.pop()
                            used_diffs -= subset
        
            return False
        
        if _assign_row_diffs(all_diffs, 0, deadline):
            return rows
        return None
    
    def _greedy_diff_space(seed_offset=0):
        """Construct rows by greedily assigning difference sets in the difference space."""
        rng = random.Random(42 + seed_offset)
        # Estimate a good scope
        # Lower bound: n_rows * n_diffs_per_row differences needed, all distinct, in [1, scope]
        # So scope >= n_rows * n_diffs_per_row
        min_scope = n_rows * n_diffs_per_row
        # Add some slack
        scope = min_scope + int(min_scope * 0.3) + 20
        
        used_diffs = set()
        rows = []
        
        for row_idx in range(n_rows):
            # Try to find a realizable diff set from remaining diffs
            row = None
            best_diff_set = None
            best_row_val = scope + 1
            
            # Try multiple random attempts
            for attempt in range(200):
                if time.time() > deadline - 10:
                    break
                # Sample a candidate diff set
                # Prefer smaller diffs for compactness
                # Weighted sampling: smaller diffs more likely
                candidate_set = set()
                # Start with a small diff
                available = [d for d in range(1, scope + 1) if d not in used_diffs]
                if len(available) < n_diffs_per_row:
                    break
                
                # Try to build incrementally
                row_positions = [0]
                last = 0
                success = True
                for step in range(k):
                    found = False
                    # Try candidates in order of preference
                    # Prefer diffs that keep the row compact
                    candidates = []
                    for d in available:
                        if d in candidate_set:
                            continue
                        new_pos = last + d
                        if new_pos > scope:
                            continue
                        # Check feasibility
                        valid = True
                        for i in range(len(row_positions)):
                            diff = new_pos - row_positions[i]
                            if diff == d and i == len(row_positions) - 1:
                                continue
                            if diff not in candidate_set and diff not in available:
                                valid = False
                                break
                        if valid:
                            candidates.append((new_pos, d))
                    
                    if not candidates:
                        success = False
                        break
                    
                    # Sort by new_pos (prefer compact)
                    candidates.sort()
                    # Pick from top few with some randomness
                    top_n = min(5, len(candidates))
                    idx = rng.randint(0, top_n - 1) if rng.random() < 0.3 else 0
                    new_pos, d = candidates[idx]
                    
                    for i in range(len(row_positions)):
                        diff = new_pos - row_positions[i]
                        candidate_set.add(diff)
                    
                    row_positions.append(new_pos)
                    last = new_pos
                
                if success and len(candidate_set) == n_diffs_per_row:
                    # Verify it's valid (no conflicts with used)
                    if candidate_set & used_diffs:
                        continue
                    # Verify the row is correct
                    row = _check_realizable(candidate_set, scope)
                    if row is not None:
                        row_val = max(row)
                        if row_val < best_row_val:
                            best_row_val = row_val
                            best_diff_set = candidate_set
                            row = row
                            if row_val <= scope * 0.9:  # Good enough
                                break
            
            if row is None:
                # Fallback: element-level greedy
                row = [0]
                for _ in range(k):
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            break
                        x += 1
                    row.append(x)
                    used_diffs |= new_diffs
                # Recompute scope
                scope = max(scope, max(row))
            else:
                used_diffs |= best_diff_set
            
            rows.append(row)
        
        return rows
    
    def local_search_shrink(rows, deadline, rng):
        """Local search: try to reduce the maximum entry by modifying individual elements."""
        rows = [list(r) for r in rows]
        best_scope = _scope(rows)
        used = _all_diffs(rows)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_scope = _scope(rows)
            
            for ri in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    if old_val < current_scope:
                        continue
                    
                    low = row[pos - 1] + 1
                    high = old_val - 1
                    if pos + 1 < len(row):
                        high = min(high, row[pos + 1] - 1)
                    
                    if low > high:
                        continue
                    
                    # Try values from high to low (prefer smaller)
                    for new_val in range(high, low - 1, -1):
                        if time.time() > deadline:
                            break
                        
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for i in range(pos + 1, len(row)):
                            old_diffs.add(row[i] - old_val)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        
                        other_diffs = used - old_diffs
                        if new_diffs & other_diffs:
                            continue
                        
                        row[pos] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        break
        
        return rows
    
    def diff_space_local_search(rows, deadline, rng, iterations=5000):
        """Local search in difference space: try to swap differences between rows
        to reduce the overall scope."""
        rows = [list(r) for r in rows]
        best_scope = _scope(rows)
        used = _all_diffs(rows)
        row_diffs = [_row_diffs(r) for r in rows]
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            # Pick two random rows
            r1, r2 = rng.sample(range(n_rows), 2)
            row1, row2 = rows[r1], rows[r2]
            
            # Try to swap a difference from row1 with one from row2
            d1_list = list(row_diffs[r1])
            d2_list = list(row_diffs[r2])
            
            d1 = rng.choice(d1_list)
            d2 = rng.choice(d2_list)
            
            if d1 == d2:
                continue
            
            # Try removing d1 from row1 and adding d2
            # This requires reconstructing row1 with d2 instead of d1
            # and row2 with d1 instead of d2
            
            # For now, just try element-level moves
            # Pick a position in row1 to modify
            pi = rng.randint(1, k)
            old_val = row1[pi]
            lo = row1[pi - 1] + 1
            hi = min(best_scope, row1[pi + 1] - 1) if pi < k else best_scope
            
            if lo >= hi:
                continue
            
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
            old_diffs = set()
            for i in range(pi):
                old_diffs.add(old_val - row1[i])
            for i in range(pi + 1, len(row1)):
                old_diffs.add(row1[i] - old_val)
            
            new_diffs = set()
            for i in range(pi):
                new_diffs.add(new_val - row1[i])
            for i in range(pi + 1, len(row1)):
                new_diffs.add(row1[i] - new_val)
            
            other_diffs = used - old_diffs
            if new_diffs & other_diffs:
                continue
            
            if new_val <= row1[pi - 1] or (pi < k and new_val >= row1[pi + 1]):
                continue
            
            row1[pi] = new_val
            used -= old_diffs
            used |= new_diffs
            row_diffs[r1] = new_diffs | (row_diffs[r1] - old_diffs)
            
            new_scope = max(max(r) for r in rows)
            if new_scope < best_scope:
                best_scope = new_scope
        
        return rows, best_scope
    
    # Main search: use difference-space based construction
    best_rows = None
    best_scope = float('inf')
    
    # Try difference-space greedy with multiple seeds
    for seed in range(15):
        if time.time() > deadline - 40:
            break
        rows = _greedy_diff_space(seed_offset=seed)
        if rows is not None and len(rows) == n_rows:
            scope = _scope(rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope
    
    # Fallback: simple element-level greedy if diff-space didn't work
    if best_rows is None:
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
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)
    
    # Apply local search
    rng = random.Random(123)
    if time.time() < deadline - 15:
        polished = local_search_shrink(best_rows, deadline - 10, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Difference-space local search
    if time.time() < deadline - 15:
        polished, polished_scope = diff_space_local_search(best_rows, deadline - 10, rng, iterations=10000)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Final polish
    if time.time() < deadline - 3:
        rng = random.Random(999)
        polished = local_search_shrink(best_rows, deadline - 3, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    return best_rows
