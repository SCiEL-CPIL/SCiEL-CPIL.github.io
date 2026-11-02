"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import heapq
from collections import defaultdict
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        arr = np.array(row)
        n = len(arr)
        if n < 2:
            return set()
        # Compute all pairwise differences using broadcasting
        diff_matrix = arr[np.newaxis, :] - arr[:, np.newaxis]
        # Take upper triangle (i < j means row[i] < row[j], so diff > 0)
        iu = np.triu_indices(n, k=1)
        diffs = diff_matrix[iu]
        return set(diffs.tolist())

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_set(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Best-First Search over difference slots ----------
    # Key insight: treat the problem as 1D spatial packing where each row
    # occupies a set of "difference shells". Build the most rigid rows first
    # (those with largest spans) to minimize backtracking.

    def diff_count(row):
        """Number of positive differences in a row."""
        n = len(row)
        return n * (n - 1) // 2

    def rigidity_score(row, used_other):
        """Score how 'rigid' a row is: higher means it constrains the solution more.
        Rigid rows should be placed first to minimize backtracking.
        """
        if not row or len(row) < 2:
            return 0
        max_entry = max(row)
        n_diffs = diff_count(row)
        # Rigidity increases with span and number of differences
        # Rows with large spans use many difference slots, constraining others
        span = max_entry
        # Penalty for using small differences (more contested)
        small_diffs = sum(1 for d in row_diffs_set(row) if d <= 15)
        # Combine: prefer rows that use many slots but avoid small contested diffs
        return span * 1.0 + n_diffs * 0.5 - small_diffs * 2.0

    def best_first_build_row(used_other, max_span, beam_width=10, max_depth=None):
        """Build a row using best-first search over partial rows.
        State: (row, current_diffs, last_entry)
        Priority: fewer used differences (more flexibility), then smaller max entry
        """
        if max_depth is None:
            max_depth = max_span + k * 5  # generous bound

        # Priority queue: (score, row, current_diffs, last_entry)
        # Lower score = better (fewer differences used, smaller span)
        initial_state = ([0], set(), 0)
        pq = []
        heapq.heappush(pq, (0.0, initial_state))
        
        visited = set()  # Track (tuple(row), frozenset(current_diffs)) to avoid revisits
        best_row = None
        best_score = float('inf')
        
        nodes_expanded = 0
        max_nodes = 5000  # Limit search space
        
        while pq and nodes_expanded < max_nodes and time_left() > 3:
            score, state = heapq.heappop(pq)
            row, current_diffs, last_entry = state
            
            # Check if we've reached a complete row
            if len(row) == k + 1:
                if score < best_score:
                    best_score = score
                    best_row = list(row)
                continue
            
            # Create visited key
            row_key = tuple(row)
            diffs_key = frozenset(current_diffs)
            state_key = (row_key, diffs_key)
            
            if state_key in visited:
                continue
            visited.add(state_key)
            nodes_expanded += 1
            
            # Generate next states
            for trial_x in range(last_entry + 1, min(last_entry + 50, max_span + 1)):
                # Quick check: can we add trial_x?
                new_diffs = set()
                valid = True
                for a in row:
                    d = trial_x - a
                    if d <= 0:
                        valid = False
                        break
                    if d in used_other or d in current_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check if this would exceed max_span
                if trial_x > max_span:
                    continue
                
                new_row = row + [trial_x]
                new_current_diffs = current_diffs | new_diffs
                
                # Score: prefer fewer total differences used (more flexibility for other rows)
                # and smaller max entry
                n_diffs_so_far = len(new_current_diffs)
                expected_total_diffs = diff_count(new_row)
                # Penalty for using too many small diffs
                small_penalty = sum(1 for d in new_current_diffs if d <= 10) * 0.3
                # Prefer smaller spans
                span_penalty = trial_x * 0.1
                # Prefer rows that are "balanced" in their difference usage
                if n_diffs_so_far > 0:
                    avg_d = sum(new_current_diffs) / n_diffs_so_far
                    balance_penalty = abs(avg_d - trial_x / 2) * 0.05
                else:
                    balance_penalty = 0
                
                new_score = n_diffs_so_far * 0.5 + small_penalty + span_penalty + balance_penalty
                
                # Only expand if this state is promising
                if new_score < best_score * 1.5:  # Prune clearly worse states
                    new_state = (new_row, new_current_diffs, trial_x)
                    heapq.heappush(pq, (new_score, new_state))
        
        return best_row

    def try_reduce_row_bfs(rows, row_idx, used_other):
        """Try to find a row at row_idx with smaller max entry using best-first search."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        
        # Try progressively smaller max spans
        for target_max in range(current_max - 1, max(k, current_max - 20), -1):
            if time_left() < 5:
                break
            
            row = best_first_build_row(used_other, target_max, beam_width=10)
            if row is not None and max(row) <= target_max:
                if max(row) < best_max:
                    best_row = list(row)
                    best_max = max(row)
        
        return best_row, best_max

    def local_polish_bfs(rows, max_passes=10, time_threshold=8):
        """Reduce scope by locally re-optimizing each row using best-first search.
        Order rows by rigidity: most rigid first.
        """
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time_left() > time_threshold and pass_count < max_passes:
            improved = False
            pass_count += 1
            
            # Order by rigidity: most rigid rows first
            def get_rigidity(i):
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                return rigidity_score(rows[i], used_other)
            
            order = sorted(range(n_rows), key=get_rigidity, reverse=True)
            
            for i in order:
                if time_left() < time_threshold:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row_bfs(rows, i, used_other)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    def final_high_precision_polish_bfs(rows):
        """Final polish pass with best-first search and tighter constraints."""
        rows = [list(r) for r in rows]
        
        # Try to reduce each row by 1 more entry with aggressive search
        for _ in range(5):
            if time_left() < 5:
                break
            improved = False
            for i in range(n_rows):
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                
                # Try to reduce by exactly 1
                target = max(rows[i]) - 1
                if target >= k:
                    row = best_first_build_row(used_other, target, beam_width=15)
                    if row is not None and max(row) <= target:
                        rows[i] = row
                        improved = True
            
            if not improved:
                break
        
        return rows

    def kick_bfs(rows):
        """Perturb the solution using best-first search with larger search space."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Try best-first search with larger max_span for exploration
            success = False
            for attempt in range(10):
                max_span = max(rows[idx]) + 20 + attempt * 15
                row = best_first_build_row(used_other, max_span, beam_width=15)
                if row is not None:
                    rows[idx] = row
                    success = True
                    break
            
            if not success:
                # Fallback to greedy
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row
        
        return rows

    # Initial polish with best-first search
    if time_left() > 20:
        polished = local_polish_bfs(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search with best-first search
    while time_left() > 30:
        # 1. Local Search
        polished = local_polish_bfs(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        # 2. Kick
        if time_left() < 25:
            break

        kicked_rows = kick_bfs(best_rows)

        # 3. Local Search on kicked rows
        polished_kicked = local_polish_bfs(kicked_rows)

        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final high-precision polish
    if time_left() > 12:
        polished = final_high_precision_polish_bfs(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
