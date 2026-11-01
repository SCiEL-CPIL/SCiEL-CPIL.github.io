def construct(n_rows, k):
    import sys
    import time
    from itertools import combinations

    sys.setrecursionlimit(100000)
    start_time = time.time()
    time_limit = 115.0  # Leave some buffer before 120s timeout

    # Each row has k+1 entries: (0, a1, a2, ..., ak) with 0 < a1 < ... < ak
    # All positive differences within a row must be globally unique across all rows.
    # Minimize the largest entry.

    # Step 1: Greedy construction for initial solution
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]

    # Step 2: Try to improve via branch-and-bound with bitsets
    # For a given max_val M, try to find n_rows non-conflicting rows with all entries <= M.

    def solve_with_max(M, n_rows, k, time_limit):
        """Try to find n_rows mutually non-conflicting rows with all entries <= M."""
        if M < k:
            return None

        # Generate all candidate rows: (0, a1, ..., ak) with 0 < a1 < ... < ak <= M
        # Compute difference bitmask for each
        # Differences range from 1 to M, so we need M bits

        candidates = []
        # For efficiency, generate combinations and compute masks
        # C(M, k) can be large; for M=112, k=5: ~60M which is too many
        # We need smarter generation or pruning

        # Alternative: generate rows recursively with pruning
        # Build row element by element, tracking used differences

        found = [False]
        selected = []
        node_count = [0]
        node_limit = 2000000  # Limit search nodes

        def build_row(start, current_row, current_mask, rows_selected):
            """Recursively build one row, then proceed to next row."""
            if found[0] or node_count[0] > node_limit:
                return
            node_count[0] += 1

            if len(current_row) == k + 1:
                # Row is complete, add to selected and move to next row
                if rows_selected + 1 == n_rows:
                    found[0] = True
                    selected.extend(current_row)
                    return
                # Start next row
                build_row(1, [0], current_mask, rows_selected + 1)
                return

            # Try extending current_row with next element x > current_row[-1]
            last = current_row[-1]
            for x in range(last + 1, M + 1):
                # Check if x - a for all a in current_row creates new differences
                # that don't conflict with current_mask
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if current_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                if not valid:
                    continue
                build_row(x + 1, current_row + [x], current_mask | new_mask, rows_selected)
                if found[0]:
                    return

        # Start with first row
        build_row(1, [0], 0, 0)

        if not found[0]:
            return None

        # Reconstruct rows from selected (which contains all elements flattened)
        # Actually, selected only contains the last complete row's elements... 
        # We need to track all rows properly. Let me redo this.

        # Let me use a different approach: track all selected rows
        found2 = [False]
        selected_rows = []
        node_count2 = [0]

        def build_all_rows(row_idx, current_row, current_mask, all_used_mask):
            """Build row_idx (0-indexed) row, then proceed to next."""
            if found2[0] or node_count2[0] > node_limit:
                return
            node_count2[0] += 1

            if len(current_row) == k + 1:
                # Row is complete
                selected_rows.append(current_row[:])
                if row_idx + 1 == n_rows:
                    found2[0] = True
                    return
                # Start next row
                build_all_rows(row_idx + 1, [0], all_used_mask | current_mask, all_used_mask | current_mask)
                if not found2[0]:
                    selected_rows.pop()
                return

            last = current_row[-1]
            for x in range(last + 1, M + 1):
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if (all_used_mask | current_mask) & bit:
                        valid = False
                        break
                    new_mask |= bit
                if not valid:
                    continue
                build_all_rows(row_idx, current_row + [x], current_mask | new_mask, all_used_mask)
                if found2[0]:
                    return

        build_all_rows(0, [0], 0, 0)

        if not found2[0]:
            return None

        return [list(r) for r in selected_rows]

    # Try to find better solutions by decreasing max_val
    # But first, let's try a smarter approach: iterative deepening with time limits

    # Try values below the greedy solution
    for target_max in range(best_max - 1, max(best_max - 15, k), -1):
        if time.time() - start_time > time_limit:
            break
        result = solve_with_max(target_max, n_rows, k, time_limit)
        if result is not None:
            best_rows = result
            best_max = target_max
            break

    return best_rows
