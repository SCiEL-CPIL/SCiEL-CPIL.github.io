import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    def get_max_scope(rows):
        return max(max(r) for r in rows)

    def get_involved_diffs(row, pos, val):
        diffs = set()
        for i in range(pos):
            diffs.add(val - row[i])
        for j in range(pos + 1, len(row)):
            diffs.add(row[j] - val)
        return diffs

    # Interleaved greedy: grow all rows in parallel to balance row growth
    def generate_greedy_interleaved():
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        # Grow each row to k+1 elements, interleaving
        # We use a round-robin approach, but if a row is significantly behind, we might prioritize it?
        # For simplicity, standard round-robin is good for balance.
        while any(len(r) < k + 1 for r in rows):
            for r_idx in range(n_rows):
                row = rows[r_idx]
                if len(row) >= k + 1:
                    continue
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
        return rows, used_diffs

    # Standard greedy (row-by-row)
    def generate_greedy_sequential():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, used_diffs

    # Try both and pick the better initial solution
    rows_a, used_a = generate_greedy_interleaved()
    scope_a = get_max_scope(rows_a)
    rows_b, used_b = generate_greedy_sequential()
    scope_b = get_max_scope(rows_b)

    if scope_a <= scope_b:
        rows = [r[:] for r in rows_a]
        used_diffs = set(used_a)
        current_scope = scope_a
    else:
        rows = [r[:] for r in rows_b]
        used_diffs = set(used_b)
        current_scope = scope_b

    best_rows = [r[:] for r in rows]
    best_scope = current_scope

    # Phase 1: Aggressive Hill Climbing with Shuffling
    # Focus on reducing the scope by moving elements down.
    improved = True
    hc_iter = 0
    max_hc_iters = 50
    while improved and hc_iter < max_hc_iters and time.time() < deadline - 15:
        improved = False
        hc_iter += 1
        row_order = list(range(n_rows))
        if hc_iter > 1:
            random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 15:
                break
            row = rows[r_idx]
            for pos in range(1, k + 1):
                old_val = row[pos]
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < k else old_val

                if low >= old_val:
                    continue

                # Try to reduce value
                for new_val in range(low, old_val):
                    old_involved = get_involved_diffs(row, pos, old_val)
                    used_diffs.difference_update(old_involved)

                    new_involved = get_involved_diffs(row, pos, new_val)

                    if len(new_involved) != len(old_involved):
                        used_diffs |= old_involved
                        continue

                    if new_involved & used_diffs:
                        used_diffs |= old_involved
                        continue

                    # Valid
                    used_diffs |= new_involved
                    row[pos] = new_val

                    new_scope = get_max_scope(rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        improved = True
                        break
                    else:
                        used_diffs.difference_update(new_involved)
                        used_diffs |= old_involved
                        row[pos] = old_val

                if improved:
                    break
            if improved:
                break

    # Phase 2: Simulated Annealing with Row Rebuilds
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    def rebuild_row(r_idx):
        """
        Attempts to reconstruct row r_idx greedily to minimize its max element,
        given the current used_diffs from other rows.
        Returns (new_row, new_scope, success_flag).
        """
        # Temporarily remove this row's diffs from used_diffs
        row = rows[r_idx]
        old_involved = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                old_involved.add(row[j] - row[i])
        
        used_diffs.difference_update(old_involved)
        
        # Greedily construct new row
        new_row = [0]
        # We want to minimize the last element. 
        # Greedy strategy: pick the smallest valid x for each position.
        for pos in range(1, k + 1):
            x = new_row[-1] + 1
            found = False
            # Limit search space to avoid infinite loops if impossible (though it should be possible)
            # We know a solution exists (the old row), so we can search up to old_row[pos] or a bit more
            # But to minimize, we start from low.
            # To be safe and fast, we search up to the old value + some buffer, or just until found.
            # Since we removed old diffs, the old row is a valid candidate.
            max_search = row[-1] + 20 
            while x <= max_search:
                new_diffs = {x - a for a in new_row}
                if not new_diffs & used_diffs:
                    new_row.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
                x += 1
            
            if not found:
                # Restore old state
                used_diffs.difference_update(set(d for d in used_diffs if d in old_involved)) # This is tricky, better to just revert logic
                # Actually, if not found, we should revert the partial changes.
                # Let's just return failure and revert everything.
                # We added some diffs to used_diffs for new_row. We need to remove them.
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                         used_diffs.discard(new_row[j] - new_row[i])
                # Re-add old diffs
                used_diffs |= old_involved
                return None, current_scope, False

        new_scope = max(max(r) for r in rows) # Note: rows[r_idx] is still old, so this is wrong.
        # We need to calculate scope with new_row
        temp_scope = max([max(r) for r in rows if r != row] + [max(new_row)])
        
        # If we accept, we update rows[r_idx] and used_diffs is already updated.
        # If we reject, we must revert used_diffs.
        return new_row, temp_scope, True

    while time.time() < deadline - 5:
        # Decide move type
        if random.random() < 0.2:
            # Row Rebuild Move
            r_idx = random.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_scope = current_scope
            
            # Perform rebuild
            new_row, new_scope_val, success = rebuild_row(r_idx)
            
            if success:
                # Check if we should accept
                # If new_scope_val < current_scope, accept
                # Else, accept with prob exp(-delta/T)
                if new_scope_val < current_scope:
                    rows[r_idx] = new_row
                    current_scope = new_scope_val
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    delta = new_scope_val - current_scope
                    if random.random() < math.exp(-delta / T):
                        rows[r_idx] = new_row
                        current_scope = new_scope_val
                    else:
                        # Revert: restore old row and old diffs
                        # used_diffs currently has new_row's diffs.
                        # We need to remove new_row's diffs and add old_row's diffs.
                        new_involved = set()
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                new_involved.add(new_row[j] - new_row[i])
                        used_diffs.difference_update(new_involved)
                        
                        old_involved = set()
                        for i in range(len(old_row)):
                            for j in range(i + 1, len(old_row)):
                                old_involved.add(old_row[j] - old_row[i])
                        used_diffs |= old_involved
            # else: rebuild failed, state is already reverted in rebuild_row
            
        else:
            # Standard Local Move
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, k)
            old_val = row[pos]

            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < k else old_val + 30

            # Bias: if this is the last element (the max), try to reduce it
            if pos == k:
                if low < old_val:
                    new_val = random.randint(low, old_val - 1)
                else:
                    continue
            else:
                # Random neighborhood - wider for exploration
                search_low = max(low, old_val - 15)
                search_high = min(high, old_val + 15)
                if search_low > search_high:
                    continue
                new_val = random.randint(search_low, search_high)

            if new_val == old_val:
                continue

            old_involved = get_involved_diffs(row, pos, old_val)
            used_diffs.difference_update(old_involved)

            new_involved = get_involved_diffs(row, pos, new_val)

            if len(new_involved) != len(old_involved):
                used_diffs |= old_involved
                continue

            if new_involved & used_diffs:
                used_diffs |= old_involved
                continue

            # Valid move
            used_diffs |= new_involved
            row[pos] = new_val

            new_scope = get_max_scope(rows)

            if new_scope < current_scope:
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                delta = new_scope - current_scope
                if random.random() < math.exp(-delta / T):
                    current_scope = new_scope
                else:
                    used_diffs.difference_update(new_involved)
                    used_diffs |= old_involved
                    row[pos] = old_val

        T *= cooling
        if T < min_T:
            # Restart with new temperature + quick hill climb
            T = 5.0
            improved_hc = True
            while improved_hc and time.time() < deadline - 5:
                improved_hc = False
                for r_idx2 in range(n_rows):
                    row2 = rows[r_idx2]
                    for pos2 in range(1, k + 1):
                        old_val2 = row2[pos2]
                        low2 = row2[pos2 - 1] + 1
                        high2 = row2[pos2 + 1] - 1 if pos2 < k else old_val2
                        if low2 >= old_val2:
                            continue
                        for new_val2 in range(low2, old_val2):
                            old_inv2 = get_involved_diffs(row2, pos2, old_val2)
                            used_diffs.difference_update(old_inv2)
                            new_inv2 = get_involved_diffs(row2, pos2, new_val2)
                            if len(new_inv2) != len(old_inv2):
                                used_diffs |= old_inv2
                                continue
                            if new_inv2 & used_diffs:
                                used_diffs |= old_inv2
                                continue
                            used_diffs |= new_inv2
                            row2[pos2] = new_val2
                            new_scope2 = get_max_scope(rows)
                            if new_scope2 < current_scope:
                                current_scope = new_scope2
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best_rows = [r[:] for r in rows]
                                improved_hc = True
                                break
                            else:
                                used_diffs.difference_update(new_inv2)
                                used_diffs |= old_inv2
                                row2[pos2] = old_val2
                        if improved_hc:
                            break
                    if improved_hc:
                        break

    # Phase 3: Final polish - one more pass of hill climbing on best_rows
    # Reconstruct used_diffs from best_rows
    final_used = set()
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                final_used.add(r[j] - r[i])

    # Try to improve best_rows with a quick local search
    for r_idx in range(n_rows):
        row = best_rows[r_idx]
        for pos in range(1, k + 1):
            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < k else old_val
            if low >= old_val:
                continue
            for new_val in range(low, old_val):
                old_involved = get_involved_diffs(row, pos, old_val)
                final_used.difference_update(old_involved)
                new_involved = get_involved_diffs(row, pos, new_val)
                if len(new_involved) != len(old_involved):
                    final_used |= old_involved
                    continue
                if new_involved & final_used:
                    final_used |= old_involved
                    continue
                final_used |= new_involved
                row[pos] = new_val
                new_scope = get_max_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    break
                else:
                    final_used.difference_update(new_involved)
                    final_used |= old_involved
                    row[pos] = old_val

    return best_rows
