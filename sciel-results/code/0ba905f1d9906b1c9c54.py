import random
import time
import sys

def construct(n_rows, k):
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        break
                    x += 1
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_init()
    current_cost = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 110
    T = 50.0
    cooling = 0.99995
    MAX_VAL = 200

    while time.time() - start_time < timeout:
        # Save state
        old_rows = [r[:] for r in rows]
        old_used = set(used_diffs)
        old_cost = current_cost

        move_type = random.random()
        valid_move = False
        new_cost = old_cost

        if move_type < 0.4:
            # Shift
            row_idx = random.randint(0, n_rows - 1)
            shift = random.choice([-1, 1])
            new_row = rows[row_idx][:]
            ok = True
            if shift == 1:
                if new_row[-1] + 1 > MAX_VAL: ok = False
            else:
                if new_row[1] - 1 <= 0: ok = False
            
            if ok:
                for i in range(1, len(new_row)):
                    new_row[i] += shift
                # Check increase
                for i in range(1, len(new_row)):
                    if new_row[i] <= new_row[i-1]:
                        ok = False
                        break
                if ok:
                    # Check diffs
                    new_diffs = get_diffs(new_row)
                    # Check against used_diffs minus current row's diffs
                    old_diffs = get_diffs(rows[row_idx])
                    if new_diffs & (used_diffs - old_diffs):
                        ok = False
                    else:
                        valid_move = True
                        # Apply
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        rows[row_idx] = new_row
                        new_cost = max(max(r) for r in rows)

        elif move_type < 0.7:
            # Swap
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) > 2:
                i, j = random.sample(range(1, len(row)), 2)
                if row[i] > row[j]: i, j = j, i
                new_row = row[:]
                new_row[i], new_row[j] = new_row[j], new_row[i]
                ok = True
                for m in range(1, len(new_row)):
                    if new_row[m] <= new_row[m-1]:
                        ok = False
                        break
                if ok:
                    new_diffs = get_diffs(new_row)
                    old_diffs = get_diffs(row)
                    if new_diffs & (used_diffs - old_diffs):
                        ok = False
                    else:
                        valid_move = True
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        rows[row_idx] = new_row
                        new_cost = max(max(r) for r in rows)

        else:
            # Rebuild
            row_idx = random.randint(0, n_rows - 1)
            old_row = rows[row_idx]
            old_diffs = get_diffs(old_row)
            
            # Remove old diffs
            used_diffs.difference_update(old_diffs)
            
            new_row = [0]
            ok = True
            while len(new_row) < k + 1:
                x = new_row[-1] + 1
                found = False
                while x <= MAX_VAL:
                    conflict = False
                    for a in new_row:
                        if (x - a) in used_diffs:
                            conflict = True
                            break
                    if not conflict:
                        for a in new_row:
                            used_diffs.add(x - a)
                        new_row.append(x)
                        found = True
                        break
                    x += 1
                if not found:
                    ok = False
                    break
            
            if ok:
                valid_move = True
                rows[row_idx] = new_row
                new_cost = max(max(r) for r in rows)
            else:
                # Rollback
                used_diffs.update(old_diffs)
                rows[row_idx] = old_row
                valid_move = False

        if valid_move:
            # Accept/Reject
            delta = new_cost - old_cost
            if delta <= 0 or random.random() < pow(2.71828, -delta / T):
                current_cost = new_cost
                # State is already updated
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [r[:] for r in rows]
            else:
                # Rollback
                rows = old_rows
                used_diffs = old_used
                current_cost = old_cost
        
        T *= cooling

    return best_rows
