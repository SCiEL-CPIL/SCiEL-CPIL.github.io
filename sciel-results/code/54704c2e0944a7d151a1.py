import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140

    def elapsed():
        return time.time() - start_time

    def time_up():
        return elapsed() > deadline

    def get_max(rows):
        return max(max(row) for row in rows)

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows

    rows = greedy_construct()
    best_rows = rows[:]
    best_max = get_max(rows)

    def check_row_valid(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return None
                diffs.add(d)
        return diffs

    def get_other_diffs(rows, exclude_idx):
        diffs = set()
        for i in range(n_rows):
            if i == exclude_idx:
                continue
            row = rows[i]
            n = len(row)
            for a in range(n):
                for b in range(a + 1, n):
                    diffs.add(row[b] - row[a])
        return diffs

    def hill_climb(rows):
        nonlocal best_max, best_rows
        rows = [r[:] for r in rows]
        no_improve = 0
        max_no_improve = 100000

        while not time_up() and no_improve < max_no_improve:
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)

            old_val = rows[r_idx][pos]

            if random.random() < 0.85:
                delta = random.randint(1, max(1, old_val // 5))
                new_val = old_val - delta
            else:
                delta = random.randint(1, 5)
                new_val = old_val + delta

            if new_val <= rows[r_idx][pos - 1]:
                no_improve += 1
                continue

            test_row = rows[r_idx][:]
            test_row[pos] = new_val
            test_diffs = check_row_valid(test_row)
            if test_diffs is None:
                no_improve += 1
                continue

            other_diffs = get_other_diffs(rows, r_idx)
            if test_diffs & other_diffs:
                no_improve += 1
                continue

            rows[r_idx][pos] = new_val
            new_max = get_max(rows)
            no_improve = 0

            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]

        return rows

    hill_climb(rows)

    return best_rows
