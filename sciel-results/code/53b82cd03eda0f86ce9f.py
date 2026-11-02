import time
import random
import sys

# EVOLVE-BLOCK-START

class DiffTracker:
    __slots__ = ('cap', 'used')

    def __init__(self, cap):
        self.cap = cap
        self.used = bytearray(cap + 1)

    def is_free(self, d):
        return d <= self.cap and self.used[d] == 0

    def check_all(self, diffs):
        u = self.used
        cap = self.cap
        for d in diffs:
            if d > cap or u[d]:
                return False
        return True

    def add(self, d):
        self.used[d] = 1

    def remove(self, d):
        self.used[d] = 0

    def add_many(self, diffs):
        u = self.used
        for d in diffs:
            u[d] = 1

    def remove_many(self, diffs):
        u = self.used
        for d in diffs:
            u[d] = 0


def row_diffs(row):
    out = []
    n = len(row)
    for i in range(n):
        a = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - a)
    return out


def is_valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        for d in row_diffs(row):
            if d in used:
                return False
            used.add(d)
    return True


def max_entry(rows):
    return max(max(r) for r in rows)


def greedy_construct(n_rows, k, cap, rng, strategy='smallest'):
    tracker = DiffTracker(cap)
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        while len(row) < k + 1:
            found = False
            if strategy == 'smallest':
                order = range(last + 1, cap + 1)
            else:
                cands = []
                for v in range(last + 1, cap + 1):
                    diffs = [v - a for a in row]
                    if len(set(diffs)) != len(diffs):
                        continue
                    if not tracker.check_all(diffs):
                        continue
                    cands.append(v)
                if not cands:
                    break
                if strategy == 'spread':
                    v = cands[-1]
                else:
                    v = cands[rng.randrange(len(cands))]
                diffs = [v - a for a in row]
                row.append(v)
                tracker.add_many(diffs)
                last = v
                found = True
                continue

            for v in range(last + 1, cap + 1):
                diffs = [v - a for a in row]
                if len(set(diffs)) != len(diffs):
                    continue
                if not tracker.check_all(diffs):
                    continue
                row.append(v)
                tracker.add_many(diffs)
                last = v
                found = True
                break
            if not found:
                return None, None
        rows.append(row)
    return rows, tracker


def simulated_annealing(rows, n_rows, k, rng, time_limit):
    current_rows = [r[:] for r in rows]
    best_rows = [r[:] for r in rows]
    current_val = max_entry(current_rows)
    best_val = current_val
    
    global_diffs = set()
    for r in current_rows:
        global_diffs.update(row_diffs(r))
        
    T = 50.0
    cooling = 0.9995
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < 0.01:
            break
            
        ri = rng.randrange(n_rows)
        row = current_rows[ri]
        j = rng.randrange(1, k + 1)
        
        old_diffs = row_diffs(row)
        for d in old_diffs:
            global_diffs.remove(d)
            
        low = row[j-1] + 1
        high = row[j+1] - 1 if j < k else current_val + 10
        
        move_made = False
        for _ in range(20):
            if low > high:
                break
            v = rng.randrange(low, high + 1)
            if v == row[j]:
                continue
                
            new_row = row[:]
            new_row[j] = v
            
            if not all(new_row[i] < new_row[i+1] for i in range(k)):
                continue
            
            new_diffs = row_diffs(new_row)
            if len(set(new_diffs)) != len(new_diffs):
                continue
            
            conflict = False
            for d in new_diffs:
                if d in global_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            new_max_row = max(new_row)
            new_global_max = 0
            for idx, r in enumerate(current_rows):
                if idx == ri:
                    new_global_max = max(new_global_max, new_max_row)
                else:
                    new_global_max = max(new_global_max, max(r))
            
            delta = new_global_max - current_val
            if delta < 0:
                accept = True
            else:
                accept = rng.random() < pow(2.71828, -delta / T)
                
            if accept:
                current_rows[ri] = new_row
                current_val = new_global_max
                global_diffs.update(new_diffs)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                move_made = True
                break
        
        if not move_made:
            global_diffs.update(old_diffs)
            
        T *= cooling
        
    return best_rows, best_val


def construct(n_rows, k):
    start = time.time()
    deadline = start + 140.0
    
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs
    
    rng = random.Random(42)
    
    init_cap = lb + 10 * k + 50
    rows, _ = greedy_construct(n_rows, k, init_cap, rng, 'smallest')
    if rows is None:
        rows, _ = greedy_construct(n_rows, k, lb + 50 * k + 200, rng, 'smallest')
        
    best_rows = rows
    best_val = max_entry(best_rows)
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_rows, best_val = simulated_annealing(best_rows, n_rows, k, rng, remaining_time - 5)
        
    if not is_valid(best_rows, n_rows, k):
        wide = best_val + 100
        rows, _ = greedy_construct(n_rows, k, wide, random.Random(7), 'smallest')
        if rows is not None:
            best_rows = rows
            best_val = max_entry(best_rows)
            
    return best_rows

# EVOLVE-BLOCK-END
