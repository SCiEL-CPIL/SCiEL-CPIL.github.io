import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 145
    
    MAX_DIFF = 150
    diff_bit = [1 << i for i in range(MAX_DIFF + 1)]
    
    def get_mask(diffs):
        mask = 0
        for d in diffs:
            if d <= MAX_DIFF:
                mask |= diff_bit[d]
        return mask

    def _valid(rows):
        used = set()
        for row in rows:
            if not row or row[0] != 0: return False
            for i in range(len(row)-1):
                if row[i+1] <= row[i]: return False
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used: return False
                    used.add(d)
        return True

    def greedy_construct(seed):
        rng = random.Random(seed)
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last = row[-1]
                found = False
                for x in range(last + 1, 151):
                    diffs = [x - a for a in row]
                    mask = get_mask(diffs)
                    if used_mask & mask:
                        continue
                    row.append(x)
                    used_mask |= mask
                    found = True
                    break
                if not found:
                    return None
            rows.append(row)
        return rows

    # Phase 1: Greedy
    best_rows = None
    best_scope = float('inf')
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        rows = greedy_construct(seed)
        if rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows

    if best_rows is None:
        return [[0, 1, 2, 3, 4, 5]] # Fallback, likely invalid but prevents crash

    # Phase 2: Bitset Backtracking to improve scope
    # Target: 112
    # Try M from 112 to best_scope - 1
    
    def backtrack_search(M, deterministic=True, seed=0):
        rng = random.Random(seed)
        start_t = time.time()
        
        def solve(rows, used_mask, row_idx):
            if time.time() > deadline - 2:
                return None
            if row_idx == n_rows:
                return rows
            row = [0]
            return build_row(rows, used_mask, row_idx, row, 1)

        def build_row(rows, used_mask, row_idx, row, depth):
            if time.time() > deadline - 2:
                return None
            if depth == k + 1:
                new_rows = rows + [row]
                return solve(new_rows, used_mask, row_idx + 1)
            
            last = row[-1]
            candidates = list(range(last + 1, M + 1))
            
            if deterministic:
                # Smallest first
                pass
            else:
                rng.shuffle(candidates)
            
            for x in candidates:
                diffs = [x - a for a in row]
                mask = get_mask(diffs)
                if used_mask & mask:
                    continue
                
                result = build_row(rows, used_mask, row_idx, row + [x], depth + 1)
                if result is not None:
                    return result
            return None

        return solve([], 0, 0)

    # Try to find solution for M=112
    for M in range(112, best_scope):
        if time.time() > deadline - 5:
            break
        
        # Try deterministic first
        result = backtrack_search(M, deterministic=True)
        if result and _valid(result):
            new_scope = max(max(r) for r in result)
            if new_scope < best_scope:
                best_rows = result
                best_scope = new_scope
            if M == 112:
                break # Found optimal known
        
        # If not found, try randomized
        if time.time() < deadline - 10:
            for seed in range(5):
                if time.time() > deadline - 5:
                    break
                result = backtrack_search(M, deterministic=False, seed=seed)
                if result and _valid(result):
                    new_scope = max(max(r) for r in result)
                    if new_scope < best_scope:
                        best_rows = result
                        best_scope = new_scope
                    if M == 112:
                        break
            if M == 112:
                break

    return best_rows
