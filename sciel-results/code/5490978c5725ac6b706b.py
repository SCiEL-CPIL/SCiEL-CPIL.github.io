import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    def validate_rows(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def solve_for_scope(M, seed=0):
        rng = random.Random(seed)
        rows = [[0] for _ in range(n_rows)]
        used_mask = 0
        
        sys.setrecursionlimit(10000)
        
        def backtrack(row_idx, elem_idx, used_mask, rows):
            if time.time() > deadline - 5:
                return False
            
            if elem_idx == k + 1:
                if row_idx == n_rows - 1:
                    return True
                return backtrack(row_idx + 1, 1, used_mask, rows)
            
            prev_val = rows[row_idx][elem_idx - 1]
            
            candidates = []
            for x in range(prev_val + 1, M + 1):
                valid = True
                new_diffs = []
                for i in range(elem_idx):
                    d = x - rows[row_idx][i]
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return False
            
            rng.shuffle(candidates)
            
            for x, new_diffs in candidates:
                new_mask = used_mask
                for d in new_diffs:
                    new_mask |= (1 << d)
                
                rows[row_idx][elem_idx] = x
                
                if backtrack(row_idx, elem_idx + 1, new_mask, rows):
                    return True
            
            return False
        
        if backtrack(0, 1, 0, rows):
            return rows
        else:
            return None

    best_rows = None
    current_M = lb
    
    attempts_per_M = 20
    
    while current_M < lb + 150:
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(attempts_per_M):
            if time.time() > deadline - 5:
                break
            rows = solve_for_scope(current_M, seed=current_M * 100 + attempt)
            if rows is not None:
                if validate_rows(rows):
                    best_rows = rows
                    found = True
                    break
        
        if found:
            break
        
        current_M += 1
    
    if best_rows is None:
        return None

    def local_improve(rows, seed=42):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            
            improved = False
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            
            row.pop(elem_idx)
            
            used = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        used.add(r[j] - r[i])
            
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1
            
            for new_val in range(max_val, min_val - 1, -1):
                valid = True
                for a in row[:elem_idx]:
                    if (new_val - a) in used:
                        valid = False
                        break
                if not valid:
                    continue
                for a in row[elem_idx:]:
                    if (a - new_val) in used:
                        valid = False
                        break
                if valid:
                    row.insert(elem_idx, new_val)
                    improved = True
                    break
            
            if not improved:
                row.insert(elem_idx, old_val)
                
            new_scope = max(max(r) for r in rows)
            if new_scope < scope:
                scope = new_scope
                
        return rows, scope

    improved_rows, imp_scope = local_improve(best_rows, seed=12345)
    if imp_scope < max(max(r) for r in best_rows):
        best_rows = improved_rows

    return best_rows

if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
