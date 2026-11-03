import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 118.0
    
    # Fallback Greedy
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    start_M = total_diffs
    
    def generate_sidon_rows(M):
        rows = []
        
        def backtrack(idx, current_row, used_diffs_mask):
            if idx == row_len:
                rows.append((tuple(current_row), used_diffs_mask))
                return
            
            last = current_row[-1]
            min_next = last + 1
            # We need to place (row_len - idx) elements.
            # The current one is at idx.
            # We need row_len - idx - 1 more elements after this one.
            # So x <= M - (row_len - idx - 1)
            max_next = M - (row_len - idx - 1)
            
            if min_next > max_next:
                return
            
            for x in range(min_next, max_next + 1):
                new_diffs = [x - a for a in current_row]
                new_mask = used_diffs_mask
                valid = True
                for d in new_diffs:
                    bit = 1 << d
                    if new_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if valid:
                    current_row.append(x)
                    backtrack(idx + 1, current_row, new_mask)
                    current_row.pop()
        
        backtrack(0, [0], 0)
        return rows

    def find_packing(candidates, M, target_count):
        best_solution = None
        start_time = time.time()
        
        # Universe mask for differences 1..M
        # Bit i represents difference i
        universe_mask = 0
        for i in range(1, M + 1):
            universe_mask |= (1 << i)
            
        def dfs(start_idx, selected, used_mask, remaining_slots):
            nonlocal best_solution
            if time.time() - start_time > 10.0:
                return False
            
            if remaining_slots == 0:
                best_solution = selected[:]
                return True
            
            # Pruning: Check available bits
            available_mask = universe_mask ^ used_mask
            if available_mask.bit_count() < (k * (k + 1) // 2) * remaining_slots:
                return False
            
            for i in range(start_idx, len(candidates)):
                row, mask = candidates[i]
                if used_mask & mask:
                    continue
                
                selected.append(row)
                if dfs(i + 1, selected, used_mask | mask, remaining_slots - 1):
                    return True
                selected.pop()
                
            return False
        
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        if dfs(0, [], 0, target_count):
            return best_solution
        return None

    for M in range(start_M, start_M + 20):
        if time.time() > deadline - 5.0:
            break
            
        candidates = generate_sidon_rows(M)
        if not candidates:
            continue
            
        solution = find_packing(candidates, M, n_rows)
        if solution:
            rows = [list(r) for r in solution]
            return rows

    return best_rows
