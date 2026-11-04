"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n_rows, k):
    def gen_rulers(max_mark):
        """Generate all Golomb rulers with k+1 marks, max mark <= max_mark."""
        rulers = []
        marks = [0]
        
        def backtrack(idx):
            if idx == k + 1:
                rulers.append(marks[:])
                return
            
            start = marks[idx-1] + 1 if idx > 0 else 1
            for val in range(start, max_mark + 1):
                valid = True
                for i in range(idx):
                    d = val - marks[i]
                    for j in range(i):
                        if marks[i] - marks[j] == d:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    marks.append(val)
                    backtrack(idx + 1)
                    marks.pop()
        
        backtrack(1)
        return rulers
    
    def ruler_diffs(ruler):
        diffs = set()
        for i in range(len(ruler)):
            for j in range(i+1, len(ruler)):
                diffs.add(ruler[j] - ruler[i])
        return diffs
    
    start_time = time.time()
    time_limit = 100
    
    low = n_rows * (k+1) * k // 2
    high = 150
    
    for m in range(low, high + 1):
        if time.time() - start_time > time_limit:
            break
        
        all_rulers = gen_rulers(m)
        if not all_rulers:
            continue
        
        ruler_info = [(r, ruler_diffs(r)) for r in all_rulers]
        
        used = set()
        assigned = []
        
        def backtrack(row_idx):
            if time.time() - start_time > time_limit:
                return None
            
            if row_idx == n_rows:
                return [r for r, _ in assigned]
            
            for r, diffs in ruler_info:
                if not (diffs & used):
                    used.update(diffs)
                    assigned.append((r, diffs))
                    
                    result = backtrack(row_idx + 1)
                    if result is not None:
                        return result
                    
                    used.difference_update(diffs)
                    assigned.pop()
            
            return None
        
        result = backtrack(0)
        if result is not None:
            return result
    
    # Fallback: greedy
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
