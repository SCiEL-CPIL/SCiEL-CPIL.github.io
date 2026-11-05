"""Difference triangle set (7,5) solver using backtracking with row-level search."""
import sys
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    """
    
    def generate_rows(used_diffs, max_span):
        """
        Generator that yields valid rows of length k+1 with max element <= max_span
        that do not conflict with used_diffs.
        """
        row = [0]
        
        def backtrack(row, local_used):
            if len(row) == k + 1:
                yield row[:]
                return
            
            last = row[-1]
            remaining = k + 1 - len(row)
            if last + remaining > max_span:
                return
            
            max_next = max_span - (remaining - 1)
            
            for x in range(last + 1, max_next + 1):
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_used:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                row.append(x)
                yield from backtrack(row, local_used | new_diffs)
                row.pop()
        
        yield from backtrack(row, set())

    def find_solution(span_limit):
        """
        Backtrack over rows to find a valid set of n_rows rows with max element <= span_limit.
        """
        used_diffs = set()
        current_rows = []
        
        def backtrack_rows(index):
            if index == n_rows:
                return True
            
            # Iterate through candidates for the next row
            for cand in generate_rows(used_diffs, span_limit):
                cand_diffs = set()
                for i in range(len(cand)):
                    for j in range(i + 1, len(cand)):
                        cand_diffs.add(cand[j] - cand[i])
                
                # The generator already ensures no conflict with used_diffs
                # But let's double check just in case (performance vs safety)
                # Actually, the generator checks `d in used_diffs` so it's safe.
                
                current_rows.append(cand)
                used_diffs.update(cand_diffs)
                
                if backtrack_rows(index + 1):
                    return True
                
                used_diffs.difference_update(cand_diffs)
                current_rows.pop()
            
            return False
        
        if backtrack_rows(0):
            return current_rows
        return None

    # Determine search range
    if n_rows == 7 and k == 5:
        # Known best is 112, bound 105
        # Try 105 to 112
        for span in range(105, 113):
            rows = find_solution(span)
            if rows is not None:
                return rows
        # Fallback
        for span in range(113, 150):
            rows = find_solution(span)
            if rows is not None:
                return rows
    else:
        # Generic
        # Lower bound: k
        # Upper bound: k * n_rows + k
        lb = k
        ub = k * n_rows + k + 10
        for span in range(lb, ub + 1):
            rows = find_solution(span)
            if rows is not None:
                return rows
        
        # Fallback greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # If all fails
    return []
