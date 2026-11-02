"""Difference triangle set (4,7): backtracking over difference slots with MRV."""
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses backtracking search over the difference set with dynamic variable ordering (MRV).
    """
    total_diffs = n_rows * k * (k + 1) // 2
    # We need to assign each of the total_diffs difference values to a unique integer >= 1.
    # But the differences are constrained: for each row, the differences must be consistent
    # with a set of k+1 points starting at 0.
    
    # Alternative approach: we build rows one by one, but use backtracking with MRV
    # on which difference to place next.
    
    # Actually, let's think differently. The key insight is that each row is determined
    # by its k+1 elements. The differences from 0 are just the elements themselves (except 0).
    # So each row has k "non-zero" elements, and the differences include all pairwise differences.
    
    # Let's use a different strategy: backtracking search where we assign elements to rows.
    # For each row, we have k+1 positions (including 0). The differences that must be unique
    # are all a_j - a_i for i < j.
    
    # Strategy: We'll build the solution incrementally. At each step, we decide the next
    # element to place. We use MRV: among all "slots" (row, position) that still need to be
    # filled, pick the one with the fewest valid values given the current used differences.
    
    # For a row with existing elements [0, e1, e2, ..., em], the next element x must satisfy:
    # - x > em (strictly increasing)
    # - For each existing element a in the row, x - a is not in used
    # - Also, for any two new elements... but we add one at a time.
    
    # Actually, since we add elements one by one in increasing order within a row, when we add
    # a new element x to a row with existing elements, the new differences are x - a for each
    # a in the row. These must all be unused.
    
    # MRV: For each row that isn't complete, the number of "available" next values depends on
    # how many small integers would conflict. The row that is most constrained (fewest options
    # near its current max) should be filled next.
    
    # Let's implement this:
    
    sys.setrecursionlimit(100000)
    
    used = set()
    rows = [[0] for _ in range(n_rows)]
    # Track the last element in each row
    last = [0] * n_rows
    # Number of elements in each row
    counts = [1] * n_rows
    
    # We want to minimize the maximum element. Let's do a search that tries to keep
    # elements small.
    
    # For MRV, we need to estimate how constrained each row is. A row with more elements
    # already placed is more constrained because it has more differences that could conflict.
    # Also, a row whose next element would need to be large is more "urgent" in some sense.
    
    # Let's define: for each incomplete row, the minimum possible next value is last[row] + 1.
    # The number of conflicts at that value tells us how constrained it is.
    
    def get_mrv_row():
        """Find the incomplete row with the most constraints (fewest valid next values)."""
        best_row = -1
        best_score = float('inf')
        for r in range(n_rows):
            if counts[r] >= k + 1:
                continue
            # Try the smallest possible next value and count how many small values are blocked
            # We'll estimate by checking the first few candidates
            min_val = last[r] + 1
            # Count how many of the next 20 values are blocked
            blocked = 0
            for candidate in range(min_val, min_val + 20):
                # Check if any difference candidate - a for a in rows[r] is in used
                conflict = False
                for a in rows[r]:
                    if candidate - a in used:
                        conflict = True
                        break
                if conflict:
                    blocked += 1
            # More blocked = more constrained
            score = blocked
            if score < best_score:
                best_score = score
                best_row = r
        return best_row
    
    # This approach might be slow. Let's try a simpler but effective heuristic:
    # Always fill the row that has the fewest elements first (least complete),
    # or the row that is most "stuck".
    
    # Actually, let's try a different approach: iterative deepening on the maximum value.
    # For a given max value M, check if a valid configuration exists using backtracking.
    
    # But that might be too slow. Let's try the greedy-with-backtracking approach.
    
    def backtrack(depth):
        if depth == n_rows * k:
            return True
        
        # Find the next row to extend using MRV heuristic
        # Choose the row with the smallest number of available next values
        best_row = -1
        best_options = None
        best_count = float('inf')
        
        for r in range(n_rows):
            if counts[r] >= k + 1:
                continue
            min_val = last[r] + 1
            # Count available values up to some limit
            count = 0
            options = []
            for candidate in range(min_val, min_val + 50):
                conflict = False
                for a in rows[r]:
                    if candidate - a in used:
                        conflict = True
                        break
                if not conflict:
                    count += 1
                    if count <= 20:
                        options.append(candidate)
                    if count >= best_count:
                        break
            if count < best_count:
                best_count = count
                best_row = r
                best_options = options
                if count == 0:
                    return False
        
        if best_row == -1:
            return True  # All rows complete
        
        if best_count == 0:
            return False
        
        # Try each option for best_row
        for candidate in best_options:
            # Check all differences
            new_diffs = set()
            valid = True
            for a in rows[best_row]:
                d = candidate - a
                if d in used:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Place the element
            rows[best_row].append(candidate)
            last[best_row] = candidate
            counts[best_row] += 1
            used.update(new_diffs)
            
            if backtrack(depth + 1):
                return True
            
            # Undo
            rows[best_row].pop()
            counts[best_row] -= 1
            last[best_row] = rows[best_row][-1] if rows[best_row] else 0
            used.difference_update(new_diffs)
        
        return False
    
    if backtrack(0):
        return rows
    
    # Fallback to greedy if backtracking fails
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
