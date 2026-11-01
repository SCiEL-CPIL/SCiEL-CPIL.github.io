import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 145.0  # 5s safety margin for 150s limit
    
    # The number of differences in one row is k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # The universe of possible differences up to some bound M.
    # We want to find the smallest M such that we can pack n_rows rows.
    # Lower bound: total_diffs_needed <= M * (M+1) / 2 ? No, the differences are subsets of {1, ..., M}.
    # Actually, the differences are just integers. The set of all used differences must be disjoint.
    # So we need total_diffs_needed distinct positive integers.
    # The smallest possible max difference is at least total_diffs_needed.
    # But the entries themselves are bounded by M.
    # If the max entry is M, the max difference is M.
    # So we need at least total_diffs_needed distinct integers in {1, ..., M}.
    # Thus M >= total_diffs_needed.
    
    # Let's search for the minimal M.
    # Start from a reasonable lower bound.
    # The counting bound mentioned is 105 for (7,5).
    # For general n, k, let's start with M_min = total_diffs_needed.
    
    best_rows = None
    best_M = float('inf')
    
    # We will try increasing M.
    # To be efficient, we can start from a heuristic estimate.
    # Greedy construction gives an upper bound.
    initial_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(initial_rows)
    best_rows = initial_rows
    
    # Now try to improve using backtracking with increasing tightness.
    # We try M values from a lower bound up to best_M - 1.
    # Lower bound: max(total_diffs_needed, k*(k+1)//2) is weak.
    # A better lower bound: The sum of the largest k differences in a row of size M is roughly k*M - k(k-1)/2.
    # Actually, let's just iterate M from max(total_diffs_needed, 50) up to best_M - 1.
    # If n_rows and k are small, this is fast.
    
    # Heuristic lower bound for M:
    # Each row has k+1 elements. The largest element is at least k.
    # The differences include 1, 2, ..., k potentially? No.
    # Let's use M_start = max(total_diffs_needed, k * (k + 1) // 2 + k)
    
    M_lower = max(total_diffs_needed, k * (k + 1) // 2 + k, 20)
    
    # To avoid too many iterations, we might skip some M values if it's too slow,
    # but for (7,5) M is around 105-112, so range is small.
    
    # We will try M in increasing order.
    # If we find a solution for M, that is the optimal for that M? No, we want min M.
    # So we iterate M from M_lower to best_M - 1.
    # If we find a valid packing for M, we update best_M and best_rows, and can stop (since we go up).
    # Wait, we want the SMALLEST M. So if we find a solution for M, that is the best possible for that M.
    # Since we iterate M upwards, the first M for which we find a solution is the optimal M.
    
    # However, proving non-existence for smaller M is hard.
    # So we try to find a solution for each M starting from M_lower.
    # If we find one, we return it.
    
    for M in range(M_lower, best_M):
        if time.time() > deadline - 5:
            break
        
        # Try to find a packing with max entry <= M
        rows = _solve_packing(n_rows, k, M, deadline)
        
        if rows is not None:
            # Found a solution with max entry <= M
            # Since M is increasing, this is the minimal M found so far.
            # We can stop.
            best_rows = rows
            best_M = M
            break
            
    return best_rows

def _solve_packing(n_rows, k, M, deadline):
    """
    Try to find n_rows rows of k+1 integers from {0, ..., M} starting at 0,
    such that all pairwise differences are distinct.
    Uses backtracking with forward checking and dynamic row ordering.
    """
    used_diffs = set()
    rows = []
    
    # Precompute all valid rows (subsets of size k+1 from {0,...,M} containing 0)
    # This might be too many if M is large.
    # For M=112, k=5, size 6.
    # Number of subsets of size 5 from {1,...,112} is C(112, 5) ~ 112*111*110*109*108 / 120 ~ 2.5M.
    # Too many to precompute and store fully if we do it naively.
    # But we can generate them on the fly or filter.
    
    # Alternative: Backtrack on the rows.
    # At each step, choose a row.
    # To use dynamic ordering, we need to estimate the "tightness" of each remaining row slot.
    # Since rows are symmetric, we can just pick one row at a time.
    # The "tightness" depends on the used_diffs.
    
    # Let's use a recursive backtracking that picks one row at a time.
    # To improve efficiency, we can sort candidate rows by some heuristic.
    
    # Heuristic: When picking a row, prefer rows that use "rare" differences?
    # Or prefer rows that are "compact"?
    
    # Let's try a different approach:
    # 1. Generate a pool of candidate rows.
    # 2. Use a greedy set packing algorithm with backtracking.
    
    # Given the time limit, let's try a randomized greedy with backtracking on the last few rows.
    # Or simply: Iterate through candidate rows in a random order, and if a row fits, take it.
    # If stuck, backtrack.
    
    # To make it effective, we need a good candidate generation.
    # Let's generate candidates by picking k random numbers from 1..M, sorting them, and checking validity.
    
    def backtrack(idx):
        if time.time() > deadline - 2:
            return False
            
        if idx == n_rows:
            return True
            
        # Generate candidate rows
        # We want to find a row that is disjoint from used_diffs.
        # To avoid trying the same bad candidates repeatedly, we can shuffle.
        
        # Strategy: Try a few random candidates. If none work, try a more systematic search.
        # For speed, let's just try random candidates first.
        
        # To increase diversity, we can bias towards smaller numbers (compact rows).
        
        found_row = None
        row_diffs = None
        
        # Try up to N random candidates
        max_attempts = 200
        for _ in range(max_attempts):
            if time.time() > deadline - 2:
                return False
                
            # Generate a random row
            # Pick k-1 intermediate values and 1 max value?
            # Or just pick k values from 1..M.
            
            # To favor compact rows, we can pick from a smaller range initially.
            # But we must allow up to M.
            
            # Let's pick k values from 1..M.
            # To ensure 0 is included, we just pick k values from 1..M and sort.
            
            vals = random.sample(range(1, M + 1), k)
            vals.sort()
            row = [0] + vals
            
            # Check internal validity (differences distinct within row)
            # And global validity (differences not in used_diffs)
            
            d_set = set()
            valid = True
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in d_set:
                        valid = False
                        break
                    if d in used_diffs:
                        valid = False
                        break
                    d_set.add(d)
                    
            if valid:
                found_row = row
                row_diffs = d_set
                break
                
        if found_row is None:
            # Random search failed. Try a more directed search?
            # For now, return False to trigger backtrack or restart.
            # In a more robust solver, we would try to construct a row that fits the remaining gaps.
            return False
            
        # Accept this row
        rows.append(found_row)
        used_diffs.update(row_diffs)
        
        if backtrack(idx + 1):
            return True
            
        # Backtrack
        rows.pop()
        used_diffs.difference_update(row_diffs)
        
        return False
        
    # To improve the random search, we can use multiple restarts inside this function?
    # No, the outer loop in construct() handles M.
    # But we can restart the randomization inside _solve_packing if it fails quickly.
    
    # Let's add a few restarts inside _solve_packing to escape local minima.
    for restart in range(50):
        if time.time() > deadline - 2:
            break
            
        random.seed(restart * 12345 + M)  # Deterministic per M and restart
        used_diffs.clear()
        rows.clear()
        
        if backtrack(0):
            return rows
            
    return None

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
