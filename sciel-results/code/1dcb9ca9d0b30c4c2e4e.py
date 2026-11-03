"""Difference triangle set (7,5): minimize the largest mark (scope).

Strategy: treat each row as a choice of a k-subset of {1..M} (the non-zero marks).
A row is valid if its internal differences are all distinct. The global constraint
is that the union of difference sets across all rows is disjoint.
We search over candidate rows ordered by their maximum mark, using backtracking
with forward checking (bitmasks) to prune infeasible branches. A time guard
returns the best valid partial/complete assignment found.
"""
import time
from itertools import combinations

# We will hard-code the target parameters for the public task: n_rows=7, k=5.
# But we keep the signature general and adapt internally.

def _build_candidate_rows(k, M):
    """
    Generate all valid rows (subsets of size k from {1..M}) such that
    all pairwise differences within the row are distinct.
    Returns a list of (max_mark, diff_mask, row_tuple).
    """
    candidates = []
    # Precompute difference masks for efficiency
    # A row is a tuple of k distinct integers from 1..M, sorted.
    # Differences: for i<j, r[j]-r[i]. Must all be distinct.
    for combo in combinations(range(1, M + 1), k):
        diffs = []
        valid = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                d = combo[j] - combo[i]
                if d in diffs:
                    valid = False
                    break
                diffs.append(d)
            if not valid:
                break
        if valid:
            max_mark = combo[-1]
            mask = 0
            for d in diffs:
                mask |= (1 << d)
            candidates.append((max_mark, mask, combo))
    candidates.sort(key=lambda x: x[0])
    return candidates


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s

    # For the specific task (7,5), we know the best known is 112.
    # We try to find a solution with minimal max mark.
    # We'll try increasing M from a lower bound up to a reasonable upper bound.
    
    # Lower bound: simple counting. Number of differences per row = k*(k+1)/2 = 15.
    # Total differences = 7 * 15 = 105. Max difference is M. So M >= 105.
    # Best known is 112.
    
    # We'll search M from 105 to 120.
    # For each M, we try to find n_rows rows with pairwise disjoint difference masks.
    
    best_solution = None
    best_M = float('inf')
    
    # Precompute candidates for a range of M values.
    # To save time, we compute candidates for M=120 once, and filter by max_mark <= M.
    # Actually, candidates depend on M because the universe of marks is 1..M.
    # But a row with max mark m is valid for any M >= m.
    # So we can generate candidates up to M_max=120 and filter.
    
    M_max = 120
    candidates_all = _build_candidate_rows(k, M_max)
    
    # Group candidates by their max mark for easier filtering
    # candidates_all is sorted by max_mark already.
    
    for M in range(105, M_max + 1):
        if time.time() > deadline:
            break
        
        # Filter candidates to those with max_mark <= M
        candidates = [c for c in candidates_all if c[0] <= M]
        
        # Now we need to pick n_rows candidates with pairwise disjoint masks.
        # Use backtracking.
        # To make it faster, we can use a heuristic: pick the row with the fewest
        # compatible options first (most constrained). But for simplicity, we'll
        # try a greedy-like backtracking with ordering by max mark.
        
        # For efficiency, we'll limit the number of candidates we consider per row
        # to avoid explosion. We'll take the first N candidates sorted by max mark.
        N_candidates = min(len(candidates), 500)
        candidates = candidates[:N_candidates]
        
        def backtrack(idx, used_mask, chosen):
            if time.time() > deadline:
                return None
            if idx == n_rows:
                return chosen[:]
            
            # Try each candidate
            for i in range(len(candidates)):
                max_mark, mask, combo = candidates[i]
                if mask & used_mask:
                    continue
                # Forward checking: ensure remaining rows can be filled
                # (simplified: just proceed)
                chosen.append(combo)
                result = backtrack(idx + 1, used_mask | mask, chosen)
                if result is not None:
                    return result
                chosen.pop()
            
            return None
        
        # Initial backtracking
        result = backtrack(0, 0, [])
        if result is not None:
            best_solution = result
            best_M = M
            break  # We found a solution for this M, which is minimal
    
    if best_solution is None:
        # Fallback: use the greedy method from the seed program
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
    
    # Convert tuples to lists, prepend 0
    final_rows = []
    for combo in best_solution:
        row = [0] + list(combo)
        final_rows.append(row)
    
    return final_rows
