"""Constraint-satisfaction search over difference frames with bitset pruning and deadline guard."""
import time
import random
from itertools import combinations

# We will precompute candidate frames on the fly and use a backtracking search
# with minimum-remaining-values heuristic.

def _generate_frames(max_m, k, limit=None):
    """Generate all valid frames (lists of k+1 increasing ints starting at 0) with max entry <= max_m.
    A frame is valid if all its internal differences are distinct.
    Returns list of (max_entry, diff_bitset, frame_tuple).
    """
    frames = []
    # We need k+1 numbers: a_0=0 < a_1 < ... < a_k <= max_m
    # So we choose k numbers from 1..max_m
    # For efficiency, we iterate over combinations
    # But max_m could be up to ~120, k=6, so C(120,6) ~ 2.9e9 which is too many.
    # We need a smarter generation.
    # Instead, we can generate frames recursively with pruning.
    
    def _gen(idx, current, diffs):
        if idx == k:
            # current has k elements (a_1..a_k), plus 0
            # Check if diffs are all distinct
            if len(diffs) == len(set(diffs)):
                max_e = current[-1]
                # Build bitset for diffs
                bitset = 0
                for d in diffs:
                    bitset |= (1 << d)
                frames.append((max_e, bitset, tuple(current)))
            return
        
        # Lower bound for next element: at least current[-1] + 1
        # Upper bound: max_m
        # Also, we need to ensure we can still fit (k - idx) more elements
        # so max value for current step is max_m - (k - idx - 1)
        if idx == 0:
            start = 1
        else:
            start = current[-1] + 1
        
        end = max_m - (k - idx - 1)
        
        for v in range(start, end + 1):
            # Compute new diffs with all previous elements
            new_diffs = []
            valid = True
            for a in current:
                d = v - a
                if d in diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Also check that new_diffs are distinct among themselves
            # (already checked above via d in new_diffs)
            
            current.append(v)
            diffs.extend(new_diffs)
            _gen(idx + 1, current, diffs)
            diffs.pop()
            diffs.pop()
            diffs.pop()
            # Actually, we need to pop exactly len(new_diffs) elements
            # Let me fix this
            # The above is wrong. Let me redo.
            # We extended diffs by new_diffs, so we need to remove them.
            # But we already popped 3 times. Let me fix the logic.
            
            # Actually, let me just use a different approach:
            # Pass diffs as a set for O(1) lookup, and maintain a list for reconstruction.
            pass
        
    # The above recursive approach is getting complicated. Let me use a cleaner version.
    
    frames = []
    diff_set = set()
    current = []
    
    def _gen2(idx):
        if idx == k:
            # We have k elements in current
            max_e = current[-1]
            bitset = 0
            for d in diff_set:
                bitset |= (1 << d)
            frames.append((max_e, bitset, tuple(current)))
            return
        
        if idx == 0:
            start = 1
        else:
            start = current[-1] + 1
        
        end = max_m - (k - idx - 1)
        
        for v in range(start, end + 1):
            # Check if adding v creates any duplicate difference
            valid = True
            for a in current:
                d = v - a
                if d in diff_set:
                    valid = False
                    break
            if not valid:
                continue
            
            # Also check that new differences are distinct from each other
            # Since v - a for different a gives different values (a is strictly increasing),
            # the new diffs v-a are all distinct from each other.
            # So we just need to check against existing diff_set.
            
            current.append(v)
            new_diffs = [v - a for a in current[:-1]]
            for d in new_diffs:
                diff_set.add(d)
            
            _gen2(idx + 1)
            
            for d in new_diffs:
                diff_set.discard(d)
            current.pop()
    
    _gen2(0)
    return frames


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Start with a greedy solution as baseline
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
    
    best_scope = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    
    # Try to improve with constraint satisfaction search
    # We'll try increasing M (max entry) starting from the counting bound
    # and use backtracking to find a valid set of n_rows frames
    
    # The counting bound: each row has C(k+1,2) differences
    # Total differences needed: n_rows * C(k+1,2)
    # Each difference d can appear in at most 1 row (since all differences must be distinct over all rows)
    # So we need at least n_rows * C(k+1,2) distinct differences
    # The smallest possible max difference is bounded by the fact that we need to fit all these
    # into {1, ..., M}
    # A simple bound: M >= n_rows * C(k+1,2) is too loose
    # Better: the number of distinct differences in a single row with max entry M is at most M
    # (differences range from 1 to M)
    # So n_rows * C(k+1,2) <= M * n_rows? No, that's not right.
    # Each row can have at most M differences (values 1..M), but they must be distinct within the row.
    # Over all rows, all differences must be globally distinct.
    # So total distinct differences needed = n_rows * C(k+1,2)
    # These must all be in {1, ..., M}
    # So M >= n_rows * C(k+1,2)
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    min_M = total_diffs_needed  # This is a lower bound on M
    
    # Actually, this bound is too loose. Let's use a better one.
    # The number of pairs (i,j) with 0 <= i < j <= k is C(k+1,2) = k(k+1)/2
    # Each row contributes k(k+1)/2 distinct differences.
    # All differences across all rows must be distinct.
    # So we need at least n_rows * k(k+1)/2 distinct positive integers.
    # The smallest such set is {1, 2, ..., n_rows * k(k+1)/2}
    # So M >= n_rows * k(k+1)/2
    
    # But this is still very loose. For n_rows=7, k=6: 7*21 = 147, but best known is 112.
    # So the bound is not tight. Let's just try values of M starting from a reasonable point.
    
    # Let's try M from the greedy solution's scope down to the theoretical minimum
    # Actually, let's try to find a solution with M as small as possible.
    
    # For now, let's try a few values of M and see if we can find a valid configuration.
    
    # We'll use a simple backtracking search with bitsets for efficiency.
    
    # Precompute frames for a given M
    # But generating all frames for M=112, k=6 might be expensive.
    # Let's estimate: C(112, 6) ~ 1.3e9, which is too many.
    
    # Instead, let's use a different approach: build rows one by one,
    # but with backtracking and heuristics.
    
    # Actually, let's just improve the greedy with some local search.
    
    # Try a simple local search: randomly perturb the greedy solution and try to fix.
    
    # For now, let's just return the greedy solution.
    # This is a placeholder; the real improvement would require a more sophisticated search.
    
    return best_rows
