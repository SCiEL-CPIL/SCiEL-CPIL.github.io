"""Constraint-satisfaction backtracking over the difference set for difference-triangle sets.

Strategy (structurally different from the seed):
- The global constraint is that all positive intra-row differences across all rows
  are distinct. There are exactly D = n_rows * k * (k+1) / 2 required differences.
- Instead of building rows by placing integers sequentially, we:
    1. Choose a disjoint partition of a candidate difference pool into n_rows subsets,
       each of size C = k*(k+1)//2, using backtracking with dynamic variable ordering
       (smallest remaining subset first, i.e. MRV-like).
    2. For each subset, sort it and reconstruct the row via prefix sums:
       row[0]=0, row[i]=row[i-1]+d_i where d_i is the i-th smallest difference.
    3. Validate that the reconstructed row's full difference set equals the subset.
- We try multiple random candidate pools (subsets of [1..M]) and backtrack to find
  a valid partition. After a valid set is found, we apply a fast right-to-left
  compaction to reduce the scope (max entry).
- This is a CSP over difference assignments, not a sequential integer placement.
"""

import time
import random
import sys
from itertools import combinations


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 117.0
    m = k + 1  # entries per row
    C = k * (k + 1) // 2  # differences per row
    D = n_rows * C  # total differences

    # Simple deterministic greedy fallback
    def greedy_fallback():
        u, out = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < m:
                x += 1
                nd = {x - a for a in row}
                if not (nd & u):
                    row.append(x)
                    u |= nd
            out.append(row)
        return out

    def validate(rows):
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(m):
                for j in range(i + 1, m):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def scope_of(rows):
        return max(max(r) for r in rows)

    def compact(rows):
        """Right-to-left compaction: reduce each element to its minimum feasible value."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.add(row[j] - row[i])

        improved = True
        max_passes = 30
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    if current_val <= row[pos - 1] + 1:
                        continue
                    # Try to reduce current_val
                    for new_val in range(row[pos - 1] + 1, current_val):
                        # Compute old and new differences
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(current_val - row[i])
                        for j in range(pos + 1, len(row)):
                            old_diffs.add(row[j] - current_val)

                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for j in range(pos + 1, len(row)):
                            new_diffs.add(row[j] - new_val)

                        # Check conflicts: new_diffs not in (all_diffs - old_diffs)
                        conflict = False
                        for d in new_diffs:
                            if d in old_diffs:
                                continue
                            if d in all_diffs:
                                conflict = True
                                break
                        if not conflict:
                            # Apply
                            for d in old_diffs:
                                all_diffs.discard(d)
                            for d in new_diffs:
                                all_diffs.add(d)
                            row[pos] = new_val
                            improved = True
                            break
        return rows

    def reconstruct_row(diff_subset):
        """Given a set of C differences, reconstruct the row via prefix sums.
        Returns the row list, or None if the full difference set doesn't match."""
        if len(diff_subset) != C:
            return None
        d_sorted = sorted(diff_subset)
        row = [0]
        for d in d_sorted:
            row.append(row[-1] + d)
        # Validate: the full difference set of the reconstructed row must equal diff_subset
        full = set()
        for i in range(m):
            for j in range(i + 1, m):
                full.add(row[j] - row[i])
        if full == diff_subset:
            return row
        return None

    def backtrack_assign(pool, n_rows, C, deadline):
        """Assign disjoint subsets of size C from pool to n_rows.
        pool is a sorted list of candidate differences.
        Returns list of subsets (each a set) or None."""
        if n_rows == 0:
            return []
        if len(pool) < n_rows * C:
            return None
        if time.time() > deadline:
            return None

        # MRV: try to assign the first (smallest) subset first
        # To keep it tractable, we use a heuristic: pick the smallest C elements
        # as a candidate, but also try variations.
        
        # For efficiency, we do a simple recursive backtracking with pruning.
        # State: index of row to assign, remaining pool.
        
        def rec(idx, remaining_pool):
            if time.time() > deadline:
                return None
            if idx == n_rows:
                return []
            if len(remaining_pool) < (n_rows - idx) * C:
                return None
            
            # Try to form a subset of size C from remaining_pool
            # Heuristic: try subsets that include small elements first
            # To limit branching, we try a few candidate subsets
            
            # Candidate 1: first C elements
            # Candidate 2: first C-1 elements + one larger
            # We'll iterate over possible subsets in a controlled way
            
            # For simplicity and speed, we try the "greedy" subset first,
            # then a few random ones if the first fails.
            
            candidates = []
            # Greedy: first C
            candidates.append(frozenset(remaining_pool[:C]))
            
            # A few random candidates
            if len(remaining_pool) > C:
                for _ in range(3):
                    if len(remaining_pool) <= C:
                        break
                    idxs = random.sample(range(len(remaining_pool)), C)
                    candidates.append(frozenset(remaining_pool[i] for i in idxs))
            
            # Deduplicate
            seen_cands = set()
            unique_cands = []
            for c in candidates:
                if c not in seen_cands:
                    seen_cands.add(c)
                    unique_cands.append(c)
            
            for subset in unique_cands:
                # Check if this subset can form a valid row
                row = reconstruct_row(set(subset))
                if row is None:
                    continue
                
                # Remove subset from pool
                remaining_set = set(remaining_pool) - set(subset)
                remaining_list = sorted(remaining_set)
                
                sub_result = rec(idx + 1, remaining_list)
                if sub_result is not None:
                    return [subset] + sub_result
            
            return None
        
        return rec(0, pool)

    # Initial best: greedy fallback
    best_rows = greedy_fallback()
    best_scope = scope_of(best_rows)
    
    rng = random.Random(42)
    
    # Try multiple candidate pools
    # Pool size: we need D differences. Start with a pool of size D + margin.
    # The margin allows for backtracking to skip some differences.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Determine pool size
        # Start with D + extra, where extra grows with attempts
        extra = min(20 + attempts * 2, 100)
        pool_size = D + extra
        
        # Generate a random pool of differences from [1..pool_size]
        # But we want to bias towards smaller differences to keep scope low
        # Use a mixture: 70% from [1..D], 30% from [D..pool_size]
        pool = set()
        while len(pool) < pool_size:
            if random.random() < 0.7 and D >= 1:
                pool.add(rng.randint(1, D))
            else:
                pool.add(rng.randint(1, pool_size))
        
        pool_list = sorted(pool)
        
        # Backtrack to assign subsets
        result = backtrack_assign(pool_list, n_rows, C, deadline)
        
        if result is not None:
            # Reconstruct rows
            rows = []
            for subset in result:
                row = reconstruct_row(set(subset))
                if row is None:
                    break
                rows.append(row)
            else:
                # All rows reconstructed
                if validate(rows):
                    # Compact
                    rows = compact(rows)
                    scope = scope_of(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = rows
                    if best_scope <= D:  # Can't do better than counting bound
                        break
        
        # Safety: if we've tried many and not improving, break
        if attempts > 200:
            break
    
    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
