"""Difference triangle set (4,7): Optimized CSP solver using bitsets for difference tracking."""
import random
import sys
import time
sys.setrecursionlimit(100000)

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Uses a bitset-based backtracking solver to minimize the maximum element (scope).
    """
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used |= (1 << d)
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    def solve_scope(scope, seed, max_nodes=300000):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        nodes = [0]
        
        def build_row(idx, min_start=1):
            nonlocal used_bits, nodes
            if nodes[0] > max_nodes:
                return False
            nodes[0] += 1
            
            if idx == n_rows:
                return True
            
            row = [0]
            row_diff_bits = 0
            
            def rec(row, row_diff_bits, min_x):
                nonlocal used_bits, nodes
                if nodes[0] > max_nodes:
                    return False
                nodes[0] += 1
                
                if len(row) == k + 1:
                    used_bits |= row_diff_bits
                    rows.append(row)
                    
                    # Symmetry breaking: next row must start with a value > current row's first non-zero
                    next_min_start = row[1] + 1 if idx < n_rows - 1 else 1
                    if build_row(idx + 1, next_min_start):
                        return True
                    else:
                        rows.pop()
                        used_bits ^= row_diff_bits
                        return False

                last = row[-1]
                remaining = (k + 1) - len(row)
                max_x = scope - (remaining - 1)
                
                if max_x < min_x:
                    return False

                blocked = used_bits | row_diff_bits
                candidates = []
                
                # Try candidates in increasing order
                for x in range(min_x, max_x + 1):
                    ok = True
                    for a in row:
                        d = x - a
                        if (blocked >> d) & 1:
                            ok = False
                            break
                    if ok:
                        candidates.append(x)
                        if len(candidates) >= 20:
                            break
                
                if not candidates:
                    return False
                
                # Heuristic: smaller values first to keep scope tight
                if seed % 2 == 0:
                    x = candidates[0]
                else:
                    x = rng.choice(candidates[:min(10, len(candidates))])
                
                new_diffs = 0
                for a in row:
                    d = x - a
                    new_diffs |= (1 << d)
                
                if rec(row + [x], row_diff_bits | new_diffs, x + 1):
                    return True
                return False
            
            return rec(row, row_diff_bits, min_start)
        
        success = build_row(0, 1)
        if success:
            return rows
        return None

    t0 = time.time()
    time_limit = 115
    best_found = best_rows
    best_scope_found = best_scope
    
    # Search downwards from best_scope - 1
    for scope in range(best_scope - 1, 99, -1):
        if time.time() - t0 > time_limit:
            break
            
        found = None
        # Try a few seeds to break symmetry/tie-breaks
        for seed in range(5):
            if time.time() - t0 > time_limit:
                break
            res = solve_scope(scope, seed, max_nodes=200000)
            if res:
                found = res
                break
        
        if found:
            best_found = found
            best_scope_found = scope
        else:
            # If no solution found for this scope, we can't guarantee lower scopes will work,
            # but we continue searching lower scopes as they are "tighter" and might have different structures?
            # Actually, if scope S is impossible, scope S-1 is also impossible for the same set of differences?
            # No, the set of differences changes. But generally, if we can't pack into S, we can't pack into S-1.
            # However, the "shape" of the rows might allow a smaller max element if we rearrange.
            # So we continue.
            pass

    return best_found
