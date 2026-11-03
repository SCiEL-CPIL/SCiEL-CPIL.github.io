"""Difference triangle set (4,7): Optimized CSP solver using bitsets for difference tracking."""
import random
import sys
import time
sys.setrecursionlimit(100000)

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Uses a bitset-based backtracking solver with strong pruning to minimize the maximum element (scope).
    """
    # Greedy baseline
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
    
    def solve_scope(scope, seed, max_nodes=500000):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        nodes = [0]
        
        def build_row(idx):
            nonlocal used_bits, nodes
            if nodes[0] > max_nodes:
                return False
            nodes[0] += 1
            
            if idx == n_rows:
                return True
            
            row = [0]
            row_diff_bits = 0
            
            # Symmetry breaking: first element of row 0 is 1 (smallest possible)
            # For subsequent rows, enforce that row[i][1] > row[i-1][1]
            if idx == 0:
                min_start = 1
            else:
                min_start = rows[idx - 1][1] + 1
            
            def rec(row, row_diff_bits, min_x):
                nonlocal used_bits, nodes
                if nodes[0] > max_nodes:
                    return False
                nodes[0] += 1
                
                if len(row) == k + 1:
                    used_bits |= row_diff_bits
                    rows.append(row)
                    if build_row(idx + 1):
                        return True
                    else:
                        rows.pop()
                        used_bits ^= row_diff_bits
                        return False

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
                        if len(candidates) >= 15:
                            break
                
                if not candidates:
                    return False
                
                # Heuristic: try smaller values first for tight packing
                # If few candidates, try all
                if len(candidates) <= 5:
                    for x in candidates:
                        new_diffs = 0
                        for a in row:
                            d = x - a
                            new_diffs |= (1 << d)
                        
                        if rec(row + [x], row_diff_bits | new_diffs, x + 1):
                            return True
                    return False
                else:
                    # Try smallest first, then random if it fails?
                    # For determinism in finding *a* solution, smallest is best.
                    # For exploring space, random is better.
                    # Let's stick to smallest for speed, as we have multiple seeds.
                    x = candidates[0]
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        new_diffs |= (1 << d)
                    
                    if rec(row + [x], row_diff_bits | new_diffs, x + 1):
                        return True
                    return False
            
            return rec(row, row_diff_bits, min_start)
        
        success = build_row(0)
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
        # Try several seeds with varied node budgets
        # For smaller scopes, we need more nodes to find the solution
        # Allocate more time/nodes to the first few attempts
        for seed in range(10):
            if time.time() - t0 > time_limit:
                break
            
            # Dynamic budget: more nodes for smaller scopes (harder)
            # and earlier seeds
            base_budget = 200000
            if scope < 140:
                base_budget = 500000
            if scope < 135:
                base_budget = 1000000
                
            # Decrease budget for later seeds to allow more attempts
            budget = max(50000, base_budget - seed * 50000)
            
            res = solve_scope(scope, seed, max_nodes=budget)
            if res:
                found = res
                break
        
        if found:
            best_found = found
            best_scope_found = scope
        else:
            pass

    return best_found
