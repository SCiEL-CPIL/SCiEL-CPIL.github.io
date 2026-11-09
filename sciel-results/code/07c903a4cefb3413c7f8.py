"""Difference triangle set (7,5) - compatibility graph + backtracking clique search.
   For a given scope M, generate all valid rows, build compatibility graph,
   and find a clique of size n_rows via backtracking with pruning.
   Binary search / linear scan on M to minimize scope.
"""
import time
import random
import math
import sys
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 148.0


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _scope(rows):
    if not rows:
        return float('inf')
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if rows is None or len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        n = len(r)
        ds = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in ds:
                    return False
                ds.add(d)
        if ds & used:
            return False
        used |= ds
    return True


def _diff_set(row):
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _generate_valid_rows(k, M):
    """Generate all valid rows of k+1 entries starting at 0, max <= M.
    A row is valid if all k*(k+1)/2 pairwise differences are distinct within the row."""
    rows = []
    n = k + 1
    
    if M < k:
        return rows
    
    def _gen(pos, current, diffs_so_far):
        if _dead():
            return
        if pos == k:
            # Check all diffs are distinct
            if len(diffs_so_far) == k * (k + 1) // 2 and len(set(diffs_so_far)) == len(diffs_so_far):
                rows.append([0] + list(current))
            return
        
        start = current[-1] + 1 if current else 1
        for v in range(start, M + 1):
            new_diffs = diffs_so_far.copy()
            for a in current:
                new_diffs.append(v - a)
            # Pruning: if any diff repeats, skip
            if len(new_diffs) != len(set(new_diffs)):
                continue
            _gen(pos + 1, current + [v], new_diffs)
    
    _gen(0, [], [])
    return rows


def _clique_search(n_rows, all_rows, all_diffs, compat, time_budget=10.0):
    """Backtracking search for a clique of size n_rows."""
    deadline = time.time() + time_budget
    n = len(all_rows)
    
    if n < n_rows:
        return None
    
    # Sort by scope
    order = sorted(range(n), key=lambda i: all_rows[i][-1])
    
    # Convert to list of indices
    # Use the order for candidate lists
    
    def _search(chosen, cand_list, depth):
        if time.time() > deadline:
            return None
        if depth == n_rows:
            return chosen[:]
        
        for ci in range(len(cand_list)):
            idx = cand_list[ci]
            
            # Check compatibility with all chosen
            ok = True
            for c in chosen:
                if not compat[idx][c]:
                    ok = False
                    break
            if not ok:
                continue
            
            chosen.append(idx)
            
            # New candidates: subset of cand_list[ci+1:] compatible with idx
            new_cands = []
            for cj in range(ci + 1, len(cand_list)):
                idx2 = cand_list[cj]
                if compat[idx][idx2]:
                    new_cands.append(idx2)
            
            if len(new_cands) >= n_rows - depth - 1:  # pruning
                res = _search(chosen, new_cands, depth + 1)
                if res is not None:
                    return res
            
            chosen.pop()
        
        return None
    
    return _search([], order, 0)


def _greedy_clique(n_rows, all_rows, all_diffs, compat, seed=42):
    """Greedy: pick rows one by one, preferring those with smallest scope."""
    rnd = random.Random(seed)
    n = len(all_rows)
    
    # Sort by scope
    order = sorted(range(n), key=lambda i: all_rows[i][-1])
    
    used_diffs = set()
    chosen = []
    
    for idx in order:
        if len(chosen) >= n_rows:
            break
        ds = all_diffs[idx]
        if ds & used_diffs:
            continue
        # Also check compatibility with all chosen
        ok = True
        for c in chosen:
            if not compat[idx][c]:
                ok = False
                break
        if ok:
            used_diffs |= ds
            chosen.append(idx)
    
    if len(chosen) == n_rows:
        return [all_rows[i] for i in chosen]
    return None


def _construct(n_rows, k):
    # Lower bound
    lb = n_rows * k * (k + 1) // 2
    
    # Phase 1: Generate rows for a range of M values and find a solution
    # Start from a reasonable upper bound and work down
    
    # First, try greedy at various M to establish an upper bound
    best_rows = None
    best_M = None
    
    # Try M from lb upward with greedy
    for M in range(lb, lb + 50):
        if _dead():
            break
        all_rows = _generate_valid_rows(k, M)
        if len(all_rows) < n_rows:
            continue
        
        all_diffs = [_diff_set(r) for r in all_rows]
        n = len(all_rows)
        
        # Build compatibility
        compat = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if not (all_diffs[i] & all_diffs[j]):
                    compat[i][j] = True
                    compat[j][i] = True
        
        # Try greedy
        res = _greedy_clique(n_rows, all_rows, all_diffs, compat, 42)
        if res is not None:
            best_rows = res
            best_M = M
            break
        
        res = _greedy_clique(n_rows, all_rows, all_diffs, compat, 123)
        if res is not None:
            best_rows = res
            best_M = M
            break
    
    if best_rows is None:
        # Try higher M
        for M in range(lb + 50, lb + 120):
            if _dead():
                break
            all_rows = _generate_valid_rows(k, M)
            if len(all_rows) < n_rows:
                continue
            
            all_diffs = [_diff_set(r) for r in all_rows]
            n = len(all_rows)
            
            compat = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if not (all_diffs[i] & all_diffs[j]):
                        compat[i][j] = True
                        compat[j][i] = True
            
            res = _greedy_clique(n_rows, all_rows, all_diffs, compat, 42)
            if res is not None:
                best_rows = res
                best_M = M
                break
    
    if best_rows is None:
        # Absolute fallback
        rows = []
        for i in range(n_rows):
            offset = i * 300
            row = [offset + j * 50 for j in range(k + 1)]
            row = [x - row[0] for x in row]
            rows.append(row)
        return rows
    
    if _dead():
        return best_rows
    
    # Phase 2: Try to find smaller M using exact clique search
    remaining = _DEADLINE - time.time()
    
    for M in range(lb, best_M):
        if _dead() or remaining < 5.0:
            break
        
        all_rows = _generate_valid_rows(k, M)
        if len(all_rows) < n_rows:
            continue
        
        all_diffs = [_diff_set(r) for r in all_rows]
        n = len(all_rows)
        
        compat = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if not (all_diffs[i] & all_diffs[j]):
                    compat[i][j] = True
                    compat[j][i] = True
        
        time_budget = min(remaining - 3.0, 10.0)
        if time_budget < 2.0:
            break
        
        # Try exact search
        res = _clique_search(n_rows, all_rows, all_diffs, compat, time_budget)
        if res is not None:
            best_rows = [all_rows[i] for i in res]
            best_M = M
            remaining = _DEADLINE - time.time()
            # Don't break - try even smaller M
        else:
            remaining = _DEADLINE - time.time()
    
    # Final validation
    if best_rows is not None and not _valid(best_rows, n_rows, k):
        rows = []
        for i in range(n_rows):
            offset = i * 300
            row = [offset + j * 50 for j in range(k + 1)]
            row = [x - row[0] for x in row]
            rows.append(row)
        best_rows = rows
    
    return best_rows


def construct(n_rows, k):
    return _construct(n_rows, k)
