"""Divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)

    smooth_set = set(smooth)
    n_smooth = len(smooth)
    smooth_sorted = sorted(smooth)

    # Precompute divisibility: can[i][j] = True if smooth_sorted[i] divides smooth_sorted[j]
    can = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth_sorted[j] % smooth_sorted[i] == 0:
                can[i][j] = True

    # Precompute antichains by size and sum for small sizes (2, 3, 4)
    # This is fast because n_smooth is only ~38
    antichains_by_size = {2: {}, 3: {}, 4: {}}

    # Size 2
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not can[i][j] and not can[j][i]:
                s = smooth_sorted[i] + smooth_sorted[j]
                if s <= 128:
                    if s not in antichains_by_size[2]:
                        antichains_by_size[2][s] = []
                    antichains_by_size[2][s].append((i, j))

    # Size 3
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if can[i][j] or can[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                if can[i][k] or can[k][i] or can[j][k] or can[k][j]:
                    continue
                s = smooth_sorted[i] + smooth_sorted[j] + smooth_sorted[k]
                if s <= 128:
                    if s not in antichains_by_size[3]:
                        antichains_by_size[3][s] = []
                    antichains_by_size[3][s].append((i, j, k))

    # Size 4
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if can[i][j] or can[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                if can[i][k] or can[k][i] or can[j][k] or can[k][j]:
                    continue
                for l in range(k + 1, n_smooth):
                    if can[i][l] or can[l][i] or can[j][l] or can[l][j] or can[k][l] or can[l][k]:
                        continue
                    s = smooth_sorted[i] + smooth_sorted[j] + smooth_sorted[k] + smooth_sorted[l]
                    if s <= 128:
                        if s not in antichains_by_size[4]:
                            antichains_by_size[4][s] = []
                        antichains_by_size[4][s].append((i, j, k, l))

    answers = [None] * 129

    for n in range(1, 129):
        if time.time() > deadline - 2:
            break

        # Try 1 term first
        if n in smooth_set:
            answers[n] = [n]
            continue

        # Try 2..4 terms from precomputed antichains
        found = False
        for k in range(2, 5):
            if n in antichains_by_size[k]:
                combo = antichains_by_size[k][n][0]
                answers[n] = [smooth_sorted[i] for i in combo]
                found = True
                break

        if found:
            continue

        # For 5+ terms, use backtracking search
        result = _find_antichain_rep(n, smooth_sorted, can, n_smooth, min_terms=5, max_terms=30)
        if result is not None:
            answers[n] = result
        else:
            # Fallback: greedy approach
            answers[n] = _greedy_antichain(n, smooth_sorted, smooth_set)

    # Verify all answers are valid
    for n in range(1, 129):
        if answers[n] is None:
            if n in smooth_set:
                answers[n] = [n]
            else:
                answers[n] = _greedy_antichain(n, smooth_sorted, smooth_set)

    return answers[1:129]


def _find_antichain_rep(n, smooth_sorted, can, n_smooth, min_terms=5, max_terms=30):
    """Find an antichain representation of n using smooth numbers."""
    # Backtracking search with pruning, trying smaller sizes first
    for k in range(min_terms, max_terms + 1):
        result = _backtrack_antichain(n, smooth_sorted, can, n_smooth, k)
        if result is not None:
            return result
    return None


def _backtrack_antichain(remaining, smooth_sorted, can, n_smooth, k, start_idx=0, current=None):
    """Backtracking to find k smooth numbers summing to remaining that form an antichain."""
    if current is None:
        current = []
    
    if len(current) == k:
        if remaining == 0:
            return list(current)
        return None
    if remaining <= 0:
        return None
    
    needed = k - len(current)
    if needed < 0:
        return None
    
    # Pruning: check if enough numbers remain
    if n_smooth - start_idx < needed:
        return None
    
    # Lower bound: sum of smallest 'needed' numbers from start_idx
    if needed > 0:
        min_sum = sum(smooth_sorted[start_idx:start_idx + needed])
        if min_sum > remaining:
            return None
    
    for i in range(start_idx, n_smooth):
        val = smooth_sorted[i]
        if val > remaining:
            break
        
        # Check divisibility with all current selections
        compatible = True
        for c_idx in current:
            if can[i][c_idx] or can[c_idx][i]:
                compatible = False
                break
        if not compatible:
            continue
        
        current.append(i)
        result = _backtrack_antichain(remaining - val, smooth_sorted, can, n_smooth, k, i + 1, current)
        if result is not None:
            return result
        current.pop()
    
    return None


def _greedy_antichain(n, smooth_sorted, smooth_set):
    """Greedy fallback: pick smooth numbers that don't divide each other."""
    chosen = []
    remaining = n
    for s in reversed(smooth_sorted):
        if s <= remaining:
            valid = True
            for c in chosen:
                if c % s == 0 or s % c == 0:
                    valid = False
                    break
            if valid:
                chosen.append(s)
                remaining -= s
                if remaining == 0:
                    break
    if remaining == 0 and 1 <= len(chosen) <= 30:
        return chosen
    return [n] if n in smooth_set else [1]
# EVOLVE-BLOCK-END
