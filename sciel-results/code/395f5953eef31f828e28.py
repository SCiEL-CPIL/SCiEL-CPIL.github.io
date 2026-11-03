"""Shortest superpermutation via optimized block ordering with local search."""
import sys
import time
import random
from itertools import permutations

sys.setrecursionlimit(1000000)


def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
                elif o == 0:
                    overlap = 0
                    break
            new += block[overlap:]
        word = new
    return word


def _build_level_perms(n):
    """Build the list of (n-1)-length permutations in classic order."""
    if n == 1:
        return [[1]]
    word = _classic_superperm(n - 1)
    perms, seen = [], set()
    m = n - 1
    for i in range(len(word) - m + 1):
        window = tuple(word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
    return [list(p) for p in perms]


def _overlap(a, b):
    """Maximum overlap: suffix of a matches prefix of b."""
    max_ov = min(len(a), len(b))
    for o in range(max_ov, 0, -1):
        if a[-o:] == b[:o]:
            return o
    return 0


def _block_for_perm(p, m):
    """The block for permutation p with new symbol m: p + [m] + p."""
    return list(p) + [m] + list(p)


def _compute_overlap_matrix(blocks):
    """Precompute all-pairs overlap matrix."""
    n = len(blocks)
    ov_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                ov_matrix[i][j] = _overlap(blocks[i], blocks[j])
    return ov_matrix


def _total_length(order, blocks, ov_matrix):
    """Compute total length given an ordering."""
    if not order:
        return 0
    total = len(blocks[order[0]])
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        total += len(blocks[j]) - ov_matrix[i][j]
    return total


def _optimize_order_2opt(order, blocks, ov_matrix, time_limit=10.0, seed=42):
    """2-opt local search to minimize total length."""
    start_time = time.time()
    n_perms = len(order)
    best_order = list(order)
    best_cost = _total_length(best_order, blocks, ov_matrix)
    
    # 2-opt local search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_perms - 1):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n_perms):
                # Reverse segment [i+1..j]
                new_order = best_order[:i + 1] + best_order[i + 1:j + 1][::-1] + best_order[j + 1:]
                new_cost = _total_length(new_order, blocks, ov_matrix)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_order = new_order
                    improved = True
    
    return best_order, best_cost


def _optimize_order_3opt(order, blocks, ov_matrix, time_limit=10.0, seed=42):
    """3-opt local search: try all 3-opt moves."""
    start_time = time.time()
    n_perms = len(order)
    best_order = list(order)
    best_cost = _total_length(best_order, blocks, ov_matrix)
    
    # 3-opt local search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_perms - 2):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n_perms - 1):
                if (time.time() - start_time) > time_limit:
                    break
                for k in range(j + 1, n_perms):
                    # Try reversing segment [i+1..k]
                    new_order = best_order[:i + 1] + best_order[i + 1:k + 1][::-1] + best_order[k + 1:]
                    new_cost = _total_length(new_order, blocks, ov_matrix)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_order = new_order
                        improved = True
    
    return best_order, best_cost


def _optimize_order_hill_climb(order, blocks, ov_matrix, time_limit=10.0, seed=42):
    """Hill climbing with random restarts."""
    start_time = time.time()
    random.seed(seed)
    n_perms = len(order)
    best_order = list(order)
    best_cost = _total_length(best_order, blocks, ov_matrix)
    
    # Try several random restarts
    for restart in range(10):
        if (time.time() - start_time) > time_limit - 2.0:
            break
        
        # Random initial order
        cur_order = list(range(n_perms))
        random.shuffle(cur_order)
        cur_cost = _total_length(cur_order, blocks, ov_matrix)
        
        # Hill climb
        improved = True
        while improved and (time.time() - start_time) < time_limit - 1.0:
            improved = False
            # Try all 2-opt moves
            for i in range(n_perms - 1):
                if (time.time() - start_time) > time_limit - 1.0:
                    break
                for j in range(i + 1, n_perms):
                    new_order = cur_order[:i + 1] + cur_order[i + 1:j + 1][::-1] + cur_order[j + 1:]
                    new_cost = _total_length(new_order, blocks, ov_matrix)
                    if new_cost < cur_cost:
                        cur_cost = new_cost
                        cur_order = new_order
                        improved = True
        
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_order = cur_order
    
    return best_order, best_cost


def construct(n):
    """Return a string over '1'..'n' that contains every permutation of 1..n as a
    consecutive window. Minimises length; uses optimized block ordering."""
    deadline = time.time() + 115.0  # Safety margin under the 120s wall-clock kill

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341214321342143124312134214321"

    # Build the (n-1)-permutations in classic order
    perm_list = _build_level_perms(n)
    m = n
    n_perms = len(perm_list)
    
    # Build blocks
    blocks = [_block_for_perm(p, m) for p in perm_list]
    
    # Compute overlap matrix
    ov_matrix = _compute_overlap_matrix(blocks)
    
    # Classic length as baseline
    classic_word = _classic_superperm(n)
    classic_len = len(classic_word)
    
    # Try to optimize the ordering
    best_order = list(range(n_perms))
    best_cost = classic_len
    
    # Phase 1: 2-opt from classic order
    if (time.time() - start_time) < deadline - 30.0:
        time_for_2opt = min(20.0, deadline - time.time() - 30.0)
        if time_for_2opt > 1.0:
            best_order, best_cost = _optimize_order_2opt(best_order, blocks, ov_matrix, time_limit=time_for_2opt)
    
    # Phase 2: 3-opt refinement
    if (time.time() - start_time) < deadline - 20.0:
        time_for_3opt = min(15.0, deadline - time.time() - 20.0)
        if time_for_3opt > 1.0:
            best_order, best_cost = _optimize_order_3opt(best_order, blocks, ov_matrix, time_limit=time_for_3opt)
    
    # Phase 3: Hill climbing with random restarts
    if (time.time() - start_time) < deadline - 10.0:
        time_for_hc = min(20.0, deadline - time.time() - 10.0)
        if time_for_hc > 1.0:
            best_order, best_cost = _optimize_order_hill_climb(best_order, blocks, ov_matrix, time_limit=time_for_hc)
    
    # Build the word using the optimized order
    if best_cost < classic_len:
        word = list(blocks[best_order[0]])
        for k in range(1, len(best_order)):
            b = blocks[best_order[k]]
            ov = _overlap(word, b)
            word += b[ov:]
        
        # Verify completeness
        seen = set()
        for i in range(len(word) - n + 1):
            seen.add(tuple(word[i:i + n]))
        if len(seen) == len(perm_list):
            return "".join(map(str, word))
    
    # Fall back to classic
    return "".join(map(str, classic_word))
