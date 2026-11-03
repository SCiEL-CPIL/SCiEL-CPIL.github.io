"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from functools import lru_cache

# EVOLVE-BLOCK-START

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
            max_o = min(len(new), len(block))
            for o in range(max_o, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _build_level_perms(n):
    """
    Build the list of (n-1)-length permutations in the order they appear
    in the classic superpermutation of level n-1.
    """
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
    """Compute all-pairs overlap matrix efficiently."""
    n = len(blocks)
    ov_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                ov_matrix[i][j] = _overlap(blocks[i], blocks[j])
    return ov_matrix

def _order_cost(order, blocks, ov_matrix):
    """Compute total length for a given order."""
    if not order:
        return 0
    cost = len(blocks[order[0]])
    for k in range(1, len(order)):
        i, j = order[k-1], order[k]
        cost += len(blocks[j]) - ov_matrix[i][j]
    return cost

def _two_opt_pass(order, blocks, ov_matrix, deadline, max_rounds=50):
    """2-opt local search with deadline guard."""
    n = len(order)
    best_order = list(order)
    best_cost = _order_cost(best_order, blocks, ov_matrix)
    
    for _ in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                # Reverse segment [i+1..j]
                new_order = best_order[:i+1] + best_order[i+1:j+1][::-1] + best_order[j+1:]
                new_cost = _order_cost(new_order, blocks, ov_matrix)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_order = new_order
                    improved = True
        if not improved:
            break
    return best_order, best_cost

def _or_opt_pass(order, blocks, ov_matrix, deadline, max_rounds=20):
    """Or-opt: move a segment of length 1-3 to a different position."""
    n = len(order)
    best_order = list(order)
    best_cost = _order_cost(best_order, blocks, ov_matrix)
    
    for _ in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        for seg_len in [1, 2, 3]:
            if time.time() >= deadline:
                break
            for i in range(n - seg_len):
                if time.time() >= deadline:
                    break
                seg = best_order[i:i+seg_len]
                for j in range(n):
                    if time.time() >= deadline:
                        break
                    if j <= i or j >= i + seg_len:
                        continue
                    # Remove seg and insert at position j
                    new_order = best_order[:i] + best_order[i+seg_len:]
                    insert_pos = j if j < i else j - seg_len
                    new_order = new_order[:insert_pos] + seg + new_order[insert_pos:]
                    new_cost = _order_cost(new_order, blocks, ov_matrix)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_order = new_order
                        improved = True
        if not improved:
            break
    return best_order, best_cost

def _optimize_order_tsp(perm_list, m, time_limit=10.0):
    """
    Try to find a better ordering of perm_list to minimize total length.
    Uses greedy + 2-opt + Or-opt local search with random restarts.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    n_perms = len(perm_list)
    
    # Initial order: the classic order
    blocks = [_block_for_perm(p, m) for p in perm_list]
    n = n_perms
    ov_matrix = _compute_overlap_matrix(blocks)
    
    best_order = list(range(n))
    best_cost = _order_cost(best_order, blocks, ov_matrix)
    
    # 2-opt local search from classic order
    best_order, best_cost = _two_opt_pass(best_order, blocks, ov_matrix, deadline)
    
    # Or-opt polish
    best_order, best_cost = _or_opt_pass(best_order, blocks, ov_matrix, deadline)
    
    # Random restarts if time permits
    random.seed(42)
    num_restarts = 0
    while time.time() < deadline - 2.0:
        num_restarts += 1
        rand_order = list(range(n))
        random.shuffle(rand_order)
        
        # 2-opt from random start
        cur_order, cur_cost = _two_opt_pass(rand_order, blocks, ov_matrix, deadline)
        # Or-opt polish
        cur_order, cur_cost = _or_opt_pass(cur_order, blocks, ov_matrix, deadline)
        
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_order = cur_order
        
        # Stop if we've done enough restarts or time is tight
        if num_restarts >= 10:
            break
    
    return best_order, best_cost

def construct(n):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "1"
    
    if n == 2:
        return "121"
    
    if n == 3:
        return "123121321"
    
    # For n >= 4, use optimized construction
    # Build the (n-1)-permutations in classic order
    perm_list = _build_level_perms(n)
    m = n
    n_perms = len(perm_list)
    
    # Get classic length as baseline
    classic_word = _classic_superperm(n)
    classic_len = len(classic_word)
    
    # Try to optimize the ordering at the top level
    if n_perms <= 720 and (time.time() - start_time) < deadline - 30.0:
        time_limit_for_tsp = min(20.0, deadline - (time.time() - start_time) - 30.0)
        if time_limit_for_tsp > 1.0:
            best_order, best_cost = _optimize_order_tsp(perm_list, m, time_limit=time_limit_for_tsp)
            
            # Build the word using the optimized order
            blocks = [_block_for_perm(perm_list[i], m) for i in best_order]
            word = list(blocks[0])
            for k in range(1, len(blocks)):
                b = blocks[k]
                ov = _overlap(word, b)
                word += b[ov:]
            
            if len(word) < classic_len:
                return "".join(map(str, word))
    
    # Fall back to classic
    return "".join(map(str, classic_word))

# EVOLVE-BLOCK-END
