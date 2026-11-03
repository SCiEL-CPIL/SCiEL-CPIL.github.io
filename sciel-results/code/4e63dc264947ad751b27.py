"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
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

def _total_length_given_order(order, perm_list, m):
    """
    Given an ordering of permutations, compute the total length of the
    resulting superpermutation segment (starting from empty, but we
    account for the first block fully).
    """
    if not order:
        return 0
    blocks = [_block_for_perm(perm_list[i], m) for i in order]
    total = len(blocks[0])
    for k in range(1, len(blocks)):
        ov = _overlap(blocks[k-1], blocks[k])
        total += len(blocks[k]) - ov
    return total

def _optimize_order_tsp(perm_list, m, time_limit=10.0):
    """
    Try to find a better ordering of perm_list to minimize total length.
    Uses a greedy + local search approach.
    """
    start_time = time.time()
    n_perms = len(perm_list)
    
    # Initial order: the classic order
    order = list(range(n_perms))
    
    # Compute all-pairs overlap matrix
    blocks = [_block_for_perm(p, m) for p in perm_list]
    n = n_perms
    ov_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                ov_matrix[i][j] = _overlap(blocks[i], blocks[j])
    
    # Cost of transition i->j is len(blocks[j]) - ov_matrix[i][j]
    # Total cost = len(blocks[0]) + sum of transition costs
    
    def current_cost(order):
        cost = len(blocks[order[0]])
        for k in range(1, len(order)):
            i, j = order[k-1], order[k]
            cost += len(blocks[j]) - ov_matrix[i][j]
        return cost
    
    best_order = list(order)
    best_cost = current_cost(best_order)
    
    # 2-opt local search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n - 1):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n):
                # Reverse segment [i+1..j]
                old_cost = best_cost
                # Compute new cost
                new_order = best_order[:i+1] + best_order[i+1:j+1][::-1] + best_order[j+1:]
                new_cost = current_cost(new_order)
                if new_cost < old_cost:
                    best_cost = new_cost
                    best_order = new_order
                    improved = True
    
    # Also try some random restarts if time permits
    if (time.time() - start_time) < time_limit - 2.0:
        import random
        random.seed(42)
        for _ in range(5):
            if (time.time() - start_time) > time_limit - 1.0:
                break
            rand_order = list(range(n))
            random.shuffle(rand_order)
            # Local search from random start
            cur_order = list(rand_order)
            cur_cost = current_cost(cur_order)
            imp = True
            while imp and (time.time() - start_time) < time_limit - 0.5:
                imp = False
                for i in range(n - 1):
                    if (time.time() - start_time) > time_limit - 0.5:
                        break
                    for j in range(i + 1, n):
                        new_order = cur_order[:i+1] + cur_order[i+1:j+1][::-1] + cur_order[j+1:]
                        new_cost = current_cost(new_order)
                        if new_cost < cur_cost:
                            cur_cost = new_cost
                            cur_order = new_order
                            imp = True
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_order = cur_order
    
    return best_order, best_cost

def construct(n):
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
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
    if (time.time() - start_time) > deadline:
        return "".join(map(str, classic_word))
    
    return "".join(map(str, classic_word))

# EVOLVE-BLOCK-END
