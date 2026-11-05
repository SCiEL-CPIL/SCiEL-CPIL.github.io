import time
import random
from itertools import permutations

_START_TIME = time.time()
DEADLINE = 145.0  # Safety margin for 150s limit

def _remaining_time():
    return DEADLINE - (time.time() - _START_TIME)

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _greedy_nn_init(perms, overlap_matrix):
    """Greedy nearest-neighbor initialization for permutation ordering."""
    n_perms = len(perms)
    visited = [False] * n_perms
    path = [0]
    visited[0] = True
    
    for _ in range(n_perms - 1):
        last = path[-1]
        best_next = -1
        best_overlap = -1
        for j in range(n_perms):
            if not visited[j]:
                ov = overlap_matrix[last][j]
                if ov > best_overlap:
                    best_overlap = ov
                    best_next = j
        path.append(best_next)
        visited[best_next] = True
    
    return path

def _standard_construct(n):
    """Classic recursive construction: 1! + 2! + ... + n!"""
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _overlap_len(a, b):
    """Longest suffix of a that is a prefix of b."""
    max_ov = min(len(a), len(b))
    for ov in range(max_ov, 0, -1):
        if a[-ov:] == b[:ov]:
            return ov
    return 0

def _construct_from_order(order, n):
    """Given an ordered list of (n-1)-permutations, build the superpermutation."""
    if not order:
        return list(range(1, n + 1))
    
    word = list(order[0]) + [n] + list(order[0])
    
    for i in range(1, len(order)):
        prev = order[i - 1]
        curr = order[i]
        block = list(curr) + [n] + list(curr)
        ov = 0
        max_ov = min(len(prev), len(curr))
        for o in range(max_ov, 0, -1):
            if prev[-o:] == curr[:o]:
                ov = o
                break
        word += block[ov:]
    
    return word

def _verify_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive windows."""
    if len(word) < n:
        return False
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        found.add(window)
    return len(found) == _factorial(n)

def _get_standard_order(n):
    """Extract the order of (n-1)-permutations from the standard construction."""
    word_n_minus_1 = _standard_construct(n - 1)
    order = []
    seen = set()
    for i in range(len(word_n_minus_1) - (n - 1) + 1):
        window = tuple(word_n_minus_1[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            order.append(window)
    return order

def _compute_overlap_matrix(perms):
    """Precompute overlap matrix for all pairs of permutations."""
    n_perms = len(perms)
    overlap_matrix = [[0] * n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        pi = list(perms[i])
        for j in range(n_perms):
            if i != j:
                pj = list(perms[j])
                max_ov = min(len(pi), len(pj))
                ov = 0
                for o in range(max_ov, 0, -1):
                    if pi[-o:] == pj[:o]:
                        ov = o
                        break
                overlap_matrix[i][j] = ov
    return overlap_matrix

def _total_overlap(path, overlap_matrix):
    """Calculate total overlap for a given path."""
    total = 0
    for i in range(len(path) - 1):
        total += overlap_matrix[path[i]][path[i+1]]
    return total

def _two_opt_improve(path, overlap_matrix, max_iterations=50):
    """2-opt local search to maximize total overlap with efficient delta computation."""
    n_perms = len(path)
    best_path = path[:]
    best_score = _total_overlap(best_path, overlap_matrix)
    
    for iteration in range(max_iterations):
        if _remaining_time() < 5.0:
            break
        improved = False
        for i in range(n_perms - 1):
            for j in range(i + 1, n_perms):
                # Compute delta for reversing segment [i+1..j]
                # Affected edges: (i, i+1), (j, j+1 if exists), and internal edges
                old_score = 0
                new_score = 0
                
                # Edge before segment: (i, i+1)
                old_score += overlap_matrix[best_path[i]][best_path[i+1]]
                # Edge after segment: (j, j+1) if j+1 exists
                if j + 1 < n_perms:
                    old_score += overlap_matrix[best_path[j]][best_path[j+1]]
                    new_score += overlap_matrix[best_path[i]][best_path[j]]
                    new_score += overlap_matrix[best_path[i+1]][best_path[j+1]]
                else:
                    new_score += overlap_matrix[best_path[i]][best_path[j]]
                
                # Internal edges in segment [i+1..j]
                for k in range(i+1, j):
                    old_score += overlap_matrix[best_path[k]][best_path[k+1]]
                    new_score += overlap_matrix[best_path[k+1]][best_path[k]]
                
                delta = new_score - old_score
                if delta > 0:
                    best_path[i+1:j+1] = best_path[i+1:j+1][::-1]
                    best_score += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_path

def _or_opt_improve(path, overlap_matrix, max_iterations=50):
    """Or-opt local search: move segments of length 1-3 with delta computation."""
    n_perms = len(path)
    best_path = path[:]
    best_score = _total_overlap(best_path, overlap_matrix)
    
    for iteration in range(max_iterations):
        if _remaining_time() < 5.0:
            break
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(n_perms - seg_len):
                segment = best_path[i:i+seg_len]
                # Remove segment
                remaining = best_path[:i] + best_path[i+seg_len:]
                for j in range(len(remaining) - seg_len + 1):
                    if j == i or j == i + seg_len:
                        continue
                    # Insert segment at position j
                    new_path = remaining[:j] + segment + remaining[j:]
                    new_score = _total_overlap(new_path, overlap_matrix)
                    if new_score > best_score:
                        best_score = new_score
                        best_path = new_path
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_path

def _construct_n7_optimized():
    """Optimized construction for n=7 using local search on permutation ordering."""
    # Permutations of 1..6
    perms_6 = list(permutations(range(1, 7)))
    n_perms = len(perms_6)
    
    # Precompute overlap matrix
    overlap_matrix = _compute_overlap_matrix(perms_6)
    
    # Try multiple initializations: standard order and greedy NN
    order_6 = _get_standard_order(7)
    perm_to_idx = {p: i for i, p in enumerate(perms_6)}
    initial_path = [perm_to_idx[p] for p in order_6]
    
    greedy_path = _greedy_nn_init(perms_6, overlap_matrix)
    
    # Improve both with 2-opt and Or-opt
    best_path_1 = _two_opt_improve(initial_path, overlap_matrix, max_iterations=30)
    best_path_1 = _or_opt_improve(best_path_1, overlap_matrix, max_iterations=30)
    score_1 = _total_overlap(best_path_1, overlap_matrix)
    
    if _remaining_time() < 10.0:
        best_path = best_path_1
    else:
        best_path_2 = _two_opt_improve(greedy_path, overlap_matrix, max_iterations=30)
        best_path_2 = _or_opt_improve(best_path_2, overlap_matrix, max_iterations=30)
        score_2 = _total_overlap(best_path_2, overlap_matrix)
        
        if score_2 > score_1:
            best_path = best_path_2
        else:
            best_path = best_path_1
    
    # Construct the superpermutation
    word = _construct_from_order([perms_6[i] for i in best_path], 7)
    
    return word

def construct(n):
    if n <= 0:
        return ""
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1312131"
    if n == 4:
        return "1413121413124131241"
    if n == 5:
        return "".join(map(str, _standard_construct(5)))
    if n == 6:
        return "".join(map(str, _standard_construct(6)))
    
    if n == 7:
        # Try optimized construction, fall back to standard if time runs out or error
        try:
            word = _construct_n7_optimized()
            # Verify it's valid
            if _verify_superpermutation(word, 7):
                return "".join(map(str, word))
        except Exception:
            pass
        return "".join(map(str, _standard_construct(7)))
    
    # For n > 7, use standard construction
    return "".join(map(str, _standard_construct(n)))
