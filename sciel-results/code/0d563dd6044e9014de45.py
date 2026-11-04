"""Superpermutation construction with optimized ordering for n=7."""
import time
import random
from itertools import permutations

# Deadline guard: must finish well before 150s
_START_TIME = time.time()
DEADLINE = 140.0  # safety margin

def _remaining_time():
    return DEADLINE - (time.time() - _START_TIME)

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
    # Each block is P n P, where P is an (n-1)-permutation
    # Overlap between consecutive blocks: longest suffix of P_i matching prefix of P_{i+1}
    # (since suffix of P_i*n*P_i is suffix of P_i, and prefix of P_{i+1}*n*P_{i+1} is prefix of P_{i+1})
    
    if not order:
        return list(range(1, n + 1))
    
    # First block: order[0] + [n] + order[0]
    word = list(order[0]) + [n] + list(order[0])
    
    for i in range(1, len(order)):
        prev = order[i - 1]
        curr = order[i]
        block = list(curr) + [n] + list(curr)
        # Overlap: longest suffix of prev (which is the tail of the previous block)
        # that matches prefix of curr (which is the head of the current block)
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

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _greedy_tsp(perms, overlap_matrix, start_idx=None):
    """Greedy TSP: always pick unvisited neighbor with max overlap."""
    n_perms = len(perms)
    visited = [False] * n_perms
    path = []
    
    if start_idx is None:
        start_idx = 0
    visited[start_idx] = True
    path.append(start_idx)
    current = start_idx
    
    for _ in range(n_perms - 1):
        best_next = -1
        best_ov = -1
        for j in range(n_perms):
            if not visited[j]:
                ov = overlap_matrix[current][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        if best_next == -1:
            break
        visited[best_next] = True
        path.append(best_next)
        current = best_next
    
    return path

def _two_opt_improve(path, perms, overlap_matrix, max_iterations=50):
    """2-opt local search to maximize total overlap."""
    n_perms = len(path)
    
    def total_overlap(p):
        total = 0
        for i in range(len(p) - 1):
            total += overlap_matrix[p[i]][p[i+1]]
        return total
    
    best_path = path[:]
    best_score = total_overlap(best_path)
    
    for iteration in range(max_iterations):
        if _remaining_time() < 5.0:
            break
        improved = False
        for i in range(n_perms - 1):
            for j in range(i + 1, n_perms):
                # Try reversing segment [i+1..j]
                new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                new_score = total_overlap(new_path)
                if new_score > best_score:
                    best_score = new_score
                    best_path = new_path
                    improved = True
        if not improved:
            break
    
    return best_path

def _construct_n7_optimized():
    """Optimized construction for n=7 using TSP on permutation ordering."""
    # Permutations of 1..6
    perms_6 = list(permutations(range(1, 7)))
    n_perms = len(perms_6)
    
    # Precompute overlap matrix
    overlap_matrix = [[0] * n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        for j in range(n_perms):
            if i != j:
                overlap_matrix[i][j] = _overlap_len(list(perms_6[i]), list(perms_6[j]))
    
    # Start with the order from the standard construction
    word_6 = _standard_construct(6)
    order_6 = []
    seen = set()
    for i in range(len(word_6) - 6 + 1):
        window = tuple(word_6[i:i+6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            order_6.append(window)
    
    # Convert to indices
    perm_to_idx = {p: i for i, p in enumerate(perms_6)}
    initial_path = [perm_to_idx[p] for p in order_6]
    
    # Run greedy TSP as starting point
    greedy_path = _greedy_tsp(perms_6, overlap_matrix)
    
    # Compare scores
    def path_score(p):
        total = 0
        for i in range(len(p) - 1):
            total += overlap_matrix[p[i]][p[i+1]]
        return total
    
    # Use the better of initial and greedy as starting point
    if path_score(greedy_path) > path_score(initial_path):
        best_path = greedy_path
    else:
        best_path = initial_path
    
    # Improve with 2-opt
    best_path = _two_opt_improve(best_path, perms_6, overlap_matrix, max_iterations=100)
    
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
        return _standard_construct(5)
    if n == 6:
        return _standard_construct(6)
    
    if n == 7:
        # Try optimized construction, fall back to standard if time runs out
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
