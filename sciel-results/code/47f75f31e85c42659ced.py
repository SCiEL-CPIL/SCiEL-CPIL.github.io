import time
import random
from itertools import permutations

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit, leave 5s margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    def factorial(x):
        result = 1
        for i in range(2, x + 1):
            result *= i
        return result
    
    # Build list of all permutations of 1..n
    all_perms = list(permutations(range(1, n + 1)))
    num_all = len(all_perms)
    
    # Precompute overlap matrix for efficiency
    # overlap[i][j] = largest k such that all_perms[i][-k:] == all_perms[j][:k]
    # For n=7, this is 5040x5040. Let's use a more memory-efficient approach.
    
    # Actually, let's precompute as a list of lists of bytes
    overlap_matrix = []
    for i in range(num_all):
        row = [0] * num_all
        p = all_perms[i]
        for j in range(num_all):
            q = all_perms[j]
            for k in range(n - 1, -1, -1):
                if k == 0:
                    row[j] = 0
                    break
                if tuple(p[-k:]) == tuple(q[:k]):
                    row[j] = k
                    break
        overlap_matrix.append(row)
    
    def get_overlap_idx(i, j):
        return overlap_matrix[i][j]
    
    def path_length(path):
        if not path:
            return 0
        total = n
        for i in range(len(path) - 1):
            total += n - get_overlap_idx(path[i], path[i + 1])
        return total
    
    def nearest_neighbor():
        remaining = set(range(num_all))
        path = []
        start = random.randint(0, num_all - 1)
        path.append(start)
        remaining.remove(start)
        
        while remaining:
            current = path[-1]
            best_idx = -1
            best_cost = float('inf')
            for idx in remaining:
                cost = n - get_overlap_idx(current, idx)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            path.append(best_idx)
            remaining.remove(best_idx)
        
        return path
    
    def two_opt_simple(path, max_iterations=50):
        """Simple 2-opt: try reversing segments"""
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            for i in range(len(path) - 1):
                if time.time() > deadline:
                    return path
                for j in range(i + 2, len(path)):
                    if time.time() > deadline:
                        return path
                    # Cost before: edge (i, i+1) and edge (j, j+1) if exists
                    cost_before = n - get_overlap_idx(path[i], path[i + 1])
                    if j + 1 < len(path):
                        cost_before += n - get_overlap_idx(path[j], path[j + 1])
                    
                    # After reversal: edge (i, j) and edge (i+1, j+1) if exists
                    cost_after = n - get_overlap_idx(path[i], path[j])
                    if j + 1 < len(path):
                        cost_after += n - get_overlap_idx(path[i + 1], path[j + 1])
                    
                    if cost_after < cost_before:
                        path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                        improved = True
        
        return path
    
    # Run nearest neighbor with multiple restarts
    best_path = None
    best_len = float('inf')
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        random.seed(r * 12345 + 67890)
        path = nearest_neighbor()
        path = two_opt_simple(path, max_iterations=20)
        plen = path_length(path)
        if plen < best_len:
            best_len = plen
            best_path = path[:]
    
    # Build the word from the best path
    if best_path is None:
        # Fallback to standard construction
        word = [1]
        for m in range(2, n + 1):
            perms_in_word = []
            seen = set()
            for i in range(len(word) - (m - 1) + 1):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1:
                    if window not in seen:
                        seen.add(window)
                        perms_in_word.append(window)
            new_word = []
            for p in perms_in_word:
                block = list(p) + [m] + list(p)
                max_overlap = 0
                min_len = min(len(new_word), len(block))
                for k in range(min_len, -1, -1):
                    if k == 0:
                        max_overlap = 0
                        break
                    if new_word[-k:] == block[:k]:
                        max_overlap = k
                        break
                new_word += block[max_overlap:]
            word = new_word
        return "".join(map(str, word))
    
    word = list(all_perms[best_path[0]])
    for i in range(len(best_path) - 1):
        p = all_perms[best_path[i]]
        q = all_perms[best_path[i + 1]]
        overlap = get_overlap_idx(best_path[i], best_path[i + 1])
        word.extend(q[overlap:])
    
    return "".join(map(str, word))
