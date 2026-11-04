import time
import random
from itertools import permutations

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

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _construct_n7_optimized():
    """Optimized construction for n=7 using TSP on permutation ordering."""
    perms_6 = list(permutations(range(1, 7)))
    n_perms = len(perms_6)
    
    # Precompute overlap matrix
    overlap_matrix = [[0] * n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        p_i = perms_6[i]
        for j in range(n_perms):
            if i != j:
                overlap_matrix[i][j] = _overlap_len(list(p_i), list(perms_6[j]))
            else:
                overlap_matrix[i][j] = 0 # Self overlap not used in path
        
    # Get initial order from standard construction of n=6
    word_6 = _standard_construct(6)
    order_6 = []
    seen = set()
    for i in range(len(word_6) - 6 + 1):
        window = tuple(word_6[i:i+6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            order_6.append(window)
            
    perm_to_idx = {p: i for i, p in enumerate(perms_6)}
    best_path = [perm_to_idx[p] for p in order_6]
    
    # Calculate initial score (total overlap)
    def calc_score(path):
        score = 0
        for i in range(len(path) - 1):
            score += overlap_matrix[path[i]][path[i+1]]
        return score
    
    best_score = calc_score(best_path)
    
    # Local Search: Best Improvement Swap and Move
    # We will run this until time runs out or no improvement is found in a full pass
    
    improved = True
    while improved and _remaining_time() > 10.0:
        improved = False
        # Pass 1: Swaps
        for i in range(n_perms - 1):
            if _remaining_time() < 10.0:
                return _construct_from_order([perms_6[x] for x in best_path], 7)
            
            # Try swapping i and i+1
            # Old contribution: M[i-1][i] + M[i][i+1] + M[i+1][i+2]
            # New contribution: M[i-1][i+1] + M[i+1][i] + M[i][i+2]
            
            old_cost = 0
            if i > 0:
                old_cost += overlap_matrix[best_path[i-1]][best_path[i]]
            old_cost += overlap_matrix[best_path[i]][best_path[i+1]]
            if i + 2 < n_perms:
                old_cost += overlap_matrix[best_path[i+1]][best_path[i+2]]
                
            new_cost = 0
            if i > 0:
                new_cost += overlap_matrix[best_path[i-1]][best_path[i+1]]
            new_cost += overlap_matrix[best_path[i+1]][best_path[i]]
            if i + 2 < n_perms:
                new_cost += overlap_matrix[best_path[i]][best_path[i+2]]
                
            if new_cost > old_cost:
                # Apply swap
                best_path[i], best_path[i+1] = best_path[i+1], best_path[i]
                best_score += (new_cost - old_cost)
                improved = True
                # After a swap, the neighbors might be affected, so we should restart or continue carefully.
                # Best improvement usually restarts or continues. Let's continue but be aware.
                # Actually, if we swap i, i+1, the edge (i-1, i) and (i+1, i+2) are changed.
                # We should probably restart the pass to be safe, or at least check neighbors.
                # For simplicity and speed, let's just mark improved and continue.
                # A more robust approach is to restart the pass.
                # Given N=720, restarting is cheap.
                break
        
        if improved:
            continue
            
        # Pass 2: Moves (Or-opt 1)
        for i in range(n_perms):
            if _remaining_time() < 10.0:
                return _construct_from_order([perms_6[x] for x in best_path], 7)
            
            # Try moving i to j
            # This is O(N^2) per i, so O(N^3) total. 720^3 is too slow.
            # We can limit j to a neighborhood or use a heuristic.
            # Let's try moving i to i+1, i+2, etc. up to a small distance, or random positions.
            # Or just try moving i to every position j != i, but that's 720*720 = 500k per pass.
            # 500k * 100 passes = 50M. Might be okay in Python if optimized?
            # Let's try a few random moves if time permits, or a limited neighborhood.
            
            # Let's try moving i to a random position j
            # Actually, let's just do a few random restarts of the local search if stuck.
            # Or try a 2-opt reversal.
            
            # 2-Opt Reversal:
            # Reverse segment [i+1, j]
            # Old: M[i][i+1] + M[j][j+1]
            # New: M[i][j] + M[i+1][j+1]
            # Wait, reversing [i+1..j] changes edges (i, i+1) and (j, j+1).
            # The internal edges are reversed, but since M is not symmetric, the sum of internal edges changes.
            # Sum of internal edges in reversed path:
            # Old: M[i+1][i+2] + ... + M[j-1][j]
            # New: M[j][j-1] + ... + M[i+2][i+1]
            # So we need to calculate the difference in internal sums.
            # This is O(N) per 2-opt move. O(N^3) total. Too slow.
            
            # Let's stick to Swaps for now, and maybe add a simple random restart.
            
        if not improved:
            # If no swap improved, try a random perturbation and restart
            if _remaining_time() > 20.0:
                # Random swap
                i = random.randint(0, n_perms - 2)
                j = random.randint(0, n_perms - 2)
                if i != j:
                    best_path[i], best_path[j] = best_path[j], best_path[i]
                    # Recalculate score? Or just let the loop handle it.
                    # The loop will find improvements.
                    # We need to update best_score?
                    # It's easier to just recalculate score from scratch if we perturb.
                    best_score = calc_score(best_path)
                    improved = True # Force another pass
    
    return _construct_from_order([perms_6[x] for x in best_path], 7)

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
