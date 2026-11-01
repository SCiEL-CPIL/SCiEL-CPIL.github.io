# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code using LP relaxation + greedy rounding + local search.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave margin

    # Total number of words in the space
    N = q ** n
    if N > 100000:
        # Fallback for large spaces: use a simple constructive approach
        return _constructive_cover(q, n, radius)

    # Generate all words
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)

    # Precompute balls: for each word, which words are within distance radius
    # balls[i] is a boolean array of length num_words
    # For radius 1, word i covers word j if they differ in at most 1 position
    # This means Hamming distance <= 1

    # Efficient ball computation using numpy
    # For each position, compute which words differ in that position
    balls = np.zeros((num_words, num_words), dtype=bool)
    
    # Self-covers
    np.fill_diagonal(balls, True)
    
    if radius >= 1:
        # For each position p, words that differ only in position p
        for p in range(n):
            # Extract position p from all words
            pos_vals = all_words[:, p]
            # For each value v in 0..q-1, find words with that value at position p
            for v in range(q):
                mask = (pos_vals == v)
                # Words with value v at position p are covered by words with other values at position p
                # Actually, word i covers word j if they differ in at most 1 position
                # So for position p, word i covers word j if all_positions_match_except_p(i,j)
                
                # This is getting complex. Let's use a simpler approach for radius 1.
                # Word i covers word j if Hamming distance <= 1
                # That means: same word, or differs in exactly one position
    
    # Simpler approach: compute pairwise Hamming distances for radius 1
    # For small n, this is feasible
    # dist(i,j) = number of positions where all_words[i] != all_words[j]
    
    # For radius 1, ball(i) = {j : dist(i,j) <= 1}
    # We can compute this efficiently:
    # For each i, ball(i) includes i itself and all words that differ in exactly one position
    
    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    
    if radius >= 1:
        for p in range(n):
            # For each position p, group words by their value at position p
            # Words in the same group (same value at p) but differing elsewhere are NOT in each other's ball via position p
            # Words with different values at position p are in each other's ball if they match elsewhere
            
            # This is tricky. Let's just compute it directly for small n.
            # For each i, for each position p, generate the word that differs from i only at position p
            # There are (q-1) such words per position, so n*(q-1) neighbors per word
            
            for i in range(num_words):
                base = all_words[i].copy()
                for p in range(n):
                    old_val = base[p]
                    for new_val in range(q):
                        if new_val == old_val:
                            continue
                        base[p] = new_val
                        # Find index of this word
                        # We need a mapping from word to index
                        # For small q^n, we can use a dictionary or just check
                        # Let's precompute a mapping
                        pass
    
    # Let's precompute a mapping from word tuple to index
    word_to_idx = {}
    for idx, w in enumerate(all_words):
        word_to_idx[tuple(w)] = idx
    
    # Now compute balls properly
    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    
    if radius >= 1:
        for i in range(num_words):
            base = all_words[i]
            for p in range(n):
                old_val = base[p]
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    base[p] = new_val
                    key = tuple(base)
                    if key in word_to_idx:
                        j = word_to_idx[key]
                        balls[i, j] = True
                    base[p] = old_val
    
    # Now solve the LP relaxation
    # Minimize sum(x_i) subject to sum_{i in ball(j)} x_i >= 1 for all j
    # x_i in [0, 1]
    
    # Use scipy's linprog
    try:
        from scipy.optimize import linprog
        
        # Objective: minimize sum(x_i)
        c = np.ones(num_words)
        
        # Constraints: for each j, sum_{i: balls[i,j]} x_i >= 1
        # linprog uses A_ub @ x <= b_ub for upper bounds
        # So we negate: -sum_{i: balls[i,j]} x_i <= -1
        A_ub = -balls.T  # shape (num_words, num_words), A_ub[j, i] = -balls[i, j]
        b_ub = -np.ones(num_words)
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1)] * num_words
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.status != 0:
            raise Exception(f"LP failed: {result.message}")
        
        x = result.x
    except Exception as e:
        # Fallback to pure greedy
        return _greedy_cover_numpy(all_words, balls, num_words, deadline)
    
    # Greedy rounding: sort by x_i descending, add if it covers new words
    order = np.argsort(-x)
    selected = []
    covered = np.zeros(num_words, dtype=bool)
    
    for idx in order:
        if time.time() > deadline:
            break
        # Check if this word covers any uncovered words
        new_cover = balls[idx, ~covered]
        if new_cover.any():
            selected.append(idx)
            covered |= balls[idx, :]
    
    # Convert to list of codewords
    codewords = [all_words[i].tolist() for i in selected]
    
    # Local search: try to improve by swapping
    if time.time() < deadline - 5:
        codewords = _local_search(codewords, all_words, balls, num_words, deadline)
    
    return codewords


def _greedy_cover_numpy(all_words, balls, num_words, deadline):
    """Greedy set cover using numpy."""
    selected = []
    covered = np.zeros(num_words, dtype=bool)
    
    while not covered.all() and time.time() < deadline:
        # Find word that covers the most uncovered words
        uncovered_count = balls[:, ~covered].sum(axis=1)
        best_idx = np.argmax(uncovered_count)
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        covered |= balls[best_idx, :]
    
    return [all_words[i].tolist() for i in selected]


def _local_search(codewords, all_words, balls, num_words, deadline):
    """Try to improve the solution by removing and adding codewords."""
    current_size = len(codewords)
    current_covered = np.zeros(num_words, dtype=bool)
    for c in codewords:
        idx = all_words[list(c).index(c[0])] if False else -1
        # This is wrong, let me fix
    # Actually, let's redo this properly
    
    # Map codewords to indices
    word_to_idx = {}
    for idx, w in enumerate(all_words):
        word_to_idx[tuple(w)] = idx
    
    selected_indices = []
    for c in codewords:
        selected_indices.append(word_to_idx[tuple(c)])
    
    covered = np.zeros(num_words, dtype=bool)
    for idx in selected_indices:
        covered |= balls[idx, :]
    
    best_solution = selected_indices[:]
    best_size = len(best_solution)
    
    # Try to remove codewords that don't change coverage
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_selected = best_solution[:]
        for i in range(len(new_selected)):
            if time.time() > deadline:
                break
            # Try removing new_selected[i]
            test_covered = np.zeros(num_words, dtype=bool)
            for j, idx in enumerate(new_selected):
                if j != i:
                    test_covered |= balls[idx, :]
            if test_covered.all():
                new_selected.pop(i)
                improved = True
                best_solution = new_selected
                best_size = len(best_solution)
                break
    
    # Try adding/removing pairs
    if time.time() < deadline - 5:
        for _ in range(100):  # Limit iterations
            if time.time() > deadline:
                break
            # Try swapping one codeword with another
            if len(best_solution) > 1:
                for i in range(len(best_solution)):
                    if time.time() > deadline:
                        break
                    # Try replacing best_solution[i] with an unselected word
                    current_covered = np.zeros(num_words, dtype=bool)
                    for j, idx in enumerate(best_solution):
                        if j != i:
                            current_covered |= balls[idx, :]
                    
                    # Find words that cover the uncovered parts
                    uncovered_mask = ~current_covered
                    if not uncovered_mask.any():
                        continue
                    
                    # Try adding a word that covers uncovered words
                    # and removing one that becomes redundant
                    # This is complex, let's just try simple removals again
    
    return [all_words[i].tolist() for i in best_solution]


def _constructive_cover(q, n, radius):
    """Fallback constructive approach."""
    # Use the tetracode construction for ternary case
    if q == 3 and n == 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    # General fallback: greedy
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)
    
    # Compute balls
    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    for i in range(num_words):
        for j in range(num_words):
            if np.sum(all_words[i] != all_words[j]) <= radius:
                balls[i, j] = True
    
    return _greedy_cover_numpy(all_words, balls, num_words, time.time() + 110)
# EVOLVE-BLOCK-END
