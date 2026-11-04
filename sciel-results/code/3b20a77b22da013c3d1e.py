"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def construct(q, n, radius):
    """
    Greedy set-covering heuristic initialized with a random seed and optimized
    via simulated annealing. Models the problem as selecting a minimum subset
    of words to maximize Hamming ball coverage.
    
    For the specific task: q=3, n=6, radius=1.
    """
    # Handle general case, but optimize for the specific task parameters
    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1()
    
    # General fallback: use a simple greedy approach for other parameters
    return _general_greedy(q, n, radius)


def _solve_ternary_6_1():
    """
    Optimized solver for ternary code of length 6, covering radius 1.
    Uses greedy initialization + simulated annealing.
    """
    np.random.seed(42)
    random.seed(42)
    
    q = 3
    n = 6
    radius = 1
    
    # Precompute all possible words
    all_words = np.array(list(_itertools_product(range(q), repeat=n)), dtype=np.uint8)
    total_words = q ** n  # 729
    
    # Precompute Hamming balls for each word
    # For radius 1 in ternary: each word covers 1 + 6*2 = 13 words
    # Ball[word_idx] = set of word indices covered
    
    # Build distance matrix or use efficient coverage counting
    # For efficiency, we'll compute coverage incrementally
    
    # Precompute: for each word, which words are within distance 1
    # Use vectorized operations
    # all_words shape: (729, 6)
    
    # Compute coverage: for each candidate center, which words it covers
    # A word w is covered by center c if hamming_dist(w, c) <= 1
    # This means they differ in at most 1 position
    
    # Precompute the list of neighbors for each word
    # For ternary radius 1: neighbor is same word, or differ in one position with 2 other values
    # Total neighbors per word: 1 + 6*2 = 13
    
    # Build neighbor lists
    neighbors = []
    for i in range(total_words):
        word = all_words[i]
        covered = {i}
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                new_word = word.copy()
                new_word[pos] = new_val
                # Find index of new_word
                # Convert to tuple for lookup
                idx = _word_to_index(new_word, q, n)
                covered.add(idx)
        neighbors.append(covered)
    
    # Convert to numpy arrays for faster operations
    # Actually, let's use a different approach: precompute a coverage matrix
    # coverage_matrix[i, j] = 1 if word j is within radius of word i
    # Shape: (729, 729) - this is manageable
    
    coverage_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        for j in neighbors[i]:
            coverage_matrix[i, j] = 1
    
    # Greedy initialization
    # Start with empty set, repeatedly add the word that covers the most uncovered words
    
    uncovered = np.ones(total_words, dtype=bool)
    selected = []
    remaining = set(range(total_words))
    
    # For speed, track how many uncovered words each candidate covers
    # uncovered_count[i] = number of uncovered words in neighbors[i]
    uncovered_count = np.array([len(neighbors[i]) for i in range(total_words)], dtype=np.int32)
    
    while uncovered.any() and remaining:
        # Find the candidate that covers the most uncovered words
        # Among remaining candidates
        best_idx = -1
        best_count = -1
        for i in remaining:
            if uncovered_count[i] > best_count:
                best_count = uncovered_count[i]
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        remaining.discard(best_idx)
        
        # Update uncovered
        uncovered[neighbors[best_idx]] = False
        
        # Update uncovered counts for all remaining candidates
        # For each remaining candidate, subtract the number of newly covered words it had
        for i in remaining:
            # Count how many of neighbors[i] were just covered
            newly_covered = len(neighbors[i] & neighbors[best_idx])
            uncovered_count[i] -= newly_covered
    
    # If greedy didn't cover everything (shouldn't happen for this small case), add more
    while uncovered.any():
        # Find an uncovered word and add a center that covers it
        uncov_indices = np.where(uncovered)[0]
        target = uncov_indices[0]
        # Find a center that covers target
        for i in range(total_words):
            if coverage_matrix[i, target] == 1:
                selected.append(i)
                uncovered[neighbors[i]] = False
                break
    
    # Simulated annealing to improve
    # Objective: minimize |selected| while maintaining full coverage
    
    selected_set = set(selected)
    
    # Check if we can remove any codeword and still have full coverage
    # For each selected codeword, check if all words it covers are also covered by others
    
    def is_valid(selected_set):
        """Check if selected_set covers all words."""
        covered = np.zeros(total_words, dtype=bool)
        for i in selected_set:
            covered[neighbors[i]] = True
        return covered.all()
    
    # Try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        for i in list(selected_set):
            # Check if removing i still maintains coverage
            new_set = selected_set - {i}
            if is_valid(new_set):
                selected_set = new_set
                changed = True
    
    # Simulated annealing: try to swap one codeword for another that allows removal
    # Or try to add a codeword that allows removal of multiple others
    
    best_set = set(selected_set)
    best_size = len(best_set)
    
    # SA parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.995
    T = T_start
    max_iter = 5000
    
    for it in range(max_iter):
        if T < T_end:
            break
        
        current_set = set(best_set)
        
        # Move: randomly remove one codeword, then greedily add back if needed
        if len(current_set) > 0:
            to_remove = random.choice(list(current_set))
            trial_set = current_set - {to_remove}
            
            # Check coverage
            covered = np.zeros(total_words, dtype=bool)
            for i in trial_set:
                covered[neighbors[i]] = True
            
            if covered.all():
                # Valid improvement
                best_set = trial_set
                best_size = len(best_set)
            else:
                # Need to add back some codewords to restore coverage
                uncovered = np.where(~covered)[0]
                if len(uncovered) > 0:
                    # Add the best single codeword to cover most uncovered
                    best_add = -1
                    best_cover = -1
                    for i in range(total_words):
                        if i in trial_set:
                            continue
                        cov_count = sum(1 for j in uncovered if coverage_matrix[i, j] == 1)
                        if cov_count > best_cover:
                            best_cover = cov_count
                            best_add = i
                    if best_add >= 0:
                        trial_set.add(best_add)
                        if len(trial_set) < best_size:
                            # Net improvement
                            best_set = trial_set
                            best_size = len(best_set)
        
        T *= alpha
    
    return [list(all_words[i]) for i in best_set]


def _word_to_index(word, q, n):
    """Convert a word (array) to its index in the lexicographic enumeration."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _itertools_product(iterable, repeat=1):
    """Simple itertools.product replacement."""
    import itertools
    return list(itertools.product(iterable, repeat=repeat))


def _general_greedy(q, n, radius):
    """
    General greedy set-covering heuristic for arbitrary q, n, radius.
    """
    import itertools
    
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = q ** n
    
    # Precompute neighbors for each word
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
    
    # For efficiency with small spaces, precompute coverage
    # This is only practical for small total_words
    if total_words > 10000:
        # Use a simpler approach for large spaces
        # Just use the direct sum method as fallback
        return _direct_sum_fallback(q, n, radius)
    
    neighbors = []
    for i, word in enumerate(all_words):
        covered = set()
        for j, other in enumerate(all_words):
            if hamming_dist(word, other) <= radius:
                covered.add(j)
        neighbors.append(covered)
    
    # Greedy set cover
    uncovered = set(range(total_words))
    selected = []
    remaining = set(range(total_words))
    
    while uncovered:
        best_idx = -1
        best_count = -1
        for i in remaining:
            count = len(neighbors[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
        remaining.discard(best_idx)
        uncovered -= neighbors[best_idx]
    
    # Remove redundant codewords
    selected_set = set(selected)
    changed = True
    while changed:
        changed = False
        for i in list(selected_set):
            new_set = selected_set - {i}
            covered = set()
            for j in new_set:
                covered |= neighbors[j]
            if covered == set(range(total_words)):
                selected_set = new_set
                changed = True
    
    return [list(all_words[i]) for i in selected_set]


def _direct_sum_fallback(q, n, radius):
    """Fallback using direct sum construction."""
    import itertools
    
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # General fallback: use all-zero word repeated (not optimal but valid if radius is large enough)
    # For now, return a simple construction
    return [[0] * n]
# EVOLVE-BLOCK-END
