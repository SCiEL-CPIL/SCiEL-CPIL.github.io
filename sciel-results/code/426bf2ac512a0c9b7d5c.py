"""Rewritten program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

import numpy as np


def _build_neighbors(q, n):
    """For each word (index 0..q^n-1), return list of indices within distance 1."""
    total = q ** n
    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    # Map word -> index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i in range(total):
        w = all_words[i]
        nb = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nb.append(word_to_idx[tuple(nw)])
        neighbors.append(nb)
    return neighbors, all_words


def _is_covered(codeword_indices, neighbors, all_words):
    """Check if all words are covered by the given codewords."""
    total = len(all_words)
    covered = np.zeros(total, dtype=bool)
    for ci in codeword_indices:
        covered[neighbors[ci]] = True
    return covered.all()


def _greedy_set_cover(neighbors, total, max_iter=500):
    """Greedy set cover: pick codeword covering most uncovered words."""
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    selected = []
    
    # Precompute neighbor arrays
    nb_arrays = [np.array(nb) for nb in neighbors]
    
    for _ in range(max_iter):
        if uncovered_count == 0:
            break
        best_idx = -1
        best_count = -1
        for i in range(total):
            if not uncovered[i]:
                continue
            cnt = uncovered[nb_arrays[i]].sum()
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        uncovered[nb_arrays[best_idx]] = False
        uncovered_count -= best_count
    
    return selected


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    if radius != 1:
        # Fallback for other radii: brute force or simple construction
        if radius == 0:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
        # For radius >= 2, use direct sum approach generalized
        # For now, return a valid (possibly suboptimal) covering
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    total = q ** n  # 2187 for q=3, n=7
    
    # Build neighborhood structure
    neighbors, all_words = _build_neighbors(q, n)
    
    # Initial solution: direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    # Convert to indices
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    initial_indices = [word_to_idx[tuple(c)] for c in initial_codewords]
    initial_indices = list(set(initial_indices))
    
    best_solution = list(initial_indices)
    best_size = len(best_solution)
    
    # Verify initial solution
    assert _is_covered(best_solution, neighbors, all_words), "Initial solution is not a valid cover"
    
    # Local improvement: try removing codewords
    # For each codeword, check if removing it still maintains coverage
    covered = np.zeros(total, dtype=bool)
    for ci in best_solution:
        covered[neighbors[ci]] = True
    
    # Count how many codewords cover each word
    cover_count = np.zeros(total, dtype=np.int32)
    for ci in best_solution:
        cover_count[neighbors[ci]] += 1
    
    # Try removing codewords one by one
    removed_any = True
    while removed_any:
        removed_any = False
        current = list(best_solution)
        for idx in range(len(current)):
            if time.time() > deadline:
                break
            ci = current[idx]
            # Check if any word would become uncovered
            would_uncover = (cover_count[neighbors[ci]] == 1).any()
            if not would_uncover:
                # Remove it
                cover_count[neighbors[ci]] -= 1
                current.pop(idx)
                best_solution = current
                best_size = len(best_solution)
                removed_any = True
                break
        if time.time() > deadline:
            break
    
    # Greedy local search: try swapping codewords
    # Try to find a codeword not in solution that, when added and allowing removal of multiple, improves
    rng = random.Random(42)
    
    # Precompute which codewords are in current solution
    in_solution = np.zeros(total, dtype=bool)
    for ci in best_solution:
        in_solution[ci] = True
    
    # Try adding random non-solution codewords and then greedily removing redundant ones
    num_attempts = 0
    while time.time() < deadline - 5:
        num_attempts += 1
        # Pick a random codeword not in solution
        candidates = np.where(~in_solution)[0]
        if len(candidates) == 0:
            break
        new_ci = candidates[rng.randint(0, len(candidates) - 1)]
        
        # Try adding new_ci and see if we can remove more than 1
        trial = list(best_solution) + [new_ci]
        
        # Recompute cover counts
        cc = np.zeros(total, dtype=np.int32)
        for ci in trial:
            cc[neighbors[ci]] += 1
        
        # Try removing redundant codewords
        removed = 0
        trial_set = set(trial)
        changed = True
        while changed:
            changed = False
            for ci in list(trial_set):
                if time.time() > deadline:
                    break
                if (cc[neighbors[ci]] == 1).any():
                    continue
                # Can remove
                cc[neighbors[ci]] -= 1
                trial_set.discard(ci)
                removed += 1
                changed = True
            if time.time() > deadline:
                break
        
        if len(trial_set) < best_size:
            best_solution = list(trial_set)
            best_size = len(best_solution)
            in_solution = np.zeros(total, dtype=bool)
            for ci in best_solution:
                in_solution[ci] = True
    
    # Final verification
    if not _is_covered(best_solution, neighbors, all_words):
        # Fallback to initial solution
        best_solution = initial_indices
    
    # Convert back to list of lists
    result = [list(all_words[ci]) for ci in best_solution]
    
    return result
# EVOLVE-BLOCK-END
