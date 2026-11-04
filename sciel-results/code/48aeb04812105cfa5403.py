import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute neighbors for each word (radius 1)
    # neighbors[i] is a list of indices of words within distance 1 of words[i]
    
    # Generate all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    neighbors = []
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors.append(neigh)
    
    # Convert neighbors to numpy arrays for faster iteration
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    # We want to find a set of codewords such that every word is covered.
    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    # Let's do Greedy first.
    while uncovered.any():
        best_idx = -1
        best_count = 0
        for j in range(N):
            # Count how many neighbors of j are uncovered using numpy
            count = np.sum(uncovered[neighbors_np[j]])
            if count > best_count:
                best_count = count
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        uncovered[neighbors_np[best_idx]] = False
        
        if time.time() > deadline:
            break
    
    # Initial solution size: len(selected)
    # Now, try to reduce the size using local search.
    # A codeword j is redundant if every word in its neighborhood is covered by some other selected codeword.
    
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        coverage_count[neighbors_np[j]] += 1
        
    # Try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        for idx in range(len(selected)):
            j = selected[idx]
            # Check if j is redundant: min coverage in its neighborhood > 1
            is_redundant = np.min(coverage_count[neighbors_np[j]]) > 1
            if is_redundant:
                # Remove j
                coverage_count[neighbors_np[j]] -= 1
                selected.pop(idx)
                changed = True
                break  # Restart loop since indices changed
                
        if time.time() > deadline:
            break
            
    # Further improvement: Local Search (Swap/Move)
    # Strategy: "add-then-remove"
    # 1. Pick a random codeword not in C, add it to C'.
    # 2. Run the removal heuristic on C'.
    # 3. If len(C') < len(C), accept.
    # 4. Repeat until time runs out or no improvement.
    
    best_selected = list(selected)
    best_size = len(best_selected)
    
    # Let's implement a function to try to reduce a given set
    def try_reduce(selected_set, coverage_counts, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(selected_set)):
                j = selected_set[idx]
                is_redundant = np.min(coverage_counts[neighbors_np[j]]) > 1
                if is_redundant:
                    for nb in neighbors_np[j]:
                        coverage_counts[nb] -= 1
                    selected_set.pop(idx)
                    changed = True
                    break
        return selected_set
    
    # Local Search Loop
    # We will try random additions.
    random.seed(42) # Deterministic for reproducibility in evaluation?
    
    # We need to be careful with time.
    # To speed up, we can precompute the list of all indices and maintain a set of selected for O(1) lookup.
    all_indices = list(range(N))
    selected_set = set(selected)
    
    while time.time() < deadline - 5.0:
        # Pick a random candidate to add
        if len(selected_set) < N:
            # Efficiently pick a random candidate not in selected_set
            # Since N is small (729), we can just sample until we find one not in set
            # Or better: create a list of available candidates if it's not too expensive?
            # Actually, just sampling is fine.
            add_idx = random.choice(all_indices)
            if add_idx in selected_set:
                continue
            
            # Add it
            new_selected = list(selected)
            new_selected.append(add_idx)
            new_selected_set = set(new_selected)
            
            # Create new coverage counts
            new_cov = coverage_count.copy()
            new_cov[neighbors_np[add_idx]] += 1
            
            # Try to reduce
            new_selected = try_reduce(new_selected, new_cov, deadline)
            
            if len(new_selected) < len(selected):
                selected = new_selected
                selected_set = set(selected)
                # Update coverage_count
                coverage_count = np.zeros(N, dtype=np.int32)
                for j in selected:
                    coverage_count[neighbors_np[j]] += 1
                # Update best
                best_selected = list(selected)
                best_size = len(selected)
            # else: discard
            
    # Final check: ensure the selected set is valid (should be, by construction)
    # Convert indices back to words
    codewords = [words[i] for i in selected]
    
    return codewords
