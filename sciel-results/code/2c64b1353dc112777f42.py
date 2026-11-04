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
    # Also precompute the neighbor masks as bitsets for faster operations if needed, 
    # but for N=729, lists are fine.
    # We will use a list of lists for neighbors.
    
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
    
    # We want to find a set of codewords such that every word is covered.
    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    # To speed up greedy, we can maintain counts of how many uncovered words each codeword covers.
    # However, updating counts is O(N) per step. With ~70 steps, total O(70*N) ~ 50k ops. 
    # But calculating the max each time is O(N * avg_neigh) ~ 729 * 19 ~ 14k ops.
    # Total ~ 1M ops. Very fast.
    
    # Let's implement a slightly smarter greedy:
    # 1. Start with a random permutation or just iterate.
    # 2. Pick the one with max coverage.
    
    # We can also try multiple random restarts of the greedy if time permits, 
    # but greedy is usually deterministic and quite good. 
    # The main improvement over simple greedy is often a local search (swap/remove).
    
    # Let's do Greedy first.
    while uncovered.any():
        best_idx = -1
        best_count = 0
        # Optimization: only consider codewords that cover at least one uncovered word?
        # Actually, we can just iterate all.
        for j in range(N):
            # Count how many neighbors of j are uncovered
            count = 0
            for nb in neighbors[j]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered[nb] = False
            
        if time.time() > deadline:
            break
    
    # Initial solution size: len(selected)
    # Now, try to reduce the size using local search.
    # We can try to remove codewords one by one if they are redundant.
    # A codeword j is redundant if every word in its neighborhood is covered by some other selected codeword.
    
    # To check redundancy efficiently:
    # Maintain a count array: coverage_count[i] = number of selected codewords that cover word i.
    # A codeword j is redundant if for all i in neighbors[j], coverage_count[i] > 1.
    
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        for nb in neighbors[j]:
            coverage_count[nb] += 1
            
    # Try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        for idx in range(len(selected)):
            j = selected[idx]
            # Check if j is redundant
            is_redundant = True
            for nb in neighbors[j]:
                if coverage_count[nb] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove j
                for nb in neighbors[j]:
                    coverage_count[nb] -= 1
                selected.pop(idx)
                changed = True
                break  # Restart loop since indices changed
                
        if time.time() > deadline:
            break
            
    # Further improvement: Local Search (Swap/Move)
    # We can try to replace a codeword with another that covers more "uniquely" covered spots, 
    # or simply try to add a codeword and then remove two? No, that increases size.
    # Standard local search for set cover:
    # 1. Try to remove a codeword and see if it's still valid (already done).
    # 2. Try to swap a codeword i with a codeword j (not in set) such that the new set is still valid 
    #    and potentially allows future removals? 
    #    Actually, a better heuristic is: 
    #    "Try to add a random codeword, then run the removal heuristic. If size decreased, keep it."
    #    Or "Try to move a codeword".
    
    # Let's try a simple local search: 
    # Iterate through all possible pairs (remove i, add j). If valid, accept.
    # But checking validity is O(N * deg) per swap. 
    # Number of swaps ~ |C| * (N - |C|) ~ 70 * 660 ~ 46k. 
    # Each check O(729 * 19) ~ 14k. Total ~ 600M ops. Too slow in Python.
    
    # Better local search:
    # Focus on "tight" codewords. 
    # Or use the "add-then-remove" strategy.
    # 1. Pick a random codeword not in C, add it to C'.
    # 2. Run the removal heuristic on C'.
    # 3. If len(C') < len(C), accept.
    # 4. Repeat until time runs out or no improvement.
    
    best_selected = list(selected)
    best_size = len(best_selected)
    
    # To make the removal heuristic faster, we can use a priority queue or just iterate.
    # Given the small size, iterating is fine.
    
    # Let's implement a function to try to reduce a given set
    def try_reduce(selected_set, coverage_counts, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(selected_set)):
                j = selected_set[idx]
                is_redundant = True
                for nb in neighbors[j]:
                    if coverage_counts[nb] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for nb in neighbors[j]:
                        coverage_counts[nb] -= 1
                    selected_set.pop(idx)
                    changed = True
                    break
        return selected_set
    
    # Local Search Loop
    # We will try random additions.
    random.seed(42) # Deterministic for reproducibility in evaluation?
    
    # To avoid getting stuck, we can also try different initial greedy orderings?
    # But greedy is quite stable.
    
    # Let's try the add-remove strategy.
    # We need to be careful with time.
    
    while time.time() < deadline - 5.0:
        # Pick a random candidate to add
        if len(selected) < N:
            candidates = [i for i in range(N) if i not in selected]
            if not candidates:
                break
            add_idx = random.choice(candidates)
            
            # Add it
            new_selected = list(selected)
            new_selected.append(add_idx)
            
            # Create new coverage counts
            new_cov = np.zeros(N, dtype=np.int32)
            for j in new_selected:
                for nb in neighbors[j]:
                    new_cov[nb] += 1
            
            # Try to reduce
            new_selected = try_reduce(new_selected, new_cov, deadline)
            
            if len(new_selected) < len(selected):
                selected = new_selected
                # Update coverage_count for the main loop? 
                # Actually, we just replaced 'selected'. 
                # We need to recalculate coverage_count for 'selected' if we do more swaps?
                # Or we can just keep updating 'selected' and 'coverage_count' in sync.
                # Let's recalculate coverage_count for the new best to be safe for the next iteration's context?
                # Actually, the next iteration picks a random add from the current 'selected'.
                # So we just need 'selected' to be the current best.
                # The 'coverage_count' variable in the outer scope is not used by try_reduce (it takes its own).
                # But we need to ensure 'selected' is valid.
                pass
            else:
                # No improvement, discard
                pass
        
        if time.time() > deadline:
            break
            
    # Final check: ensure the selected set is valid (should be, by construction)
    # Convert indices back to words
    codewords = [words[i] for i in selected]
    
    return codewords
