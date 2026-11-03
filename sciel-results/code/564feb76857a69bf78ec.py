import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Precomputation: Map index to tuple and vice versa
    # We use base-q representation where index = sum(w[i] * q^(n-1-i))
    # This matches standard integer conversion.
    
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx

    # Precompute all words as tuples
    all_words = [idx_to_tuple(i) for i in range(total_words)]
    
    # Precompute cover matrix using NumPy
    # cover_matrix[i, j] = 1 if word j is within distance <= radius of word i
    # For radius 1, this is a sparse structure, but dense NumPy matrix is fine for N=2187
    # 2187 * 2187 * 1 byte (uint8) ~ 4.7 MB, very small.
    
    cover_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
    
    # Diagonal: distance 0
    np.fill_diagonal(cover_matrix, 1)
    
    if radius >= 1:
        # Precompute powers of q for index calculation
        powers = [q ** (n - 1 - pos) for pos in range(n)]
        
        for i in range(total_words):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    # Calculate j
                    # j = i + (new_val - orig_val) * power
                    j = i + (new_val - orig_val) * powers[pos]
                    cover_matrix[i, j] = 1
                    
    # Greedy Set Cover Algorithm with NumPy acceleration
    def greedy_solve(random_seed):
        rng = random.Random(random_seed)
        uncovered = np.ones(total_words, dtype=np.uint8)
        selected = []
        
        # We need to efficiently calculate gains: gain[i] = sum(cover_matrix[i] * uncovered)
        # This is a matrix-vector product.
        
        while np.any(uncovered):
            # Calculate gains for all candidates
            # cover_matrix is (N, N) uint8, uncovered is (N,) uint8
            # Result is (N,) uint16 or int32
            gains = cover_matrix.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking
            best_idx = candidates[rng.randint(0, len(candidates)-1)]
            
            selected.append(int(best_idx))
            # Update uncovered
            uncovered[cover_matrix[best_idx] == 1] = 0
            
        return selected

    def local_search_reduce(code):
        if not code:
            return code
            
        current_code = code[:]
        
        # Helper to check if a set of indices is a valid dominating set
        # We maintain a coverage count array for efficiency during swaps?
        # For N=2187, recalculating coverage is O(|C| * N). |C| ~ 180, N=2187.
        # 180 * 2187 ~ 400k ops. Very fast.
        
        def is_valid_dominating_set(indices):
            if not indices:
                return False
            coverage = np.zeros(total_words, dtype=np.uint8)
            for idx in indices:
                coverage[cover_matrix[idx] == 1] += 1
            # Check if any word has 0 coverage
            return np.all(coverage > 0)

        # Phase 1: Deletion
        # Try to remove each codeword if possible
        changed = True
        while changed:
            changed = False
            # Shuffle to randomize order of removal attempts
            random.shuffle(current_code)
            for i in range(len(current_code)):
                if time.time() > time_limit:
                    return current_code
                idx_to_remove = current_code[i]
                # Check if removing it leaves a valid set
                test_set = current_code[:i] + current_code[i+1:]
                if is_valid_dominating_set(test_set):
                    current_code = test_set
                    changed = True
                    break # Restart the loop as indices changed
                    
        # Phase 2: Swap
        # Try to replace a codeword with another to potentially enable further deletions
        # Or simply to find a different local minimum.
        # We do a limited number of swap iterations.
        swap_iterations = 50
        for _ in range(swap_iterations):
            if time.time() > time_limit:
                return current_code
            
            improved = False
            # Try swapping each codeword with a random candidate
            for i in range(len(current_code)):
                if time.time() > time_limit:
                    return current_code
                idx_to_replace = current_code[i]
                # Pick a random candidate from the whole space
                candidate_idx = random.randint(0, total_words - 1)
                if candidate_idx == idx_to_replace:
                    continue
                if candidate_idx in current_code:
                    continue
                    
                test_set = current_code[:i] + [candidate_idx] + current_code[i+1:]
                if is_valid_dominating_set(test_set):
                    current_code = test_set
                    improved = True
                    # After a swap, we might be able to delete something
                    # Run a quick deletion pass
                    del_changed = True
                    while del_changed:
                        del_changed = False
                        random.shuffle(current_code)
                        for j in range(len(current_code)):
                            if time.time() > time_limit:
                                return current_code
                            idx_to_remove = current_code[j]
                            test_set2 = current_code[:j] + current_code[j+1:]
                            if is_valid_dominating_set(test_set2):
                                current_code = test_set2
                                del_changed = True
                                break
                    break # Restart swap loop if we made a change
                    
            if not improved:
                break
                
        return current_code

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy passes with different seeds
    seed = 0
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit - 5.0:
            # Not enough time for a full local search, just try greedy
            solution = greedy_solve(seed)
            if len(solution) < best_len:
                best_len = len(solution)
                best_solution = solution
            seed += 1
            continue
            
        solution_indices = greedy_solve(seed)
        # Local search
        reduced_indices = local_search_reduce(solution_indices)
        
        if len(reduced_indices) < best_len:
            best_len = len(reduced_indices)
            best_solution = [all_words[i] for i in reduced_indices]
            
        seed += 1
        
    if best_solution is None:
        # Fallback to a simple construction if something went wrong or no time
        # This is a valid covering code, just not optimal
        # Tetracode construction for q=3, n=4, radius=1 has size 9? No, K_3(4,1)=9.
        # For n=7, we can use product codes or just return a large set.
        # Better fallback: return all words? No, that's huge.
        # Let's return the result of the first greedy if no loop completed.
        # But the loop should complete at least once.
        
        # If best_solution is still None, it means we didn't even finish one greedy?
        # Greedy is fast. Let's just do one greedy and return it.
        solution_indices = greedy_solve(0)
        best_solution = [all_words[i] for i in solution_indices]
        
    return best_solution
