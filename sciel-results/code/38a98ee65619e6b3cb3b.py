import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Handle general case fallback
    def get_fallback():
        if n == 4 and q == 3 and radius == 1:
            # Tetracode
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n > 4 and q == 3 and radius == 1:
            # Direct sum with tetracode
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # General greedy for other params
        return _general_greedy(q, n, radius, deadline)

    if q != 3 or n != 8 or radius != 1:
        return get_fallback()

    total_words = 3 ** 8
    num_neighbors = 1 + n * (q - 1) # 17

    # Precompute neighbors
    # int_to_digits
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1

    best_code = None
    best_size = total_words

    # Run multiple randomized greedy attempts
    # We will run as many as possible before deadline
    seed = 42
    while time.time() < deadline - 5.0:
        random.seed(seed)
        seed += 1
        
        uncovered = np.ones(total_words, dtype=np.uint8)
        selected_indices = []
        
        # Latin Hypercube Sampling initialization for diverse, low-correlation codewords
        # This creates a sparse initial cover with minimal ball overlap
        lhs_size = 200  # Start with a diverse set
        if lhs_size < total_words:
            # Generate Latin hypercube samples
            lhs_indices = np.zeros(lhs_size, dtype=np.int32)
            for i in range(lhs_size):
                # Random offset for each position
                offsets = [random.randint(0, q - 1) for _ in range(n)]
                # Create a word based on the index
                word_val = 0
                for pos in range(n):
                    digit = (i + offsets[pos]) % q
                    word_val += digit * (q ** pos)
                lhs_indices[i] = word_val
            
            # Remove duplicates
            unique_lhs = np.unique(lhs_indices)
            
            # Add LHS codewords to initial cover
            for idx in unique_lhs:
                if uncovered[idx]:  # Only if not already covered
                    selected_indices.append(int(idx))
                    uncovered[neighbors[idx]] = 0
        
        # Greedy Phase
        while uncovered.sum() > 0:
            # Calculate gains
            # uncovered[neighbors] is (total_words, 17)
            # sum along axis 1
            gains = uncovered[neighbors].sum(axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random pick from candidates
            if len(candidates) > 1:
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
                
            selected_indices.append(int(best_idx))
            uncovered[neighbors[best_idx]] = 0
            
            # Early exit if we already exceeded best_size
            if len(selected_indices) >= best_size:
                break
        
        if len(selected_indices) >= best_size:
            continue

        # Redundancy Removal Phase
        # Build covered count array
        covered_counts = np.zeros(total_words, dtype=np.uint8)
        for idx in selected_indices:
            covered_counts[neighbors[idx]] += 1
            
        # Iteratively remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # We need to check each codeword.
            # To be efficient, we can check them in random order or just iterate.
            # Since removal changes the array, we restart the loop after a removal.
            # To avoid O(N^2) in worst case, we can shuffle.
            # But N is ~500, so 500*500 = 250k checks. Each check is 17 ops. Very fast.
            
            # Shuffle indices to break symmetry
            indices_to_check = list(range(len(selected_indices)))
            random.shuffle(indices_to_check)
            
            for i in indices_to_check:
                if time.time() > deadline - 5.0:
                    break
                
                idx = selected_indices[i]
                # Check if all neighbors are covered by > 1 codeword
                # covered_counts[neighbors[idx]] > 1
                if np.all(covered_counts[neighbors[idx]] > 1):
                    # Redundant
                    covered_counts[neighbors[idx]] -= 1
                    selected_indices.pop(i)
                    changed = True
                    break # Restart loop with new state

        if len(selected_indices) < best_size:
            best_size = len(selected_indices)
            best_code = [int_to_digits(idx, n, q) for idx in selected_indices]
            
    if best_code is None:
        return get_fallback()
        
    return best_code

def _general_greedy(q, n, radius, deadline):
    # Simple greedy for other parameters
    total_words = q ** n
    # This might be too slow for large n/q, but fallback is only for non-target cases
    # For safety, return full space if too big
    if total_words > 10000:
        return [list(t) for t in __import__('itertools').product(range(q), repeat=n)]
        
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    # Precompute neighbors
    num_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1

    uncovered = np.ones(total_words, dtype=np.uint8)
    selected_indices = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        gains = uncovered[neighbors].sum(axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_indices.append(int(best_idx))
        uncovered[neighbors[best_idx]] = 0
        
    return [int_to_digits(idx, n, q) for idx in selected_indices]
