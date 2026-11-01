import time
import numpy as np
import random
import itertools

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

    # Precompute "inverse neighbors" (preimage) for faster swap search
    # inv_neighbors[v] is the list of codewords c such that v is in neighbors[c]
    # Since Hamming distance is symmetric, inv_neighbors[v] = neighbors[v]
    # So we can just reuse the neighbors array.

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
        
        # Greedy Phase
        while uncovered.sum() > 0:
            if time.time() > deadline - 10.0:
                break
            # Calculate gains
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
                
            selected_indices.append(best_idx)
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

        # New Phase: Local Search / Swap Optimization
        # Try to replace a codeword with another to potentially allow further reductions
        # Or simply try to find a better arrangement.
        # The inspiration suggests a "Swap" phase where we replace a codeword c with c'
        # if c' covers the "critical" words of c (words covered only by c) and doesn't uncover others.
        
        # Let's implement a simple local search:
        # 1. Identify critical words for each codeword.
        # 2. Try to replace codeword c with c' if c' covers all critical words of c.
        # 3. If successful, the coverage count of critical words remains >= 1 (actually becomes 1 from c' if c was the only one, but we remove c so it becomes 1 from c').
        # Wait, if we remove c, critical words become 0 unless covered by c'.
        # So c' MUST cover all critical words of c.
        # Also, c' might cover words that were previously covered by c, but that's fine (count increases).
        # The main benefit is if this swap makes some OTHER codeword redundant.
        # So the strategy is:
        # - Do a pass of swaps.
        # - Then do a pass of removals.
        # - Repeat.

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Shuffle to break symmetry
            indices_to_check = list(range(len(selected_indices)))
            random.shuffle(indices_to_check)
            
            for i in indices_to_check:
                if time.time() > deadline - 5.0:
                    break
                idx = selected_indices[i]
                
                # Find critical words: words in neighbors[idx] with count == 1
                crit_mask = (covered_counts[neighbors[idx]] == 1)
                crit_words = neighbors[idx][crit_mask]
                
                if len(crit_words) == 0:
                    # Should have been removed in previous phase, but just in case
                    covered_counts[neighbors[idx]] -= 1
                    selected_indices.pop(i)
                    improved = True
                    continue

                # Find candidates c' that cover all crit_words
                # A candidate c' is valid if for all v in crit_words, v is in neighbors[c']
                # This is equivalent to: crit_words is a subset of neighbors[c']
                
                # Efficient way:
                # For each v in crit_words, the set of codewords covering v is neighbors[v] (inverse neighbors).
                # We need the intersection of neighbors[v] for all v in crit_words.
                
                # Start with all possible candidates
                # Since N is small, we can just iterate through all words? No, 6561 is large.
                # But crit_words is small (usually 1-3 words).
                # Let's intersect the neighbor sets.
                
                # Initialize candidate set as neighbors of the first critical word
                # neighbors[crit_words[0]] gives all codewords that cover crit_words[0]
                candidates = set(neighbors[crit_words[0]].tolist())
                
                for v in crit_words[1:]:
                    candidates &= set(neighbors[v].tolist())
                    if not candidates:
                        break
                
                if not candidates:
                    continue

                # Remove the current codeword itself from candidates
                candidates.discard(idx)
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                c_prime = random.choice(list(candidates))
                
                # Perform the swap
                # Remove idx, add c_prime
                covered_counts[neighbors[idx]] -= 1
                covered_counts[neighbors[c_prime]] += 1
                selected_indices[i] = c_prime
                improved = True
                # Break to restart the removal/swap loop with the new state
                # This is simpler and safer than continuing the loop
                break

            # After a pass of swaps, do another pass of removals
            if improved:
                changed = True
                while changed and time.time() < deadline - 5.0:
                    changed = False
                    indices_to_check = list(range(len(selected_indices)))
                    random.shuffle(indices_to_check)
                    
                    for i in indices_to_check:
                        if time.time() > deadline - 5.0:
                            break
                        
                        idx = selected_indices[i]
                        # Check if all neighbors are covered by > 1 codeword
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
        selected_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = 0
        
    return [int_to_digits(idx, n, q) for idx in selected_indices]
