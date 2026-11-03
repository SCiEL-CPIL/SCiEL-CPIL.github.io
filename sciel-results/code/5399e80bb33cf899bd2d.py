import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    N = q ** n
    # For q=3, n=8: N = 6561
    
    # Precompute powers of q
    powers = [q ** i for i in range(n)]
    
    # Precompute neighbors: for each codeword c, list of all words within distance <= radius
    # For radius=1: the word itself + n*(q-1) neighbors
    # We'll store as numpy arrays for vectorized operations
    neighbors = np.empty((N, n * (q - 1) + 1), dtype=np.int32)
    neighbor_counts = np.empty(N, dtype=np.int32)
    
    for c in range(N):
        idx = 0
        neighbors[c, idx] = c
        idx += 1
        for i in range(n):
            base = powers[i]
            val = (c // base) % q
            for new_val in range(q):
                if new_val != val:
                    neighbors[c, idx] = c + (new_val - val) * base
                    idx += 1
        neighbor_counts[c] = idx
    
    # Helper to decode an integer to a list of digits
    def decode(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits
    
    # Greedy set cover with vectorized gain computation
    def greedy_set_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        
        # uncovered: boolean array of length N
        uncovered = np.ones(N, dtype=np.bool_)
        selected = []
        
        while np.any(uncovered):
            # Compute gain for each candidate: number of newly covered points
            # gain[c] = sum of uncovered[neighbors[c, :]]
            # Vectorized: for each c, sum uncovered over its neighbors
            gains = np.zeros(N, dtype=np.int32)
            for c in range(N):
                cnt = neighbor_counts[c]
                gains[c] = np.sum(uncovered[neighbors[c, :cnt]])
            
            # Find the candidate with maximum gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Random tie-breaking among candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_c = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_c = candidates[0]
            
            selected.append(int(best_c))
            # Update uncovered
            cnt = neighbor_counts[best_c]
            uncovered[neighbors[best_c, :cnt]] = False
        
        return selected
    
    # Local search: try to remove redundant codewords
    def local_search_remove(selected, time_limit):
        selected = list(selected)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Try to remove each codeword in random order
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(selected):
                    continue
                c = selected[i]
                # Check if removing c leaves all points covered
                # Compute union of coverage of all other selected codewords
                # and check if it covers everything
                # Efficient check: for each point, is it covered by some other selected?
                # We can do this by checking if any other selected word is within distance 1 of each point
                # But that's expensive. Instead, we maintain coverage counts.
                
                # Let's compute: for each point p, count how many selected words (other than c) cover it
                # If any point has count 0, we can't remove c.
                
                # Use numpy for speed
                other_selected = [selected[j] for j in range(len(selected)) if j != i]
                if not other_selected:
                    # Can't remove if it's the only one
                    continue
                
                # For each point, check if it's covered by any other selected word
                # A point p is covered by word w if p is in neighbors[w]
                # Equivalently, w is in neighbors[p]
                
                # Build a set of other selected words for O(1) lookup
                other_set = set(other_selected)
                
                # For each point p, check if any w in other_set is in neighbors[p]
                # This is expensive if done naively. Let's use a different approach.
                
                # Alternative: compute the union of coverage of other selected words
                # and check if it equals the full set.
                
                # Coverage of a word w is the set of points within distance 1 of w.
                # Union of coverage of other selected = set of points covered by at least one other selected.
                
                # Let's compute this using a boolean array
                covered = np.zeros(N, dtype=np.bool_)
                for w in other_selected:
                    cnt = neighbor_counts[w]
                    covered[neighbors[w, :cnt]] = True
                
                if np.all(covered):
                    selected.pop(i)
                    changed = True
                    break  # Restart with new list
        
        return selected
    
    # Run multiple iterations of greedy + local search
    best_code = None
    best_size = N + 1
    
    num_iterations = 20  # We'll adjust based on time
    
    for it in range(num_iterations):
        if time.time() > deadline - 10:
            break
        
        # Greedy set cover with random seed for diversity
        selected = greedy_set_cover(seed=it * 12345 + 678)
        
        if len(selected) >= best_size:
            # Still might improve with local search, but skip if clearly worse
            # Actually, local search can only remove, so if already >= best, skip
            # But we might get lucky... let's still try if we have time
            if time.time() > deadline - 15:
                continue
        
        # Local search: remove redundant codewords
        time_limit = deadline - 5
        selected = local_search_remove(selected, time_limit)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
    
    # If we didn't find a valid code (shouldn't happen), fallback
    if best_code is None:
        # Do one more greedy without local search
        selected = greedy_set_cover(seed=99999)
        best_code = selected
        best_size = len(selected)
    
    # Convert to list of lists
    result = []
    for c in best_code:
        result.append(decode(c))
    
    return result
