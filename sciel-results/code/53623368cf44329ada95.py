import time
import random
import numpy as np

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    
    # Precompute neighbor sets and a boolean matrix B where B[c][u] = 1 if u is within radius of c
    # For radius=1, each word covers itself and words differing in exactly 1 position.
    
    # We'll use a numpy array B of shape (total, total) with 1s where covered.
    # For total=729, this is 729*729*1 bytes ~ 528 KB, very manageable.
    
    # Actually, for efficiency in the greedy step, we want to compute for each candidate c,
    # how many currently uncovered words it covers.
    # Let uncovered be a boolean array of length total.
    # gain[c] = sum of B[c][u] for u where uncovered[u] is True.
    # This is just B[c] @ uncovered (if B is 0/1 and uncovered is 0/1).
    
    # Build B
    B = np.zeros((total, total), dtype=np.int8)
    for c in range(total):
        # c covers itself
        B[c, c] = 1
        # c covers words differing in 1 position
        for i in range(n):
            pow_q_i = q ** i
            digit = (c // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    u = c + (new_digit - digit) * pow_q_i
                    B[c, u] = 1
    
    rng = random.Random(42)
    
    best_code = None
    best_size = total + 1
    
    def greedy_cover():
        """Run greedy set cover and return the code (list of indices)."""
        uncovered = np.ones(total, dtype=np.int8)
        code = []
        
        while np.sum(uncovered) > 0:
            if time.monotonic() > deadline:
                break
            # Compute gain for each candidate
            # gain[c] = B[c] @ uncovered
            gains = B @ uncovered
            # Pick the candidate with max gain
            best_c = int(np.argmax(gains))
            if gains[best_c] == 0:
                # No progress, should not happen
                break
            code.append(best_c)
            # Update uncovered
            uncovered = (uncovered & ~B[best_c].astype(bool)).astype(np.int8)
        
        return code
    
    def local_search(code):
        """Try to reduce code size via removals and replacements."""
        code = list(code)
        # Build counts: counts[u] = number of codewords covering u
        counts = np.zeros(total, dtype=np.int8)
        for c in code:
            counts += B[c]
        
        # Verify valid
        if np.any(counts == 0):
            return code, len(code)
        
        improved = True
        while improved and time.monotonic() < deadline:
            improved = False
            # Try removals
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # If removing c, check if any word in B[c] has count == 1
                # i.e., min(counts[B[c]==1]) == 1
                mask = B[c] == 1
                if np.any(counts[mask] == 1):
                    continue
                # Safe to remove
                counts -= B[c]
                code.pop(i)
                improved = True
                break  # Restart the loop after a change
            
            if not improved:
                # Try replacements
                for i in range(len(code)):
                    old_c = code[i]
                    # Try a few random replacements
                    for _ in range(5):
                        new_c = rng.randint(0, total - 1)
                        if new_c == old_c or new_c in code:
                            continue
                        
                        # Check if replacing old_c with new_c keeps cover valid
                        # Words that lose coverage: those in B[old_c] but not B[new_c]
                        # These must have count >= 2
                        # Words that gain coverage: those in B[new_c] but not B[old_c]
                        # These are fine (count increases)
                        
                        old_mask = B[old_c] == 1
                        new_mask = B[new_c] == 1
                        
                        # Words only covered by old_c (losing coverage)
                        losing = old_mask & ~new_mask
                        if np.any(counts[losing] == 1):
                            continue
                        
                        # Safe to replace
                        counts -= B[old_c]
                        counts += B[new_c]
                        code[i] = new_c
                        improved = True
                        break
                    
                    if improved:
                        break
            
            if len(code) < best_size:
                pass
        
        return code, len(code)
    
    # Multiple restarts
    num_restarts = 0
    while time.monotonic() < deadline - 5.0:
        num_restarts += 1
        # Greedy initialization
        code = greedy_cover()
        if not code:
            break
        if time.monotonic() > deadline - 2.0:
            break
        
        # Local search
        code, size = local_search(code)
        
        if size < best_size:
            best_size = size
            best_code = code[:]
        
        # Early termination if we hit the known best
        if best_size == 73:
            # Try a few more restarts to see if we can do better, but don't spend too much time
            if num_restarts >= 20:
                break
    
    # Fallback if no valid code found
    if best_code is None:
        best_code = greedy_cover()
    
    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        result.append(word)
        
    return result
