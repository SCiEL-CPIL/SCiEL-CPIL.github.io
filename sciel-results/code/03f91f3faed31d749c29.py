import time
import random

def construct(q, n, radius):
    """
    Constructs a covering code for the given parameters.
    For the specific task (q=3, n=6, radius=1), it uses a fast greedy set-cover
    algorithm with bitset optimization to find a near-optimal code (target ~73 words).
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # If parameters are not the specific task, fall back to a simple construction
    # The task specifies q=3, n=6, radius=1.
    if q == 3 and n == 6 and radius == 1:
        # Precompute all words as integers 0 to 3**n - 1
        total_words = q ** n
        
        # Helper to convert integer to list of digits
        def int_to_word(idx):
            word = [0] * n
            for i in range(n):
                word[i] = idx % q
                idx //= q
            return word
        
        # Helper to convert list of digits to integer
        def word_to_int(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx

        # Precompute coverage mask for each potential codeword
        # coverage[i] is a bitmask where bit j is 1 if word j is covered by codeword i
        coverage = [0] * total_words
        
        for i in range(total_words):
            word = int_to_word(i)
            mask = 1 << i  # Cover itself
            
            # Cover neighbors (change one position to other q-1 values)
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = word[:]
                    new_word[pos] = new_val
                    j = word_to_int(new_word)
                    mask |= (1 << j)
            
            coverage[i] = mask

        # Greedy Set Cover
        # uncovered: bitmask of words not yet covered
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        while uncovered:
            if time.time() > deadline:
                # Fallback: return whatever we have if we ran out of time (unlikely)
                # But we need to ensure we return a valid list.
                # If we haven't covered everything, the result is invalid.
                # However, the greedy is so fast it won't happen.
                break
                
            best_idx = -1
            best_count = -1
            
            # Find the codeword that covers the most uncovered words
            # We can optimize by skipping codewords that are already covered?
            # No, a covered codeword can still cover new uncovered words.
            # But if a codeword is fully covered by the current code, adding it adds nothing.
            # Actually, we just need the one with max intersection.
            
            for i in range(total_words):
                # Quick check: if this codeword's coverage is a subset of already covered?
                # i.e., coverage[i] & uncovered == 0
                # We compute the intersection count.
                inter = coverage[i] & uncovered
                if inter:
                    # Count bits
                    # Python int.bit_count() is available in 3.10+
                    # Assuming modern Python
                    count = inter.bit_count()
                    if count > best_count:
                        best_count = count
                        best_idx = i
                        
            if best_idx == -1:
                # Should not happen if space is connected
                break
                
            code_indices.append(best_idx)
            uncovered &= ~coverage[best_idx]
            
        # Convert indices back to word lists
        code = [int_to_word(i) for i in code_indices]
        return code

    else:
        # Generic fallback for other parameters (not expected in this task)
        # Simple random or greedy might work, but let's stick to a basic construction
        # For q=3, n=6, the specific logic above is used.
        # If other params are passed, we might not know the optimal size.
        # Let's use a simple greedy for any q,n,radius.
        
        total_words = q ** n
        
        def int_to_word(idx):
            word = [0] * n
            for i in range(n):
                word[i] = idx % q
                idx //= q
            return word

        def word_to_int(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx

        # Generate neighbors for all words
        # This might be slow for large n, but for small n it's fine.
        coverage = [0] * total_words
        
        for i in range(total_words):
            word = int_to_word(i)
            mask = 1 << i
            # Generate all words within radius
            # For radius 1:
            if radius == 1:
                for pos in range(n):
                    orig_val = word[pos]
                    for new_val in range(q):
                        if new_val == orig_val:
                            continue
                        new_word = word[:]
                        new_word[pos] = new_val
                        j = word_to_int(new_word)
                        mask |= (1 << j)
            else:
                # For radius > 1, this generic generator is too slow/complex for a fallback.
                # But the task is specifically radius 1.
                pass
            
            coverage[i] = mask

        uncovered = (1 << total_words) - 1
        code_indices = []
        
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                inter = coverage[i] & uncovered
                if inter:
                    count = inter.bit_count()
                    if count > best_count:
                        best_count = count
                        best_idx = i
            if best_idx == -1:
                break
            code_indices.append(best_idx)
            uncovered &= ~coverage[best_idx]
            
        code = [int_to_word(i) for i in code_indices]
        return code
