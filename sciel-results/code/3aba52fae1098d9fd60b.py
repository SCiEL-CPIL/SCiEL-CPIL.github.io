import itertools
import time

def construct(q, n, radius):
    """
    Constructs a covering code for the given parameters.
    Currently optimized for q=3, n=7, radius=1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy for small spaces
        if q ** n > 100000:
            raise ValueError("Large space not supported by fallback")
        
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        space_size = len(all_words)
        
        def hamming_dist(w1, w2):
            return sum(1 for a, b in zip(w1, w2) if a != b)
        
        balls = []
        for w in all_words:
            ball = set()
            for v in all_words:
                if hamming_dist(w, v) <= radius:
                    ball.add(word_to_idx[v])
            # Convert to int bitset
            mask = 0
            for idx in ball:
                mask |= (1 << idx)
            balls.append(mask)
            
        full_mask = (1 << space_size) - 1
        uncovered = full_mask
        selected = []
        available = list(range(space_size))
        
        while uncovered != 0:
            best_c = -1
            best_count = -1
            for c in available:
                count = bin(balls[c] & uncovered).count('1')
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1: break
            selected.append(best_c)
            uncovered &= ~balls[best_c]
            if time.time() > deadline:
                break
        return [list(all_words[c]) for c in selected]

    # Optimized path for q=3, n=7, radius=1
    space_size = 3 ** 7
    full_mask = (1 << space_size) - 1
    
    # Precompute all words and their indices
    # We can iterate 0 to 2186 and convert to base-3
    words = []
    for i in range(space_size):
        w = []
        val = i
        for _ in range(n):
            w.append(val % 3)
            val //= 3
        w.reverse() # Most significant digit first? 
        # The order doesn't matter for distance as long as it's consistent.
        # Let's stick to the order generated: index i corresponds to base-3 representation.
        # Let's ensure standard representation.
        words.append(w)
        
    # Precompute ball masks
    # A word w is in the ball of c if dist(w, c) <= 1
    # dist <= 1 means w == c or w differs from c in exactly 1 position.
    balls = [0] * space_size
    
    for c_idx in range(space_size):
        c = words[c_idx]
        mask = 1 << c_idx # Include c itself
        
        # Include words at distance 1
        for pos in range(n):
            orig_val = c[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                new_word = c.copy()
                new_word[pos] = new_val
                # Find index of new_word
                # We need a fast mapping from word to index
                # Since we generated words by iterating i and converting to base 3,
                # we can just compute the index directly.
                # index = sum(w[i] * 3^(n-1-i))
                # Wait, my generation above:
                # val = i
                # w.append(val % 3) -> w[0] is least significant
                # w.reverse() -> w[0] is most significant
                # So index = w[0]*3^6 + w[1]*3^5 + ...
                
                # Let's precompute a lookup table for speed
                pass
        
    # Faster index calculation
    # index = 0
    # for digit in word:
    #    index = index * 3 + digit
    
    # Re-do precomputation with direct index calc
    balls = [0] * space_size
    for c_idx in range(space_size):
        c = words[c_idx]
        mask = 1 << c_idx
        
        # Distance 0
        # mask already has c_idx
        
        # Distance 1
        for pos in range(n):
            orig_val = c[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_word = c.copy()
                new_word[pos] = new_val
                # Calculate index
                new_idx = 0
                for d in new_word:
                    new_idx = new_idx * 3 + d
                mask |= (1 << new_idx)
        
        balls[c_idx] = mask

    # Greedy Set Cover
    uncovered = full_mask
    selected = []
    available = list(range(space_size))
    
    while uncovered != 0:
        best_c = -1
        best_count = -1
        for c in available:
            # Count how many new words this codeword covers
            count = bin(balls[c] & uncovered).count('1')
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c == -1:
            break
            
        selected.append(best_c)
        uncovered &= ~balls[best_c]
        
        if time.time() > deadline:
            break

    # Local Search: Remove redundant codewords
    # A codeword c is redundant if the union of balls of all other selected codewords covers the entire space.
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            # Check if time is running out
            if time.time() > deadline - 5.0:
                break
            
            # Compute union of all other balls
            union_others = 0
            for j in range(len(selected)):
                if i != j:
                    union_others |= balls[selected[j]]
                    
            if union_others == full_mask:
                # Codeword selected[i] is redundant
                selected.pop(i)
                improved = True
                break # Restart loop to handle index shifts
        
        if time.time() > deadline - 5.0:
            break

    # Convert indices back to word lists
    result = [words[c] for c in selected]
    
    return result
