import itertools
import time
import random

def construct(q, n, radius):
    # For this specific task: q=3, n=6, radius=1
    # Generalize slightly if possible, but optimize for 3,6,1
    
    total_words = q ** n
    # Map words to indices
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls (coverage sets) as bitmasks
    # Ball of word i: all words j such that dist(i, j) <= radius
    balls = [0] * total_words
    for i, w_i in enumerate(words):
        mask = 0
        # Generate all words in the ball of radius 1 around w_i
        # For radius 1, it's w_i itself and w_i with one coordinate changed
        for j in range(total_words):
            w_j = words[j]
            # Calculate Hamming distance
            dist = sum(1 for a, b in zip(w_i, w_j) if a != b)
            if dist <= radius:
                mask |= (1 << j)
        balls[i] = mask
        
    all_ones = (1 << total_words) - 1
    
    def get_best_code(seed=0):
        random.seed(seed)
        uncovered = all_ones
        code = []
        
        # Greedy construction
        while uncovered:
            best_idx = -1
            max_covered = 0
            candidates = []
            
            for i in range(total_words):
                # If i is already in code, skip? 
                # Actually, we just want to pick the one that covers most NEW words.
                # We don't need to skip if we haven't added it, but adding the same one again is useless.
                # Since we remove covered words from 'uncovered', if i was picked before, 
                # its contribution to 'uncovered' is 0 (since all its words are covered).
                # So we can just iterate all.
                
                covered = balls[i] & uncovered
                count = covered.bit_count() if hasattr(int, 'bit_count') else bin(covered).count('1')
                
                if count > max_covered:
                    max_covered = count
                    candidates = [i]
                elif count == max_covered and count > 0:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen if uncovered != 0 and balls cover space
                
            # Random tie-breaking
            chosen = random.choice(candidates)
            code.append(chosen)
            uncovered &= ~balls[chosen]
            
        # Local Search: Pruning
        # Maintain cover counts
        cover_count = [0] * total_words
        for c in code:
            # Add counts for words in ball[c]
            # This is slow if done naively. 
            # Better: just check validity by union of masks.
            pass
        
        # Simple pruning loop
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Check if removing c keeps the cover valid
                # Compute union of balls of code \ {c}
                union_mask = 0
                for d in code:
                    if d != c:
                        union_mask |= balls[d]
                
                if union_mask == all_ones:
                    # c is redundant
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
            
        return [words[i] for i in code]

    # Try multiple seeds to find the minimum
    best_code = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 110 # Leave 10s margin
    
    seed = 0
    while time.time() - start_time < timeout:
        code = get_best_code(seed)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # If we hit the known lower bound or a very good number, we can stop early?
            # Lower bound is 71. If we get 71, 72, or 73, it's great.
            if best_size <= 73:
                # We might still find 72 or 71, but 73 is the known best.
                # Let's run a few more to see if we can beat 73.
                pass
        seed += 1
        
    if best_code is None:
        # Fallback to seed program logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_code
