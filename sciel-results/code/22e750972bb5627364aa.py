import numpy as np
import itertools
import time

def construct(q, n, radius):
    # This task is specifically for q=3, n=7, radius=1
    # But we should handle general case if possible, or just assume the specific case
    # based on the task description "Football pool problem for 7 matches... ternary covering code K_3(7,1)"
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to simple construction for other cases if they appear (unlikely)
        # Direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    deadline = time.time() + 110  # 110 seconds to be safe
    
    # Precompute balls for all words
    # ball_indices[word] = list of word indices covered by this word (including itself)
    # Actually, we need balls for *centers*. Since centers are also words, it's the same.
    # A center c covers words w such that dist(c, w) <= 1.
    
    # Generate all words as tuples or base-3 integers?
    # Base-3 integers are faster for hashing and set operations.
    
    def to_base3_int(word):
        val = 0
        for i in range(n):
            val = val * 3 + word[i]
        return val

    def from_base3_int(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % 3
            val //= 3
        return word

    # Map index (0..2186) to word tuple/int
    # We will use integer representation for words to speed up distance/coverage checks?
    # Or just precompute the neighbor lists.
    
    all_words = [from_base3_int(i) for i in range(total_words)]
    
    # Precompute coverage: cover[word_idx] = list of center_idx that cover word_idx
    # And balls[center_idx] = list of word_idx covered by center_idx
    
    balls = [[] for _ in range(total_words)]
    # covers[word_idx] = list of center_idx covering word_idx (same as balls since symmetric)
    
    for c_idx, c_word in enumerate(all_words):
        # c covers itself
        balls[c_idx].append(c_idx)
        # c covers neighbors at distance 1
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = c_word[pos] + delta
                if 0 <= new_val < 3:
                    new_word = c_word.copy()
                    new_word[pos] = new_val
                    w_idx = to_base3_int(new_word)
                    balls[c_idx].append(w_idx)
    
    # Note: balls[c_idx] contains unique indices?
    # Yes, because we iterate distinct positions and deltas.
    # However, for q=3, delta -1 and 1 are distinct.
    # So len(balls[c_idx]) is always 1 + n*2 = 15?
    # Wait, distance 1 means exactly one position different.
    # In ternary, for each position, there are 2 other values.
    # So 1 + 7*2 = 15 words in the ball.
    
    # 1. Greedy Construction
    covered = np.zeros(total_words, dtype=bool)
    code = []
    code_set = set()
    
    while not covered.all():
        # Find center with max gain
        max_gain = -1
        best_c = -1
        for c in range(total_words):
            # Count how many words in balls[c] are not covered
            # We can do this by iterating balls[c]
            gain = 0
            for w in balls[c]:
                if not covered[w]:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_c = c
                if gain == 15: # Max possible
                    break
        
        if best_c == -1:
            break # Should not happen if not covered.all()
            
        code.append(best_c)
        code_set.add(best_c)
        for w in balls[best_c]:
            covered[w] = True
            
    # 2. Local Search
    # cover_count[w] = number of codewords covering w
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        for w in balls[c]:
            cover_count[w] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to remove
        removed = False
        for i in range(len(code)):
            c = code[i]
            # Decrement
            for w in balls[c]:
                cover_count[w] -= 1
            
            # Check min
            if np.min(cover_count) >= 1:
                # Remove c
                code.pop(i)
                code_set.remove(c)
                removed = True
                improved = True
                break
            else:
                # Restore
                for w in balls[c]:
                    cover_count[w] += 1
                
                # Try Swap
                # Find critical words (count == 0)
                critical = np.where(cover_count == 0)[0]
                
                if len(critical) == 1:
                    w_crit = critical[0]
                    # Find c' in balls[w_crit] not in code_set
                    for c_prime in balls[w_crit]:
                        if c_prime not in code_set:
                            # Swap
                            code[i] = c_prime
                            code_set.remove(c)
                            code_set.add(c_prime)
                            
                            # Update counts
                            for w in balls[c]:
                                cover_count[w] -= 1
                            for w in balls[c_prime]:
                                cover_count[w] += 1
                            
                            improved = True
                            removed = True # Flag to break outer loop
                            break
                elif len(critical) == 2:
                    w1, w2 = critical
                    # Intersection of balls[w1] and balls[w2]
                    set1 = set(balls[w1])
                    for c_prime in balls[w2]:
                        if c_prime in set1 and c_prime not in code_set:
                            # Swap
                            code[i] = c_prime
                            code_set.remove(c)
                            code_set.add(c_prime)
                            
                            for w in balls[c]:
                                cover_count[w] -= 1
                            for w in balls[c_prime]:
                                cover_count[w] += 1
                            
                            improved = True
                            removed = True
                            break
                # If len(critical) > 2, no single swap can cover all, skip
                
        # If we removed or swapped, loop continues to try more
        # If not, we are at a local optimum
        
    # Convert code indices back to lists of ints
    result = [all_words[c] for c in code]
    return result
