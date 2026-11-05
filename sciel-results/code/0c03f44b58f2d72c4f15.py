import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute the spheres of radius 1 for all words
    # sphere[i] is a list of words covered by word i
    spheres = []
    for i in range(total_words):
        # Convert i to base q representation
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        
        covered = [i]
        for pos in range(n):
            for val in range(q):
                if val == digits[pos]:
                    continue
                # Create new word with val at pos
                new_digits = digits[:]
                new_digits[pos] = val
                # Convert back to integer
                new_word = 0
                for d in new_digits:
                    new_word = new_word * q + d
                covered.append(new_word)
        spheres.append(covered)
    
    # Initial code using the tetracode construction
    # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # This gives 9 words of length 4.
    # We extend to length 6 by adding all 3^2=9 suffixes.
    # Total 81 words.
    
    initial_code = []
    for a in range(3):
        for b in range(3):
            head = [a, b, (a + b) % 3, (a + 2 * b) % 3]
            for c in range(3):
                for d in range(3):
                    word = head + [c, d]
                    # Convert to integer
                    w_int = 0
                    for val in word:
                        w_int = w_int * 3 + val
                    initial_code.append(w_int)
    
    # Mark covered words
    covered = np.zeros(total_words, dtype=bool)
    for w in initial_code:
        for u in spheres[w]:
            covered[u] = True
    
    current_code = initial_code.copy()
    
    # Greedy Set Cover
    # We want to add words that cover the most uncovered points.
    # To speed this up, we can just iterate through all words and pick the best one.
    # Since total_words is only 729, this is fast.
    
    while True:
        if time.time() > deadline:
            break
        
        uncovered_count = np.sum(~covered)
        if uncovered_count == 0:
            break
            
        best_word = -1
        max_new_covered = 0
        
        # Check all possible words to see which one covers the most uncovered words
        # We can skip words that are already in the code, but it doesn't hurt to check them (gain 0)
        for w in range(total_words):
            if covered[w]:
                # If the center is already covered, it might still cover other uncovered neighbors
                pass
            # Calculate how many new words this would cover
            new_covered = 0
            for u in spheres[w]:
                if not covered[u]:
                    new_covered += 1
            
            if new_covered > max_new_covered:
                max_new_covered = new_covered
                best_word = w
        
        if max_new_covered == 0:
            # Should not happen if uncovered_count > 0, but just in case
            break
        
        # Add the best word to the code
        current_code.append(best_word)
        for u in spheres[best_word]:
            covered[u] = True
            
        # If we have improved significantly, we might want to check for removals?
        # Actually, greedy set cover doesn't guarantee optimality, but it's a good heuristic.
        # Let's do a local search: try to remove a word if possible.
        # This can be done periodically or after each addition.
        # Let's try removing a word after each addition to keep the code minimal.
        
        # Try to remove a word
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_code)):
                if time.time() > deadline:
                    break
                # Temporarily remove current_code[i]
                w = current_code[i]
                # Check if all words covered by w are covered by other words
                # We need to count how many words are covered ONLY by w.
                # If 0, we can remove w.
                
                # This check is O(n * size_of_code) if done naively.
                # Better: maintain a count of how many times each word is covered.
                # But we don't have that array.
                
                # Let's just check if removing w leaves any hole.
                # A hole exists if there is a word u in spheres[w] such that no other v in current_code \ {w} covers u.
                # This is equivalent to: for all u in spheres[w], there exists v in current_code \ {w} such that u in spheres[v].
                
                # Since spheres are small (size 13), this is fast.
                # For each u in spheres[w], check if any other v covers u.
                
                can_remove = True
                for u in spheres[w]:
                    covered_by_other = False
                    for j in range(len(current_code)):
                        if j == i:
                            continue
                        if u in spheres[current_code[j]]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove w
                    current_code.pop(i)
                    improved = True
                    break # Restart the removal loop as indices changed
        
        if time.time() > deadline:
            break

    # Convert current_code (list of ints) back to list of lists
    result = []
    for w in current_code:
        digits = []
        tmp = w
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        result.append(digits)
        
    return result
