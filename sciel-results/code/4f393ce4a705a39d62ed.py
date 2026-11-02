import itertools

def construct(q, n, radius):
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    # Pre-compute neighbors for each word to speed up coverage update?
    # Or just compute on the fly. n=7 is small.
    # Let's pre-compute the "ball" of radius 1 for each word.
    # ball[w] = set of words within distance <= radius of w
    
    # This might use a lot of memory if not careful, but 2187 * 15 is small.
    balls = {}
    for w in all_words:
        neighbors = [w]
        for i in range(n):
            for delta in range(1, q):
                new_w = list(w)
                new_w[i] = (w[i] + delta) % q
                neighbors.append(tuple(new_w))
        balls[w] = set(neighbors)
        
    while uncovered:
        # Find the word in uncovered that covers the most uncovered words
        best_word = None
        max_covered = -1
        
        # We can sample if uncovered is large, but it shrinks fast.
        # Iterating all uncovered words is fine.
        for w in uncovered:
            # Count how many of w's neighbors are in uncovered
            # Note: w itself is in uncovered.
            covered_count = sum(1 for nb in balls[w] if nb in uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
                
        if best_word is None:
            break
            
        code.append(list(best_word))
        
        # Remove covered words from uncovered
        for nb in balls[best_word]:
            if nb in uncovered:
                uncovered.remove(nb)
                
    return code
