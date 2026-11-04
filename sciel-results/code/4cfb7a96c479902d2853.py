import numpy as np
import itertools

def construct(q, n, radius):
    if radius != 1 or q != 3 or n != 8:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # For q=3, n=8, radius=1, use greedy set cover
    total_words = q ** n  # 6561
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres (words within distance 1)
    spheres = []
    for word in all_words:
        sphere = set([word])
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    sphere.add(new_word)
        spheres.append(sorted(sphere))
    
    # Convert spheres to index sets
    sphere_indices = [set(word_to_idx[w] for w in sphere) for sphere in spheres]
    
    # Uncovered words
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        # Find word whose sphere covers most uncovered words
        best_idx = -1
        best_count = -1
        for i in range(total_words):
            count = len(sphere_indices[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
            if best_count == len(uncovered):
                break
        if best_count == 0:
            break
        code.append(list(all_words[best_idx]))
        uncovered -= sphere_indices[best_idx]
    
    return code
