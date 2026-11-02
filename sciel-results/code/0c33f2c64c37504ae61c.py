import itertools
import time

def construct(q, n, radius):
    # Default fallback: Tetracode construction
    def construct_tetracode(q, n, radius):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if q != 3 or n != 6 or radius != 1:
        return construct_tetracode(q, n, radius)

    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin

    # Try to construct a smaller code using n=5 + n=1 split
    # Find a covering code for q=3, n=5, R=1
    # Space size 243
    # Sphere size 1 + 5*2 = 11
    
    all_words = list(itertools.product(range(3), repeat=5))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres
    spheres = []
    for w in all_words:
        sphere = set([w])
        for i in range(5):
            for d in [-1, 1]:
                new_w = list(w)
                new_w[i] = (w[i] + d) % 3
                sphere.add(tuple(new_w))
        spheres.append(sphere)
        
    covered = [False] * 243
    code = []
    
    # Greedy selection
    # To improve quality, we can use a heuristic:
    # Score = number of uncovered words in sphere.
    # If tie, prefer words that cover words with fewer other covers (scarcity).
    # But simple greedy is fast. Let's try simple greedy first.
    
    while True:
        if time.time() > deadline - 5.0:
            break
            
        # Find best candidate
        best_idx = -1
        best_count = 0
        best_tie_breaker = 0
        
        for i, w in enumerate(all_words):
            # Count uncovered in sphere
            count = 0
            for v in spheres[i]:
                if not covered[word_to_idx[v]]:
                    count += 1
            
            if count > best_count or (count == best_count and count > 0 and i < best_idx):
                best_count = count
                best_idx = i
                
        if best_count == 0:
            break
            
        # Add to code
        code.append(all_words[best_idx])
        for v in spheres[best_idx]:
            covered[word_to_idx[v]] = True
            
    # Check if we finished in time and covered everything
    if all(covered):
        # Construct final code: combine C5 with all length-1 words
        final_code = []
        for c in code:
            for x in range(3):
                final_code.append(list(c) + [x])
        return final_code
    else:
        # Fallback to tetracode if greedy didn't finish or failed
        return construct_tetracode(q, n, radius)
