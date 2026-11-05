import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # If parameters are not the specific small case, fall back to a simple greedy or seed-like method
    # But given the context, we assume q=3, n=6, radius=1.
    # To be safe, if q**n is too large for the O((q^n)^2) precomputation, we use a different strategy.
    # For q=3, n=6, q**n = 729. 729^2 = 531441. This is fine.
    
    total_words = q ** n
    
    if total_words > 5000:
        # Fallback: Simple greedy without full precomputation of all balls
        # This is a safety net, though unlikely to be triggered for this task.
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        def hamming(w1, w2):
            return sum(1 for a, b in zip(w1, w2) if a != b)
            
        uncovered = set(all_words)
        code = []
        
        while uncovered:
            best_w = None
            max_cov = 0
            for w in all_words:
                # Calculate coverage of w
                cov = 0
                for u in uncovered:
                    if hamming(w, u) <= radius:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_w = w
            if best_w is None:
                break
            code.append(best_w)
            # Remove covered words
            to_remove = []
            for u in uncovered:
                if hamming(best_w, u) <= radius:
                    to_remove.append(u)
            for u in to_remove:
                uncovered.remove(u)
                
            if time.time() > deadline - 5:
                break
                
        # Prune
        new_code = []
        for c in code:
            redundant = True
            # Check if c covers any word not covered by others
            # This is slow for large N, but safe.
            for u in all_words:
                if hamming(c, u) <= radius:
                    covered_by_other = False
                    for c2 in code:
                        if c2 == c:
                            continue
                        if hamming(c2, u) <= radius:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
            if not redundant:
                new_code.append(c)
        return new_code

    # Specific case: q=3, n=6, radius=1
    # Precompute balls
    all_words = list(itertools.product(range(q), repeat=n))
    
    def word_to_int(word):
        val = 0
        for d in word:
            val = val * q + d
        return val

    def int_to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    int_to_word_map = [None] * total_words
    for w in all_words:
        idx = word_to_int(w)
        int_to_word_map[idx] = w
        
    def hamming_distance(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)

    balls = [set() for _ in range(total_words)]
    
    for i in range(total_words):
        wi = int_to_word_map[i]
        for j in range(total_words):
            wj = int_to_word_map[j]
            if hamming_distance(wi, wj) <= radius:
                balls[i].add(j)

    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        best_c = -1
        max_covered = 0
        for c in range(total_words):
            covered = len(uncovered & balls[c])
            if covered > max_covered:
                max_covered = covered
                best_c = c
                
        code.append(best_c)
        uncovered -= balls[best_c]
        
        if time.time() > deadline - 10:
            break

    # Pruning
    new_code = []
    for c in code:
        is_redundant = True
        for w in balls[c]:
            covered_by_other = False
            for c_prime in code:
                if c_prime == c:
                    continue
                if w in balls[c_prime]:
                    covered_by_other = True
                    break
            if not covered_by_other:
                is_redundant = False
                break
        
        if not is_redundant:
            new_code.append(c)
            
    code = new_code
    
    result = [int_to_word_map[c] for c in code]
    
    return result
