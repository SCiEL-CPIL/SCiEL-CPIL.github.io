"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized local search for q=3, n=6, radius=1 using integer representations.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        neighbors = {}
        for w in all_words:
            nb = set()
            for v in all_words:
                if sum(1 for i in range(n) if w[i] != v[i]) <= radius:
                    nb.add(v)
            neighbors[w] = nb
        
        uncovered = set(all_words)
        code = []
        while uncovered:
            best_word = None
            max_covered = -1
            for w in all_words:
                if w in code:
                    continue
                covered_count = len(neighbors[w] & uncovered)
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_word = w
            if best_word is None:
                break
            code.append(best_word)
            uncovered -= neighbors[best_word]
        return [list(c) for c in code]

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** n
    
    def to_int(w):
        val = 0
        for d in w:
            val = val * 3 + d
        return val

    def to_tuple(i):
        w = [0] * n
        for j in range(n - 1, -1, -1):
            w[j] = i % 3
            i //= 3
        return tuple(w)

    all_ints = list(range(total_words))
    
    # Precompute neighbors as sets of ints
    neighbors = {}
    for w_int in all_ints:
        w = to_tuple(w_int)
        nb = {w_int}
        for i in range(n):
            for val in range(3):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                nb.add(to_int(tuple(new_w)))
        neighbors[w_int] = nb

    # Start with tetracode-based construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            initial_code.add(to_int(tuple(w)))
            
    code = set(initial_code)
    
    # Helper to check if code covers all words
    def is_full_cover(code_set):
        covered = set()
        for c in code_set:
            covered.update(neighbors[c])
        return len(covered) == total_words

    # 1. Removal Phase
    improved = True
    while improved:
        improved = False
        for c in list(code):
            new_code = code - {c}
            if is_full_cover(new_code):
                code = new_code
                improved = True
                break

    # 2. Swap Phase: Limited iterations to avoid timeout
    for _ in range(5):
        found_swap = False
        code_list = list(code)
        for c_remove in code_list:
            # Only try swaps with words that are not in code
            # Optimization: Iterate through all_ints, skip if in code
            for c_add in all_ints:
                if c_add in code:
                    continue
                new_code = (code - {c_remove}) | {c_add}
                if is_full_cover(new_code):
                    # Check if this swap allows a subsequent removal
                    # Try removing any word from new_code (except c_add)
                    for c_rem2 in list(new_code):
                        if c_rem2 == c_add:
                            continue
                        test_code = new_code - {c_rem2}
                        if is_full_cover(test_code):
                            code = test_code
                            found_swap = True
                            break
                if found_swap:
                    break
            if found_swap:
                break
        if not found_swap:
            break

    result = [to_tuple(c) for c in code]
    return [list(c) for c in result]

# EVOLVE-BLOCK-END
