"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    For the specific case q=3, n=6, radius=1, we use a heuristic based on local search
    starting from a known good configuration (like the direct sum or a random start) 
    and iteratively improving it by swapping or replacing codewords to maximize coverage.
    
    Since the problem size is small (3^6 = 729 words), we can afford to do some exact
    computations for coverage and use local search to find a smaller set.
    
    Strategy:
    1. Start with a valid covering code (e.g., the tetracode-based one from the seed, or a greedy one).
    2. Use local search: try to remove a codeword if coverage is maintained. If not, try swapping 
       it with another word that covers more uncovered words or allows removal of another word.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: use the direct sum construction if applicable, else greedy
        # For general case, a simple greedy might be too slow or suboptimal, but let's try a basic greedy
        # However, the task specifically targets q=3, n=6, radius=1.
        # Let's implement a generic greedy for now, but optimize for the specific case.
        pass
    
    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        # Total number of words in the space
        total_words = 3 ** n
        
        # Precompute neighbors for each word (distance <= 1)
        # word is a tuple of length n
        # neighbors[word] = set of words within Hamming distance 1
        neighbors = {}
        all_words = list(itertools.product(range(3), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        for w in all_words:
            nb = set()
            nb.add(w)
            for i in range(n):
                for val in range(3):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    new_w = tuple(new_w)
                    nb.add(new_w)
            neighbors[w] = nb
        
        # Helper function to compute coverage count for a given code set
        def get_coverage(code_set):
            covered = set()
            for c in code_set:
                covered.update(neighbors[c])
            return len(covered)
        
        # Helper function to find uncovered words
        def get_uncovered(code_set):
            covered = set()
            for c in code_set:
                covered.update(neighbors[c])
            return set(all_words) - covered
        
        # Start with the tetracode-based construction (9 * 3^2 = 81 words)
        # Tetracode: [a, b, (a+b)%3, (a+2b)%3] for a,b in 0..2
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                initial_code.append(tuple(h + list(tail)))
        
        code = set(initial_code)
        
        # Local search: try to remove words
        improved = True
        while improved:
            improved = False
            # Try removing each word
            for c in list(code):
                new_code = code - {c}
                new_coverage = get_coverage(new_code)
                if new_coverage == total_words:
                    code = new_code
                    improved = True
                    break # restart loop as code changed
        
        # Try swapping: remove one, add another
        # For each word in code, try replacing it with a word not in code
        # Check if coverage remains 729
        improved = True
        while improved:
            improved = False
            code_list = list(code)
            for c_remove in code_list:
                # Try adding each word not in code
                for c_add in all_words:
                    if c_add in code:
                        continue
                    new_code = (code - {c_remove}) | {c_add}
                    new_coverage = get_coverage(new_code)
                    if new_coverage == total_words:
                        code = new_code
                        improved = True
                        break
                if improved:
                    break
        
        # Additional local search: try to find better swaps by considering uncovered words
        # For each codeword, find which words it uniquely covers
        # Then try to replace it with a word that covers more uncovered words
        
        # Try more aggressive local search with random restarts
        import random
        best_code = set(code)
        best_size = len(code)
        
        for restart in range(50):
            # Start from current best or random perturbation
            if restart == 0:
                code = set(best_code)
            else:
                # Random perturbation: remove some random words and add others
                code = set(best_code)
                for _ in range(3):
                    if code:
                        # Remove a random word
                        remove_word = random.choice(list(code))
                        code.remove(remove_word)
                        # Add a random word to maintain coverage
                        uncovered = get_uncovered(code)
                        if uncovered:
                            # Find a word that covers the most uncovered words
                            best_add = None
                            max_covered = -1
                            for w in all_words:
                                if w in code:
                                    continue
                                covered_count = len(neighbors[w] & uncovered)
                                if covered_count > max_covered:
                                    max_covered = covered_count
                                    best_add = w
                            if best_add:
                                code.add(best_add)
                        else:
                            # Coverage is complete, just add a random word
                            random_word = random.choice(all_words)
                            code.add(random_word)
            
            # Local search: try to remove words
            improved = True
            while improved:
                improved = False
                for c in list(code):
                    new_code = code - {c}
                    new_coverage = get_coverage(new_code)
                    if new_coverage == total_words:
                        code = new_code
                        improved = True
                        break
            
            # Try swapping
            improved = True
            while improved:
                improved = False
                code_list = list(code)
                for c_remove in code_list:
                    for c_add in all_words:
                        if c_add in code:
                            continue
                        new_code = (code - {c_remove}) | {c_add}
                        new_coverage = get_coverage(new_code)
                        if new_coverage == total_words:
                            code = new_code
                            improved = True
                            break
                    if improved:
                        break
            
            # Update best
            if len(code) < best_size:
                best_size = len(code)
                best_code = set(code)
        
        # Convert back to list of lists
        result = [list(c) for c in best_code]
        return result

    # Generic fallback for other parameters
    # Simple greedy algorithm
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        nb = set()
        # Generate all words within Hamming distance 'radius'
        # For small n and radius, we can do this by changing up to 'radius' positions
        # For efficiency, since n and q are small, we can just iterate all words and check distance
        # But 3^6 is small. For larger n, this would be slow.
        # Let's assume n is small enough for this to work within time limits.
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors[w] = nb
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        # Find the word that covers the most uncovered words
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

# EVOLVE-BLOCK-END
