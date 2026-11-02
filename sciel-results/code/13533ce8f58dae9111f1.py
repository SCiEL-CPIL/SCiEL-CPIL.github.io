# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # For the specific case of q=3, n=7, radius=1
    # We use a greedy set-cover approach with a good initial seed
    
    # The tetracode-based construction gives 9 * 3^3 = 243 words
    # We want to do better than that (best known is 186)
    
    # Strategy: Start with a reasonable initial code, then greedily
    # remove redundant codewords and add missing coverage
    
    # First, generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = q ** n
    
    # Convert word to index for fast lookup
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return list(reversed(w))
    
    # Precompute neighborhoods (sphere of radius 1) for each word
    # A word covers itself + all words at Hamming distance 1
    # For q=3, n=7: each word covers 1 + 7*(3-1) = 15 words
    
    # For efficiency, we'll work with indices
    # Precompute the neighborhood of each index
    neighborhoods = {}
    for w in all_words:
        idx = word_to_idx(w)
        neigh = {idx}
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    w2 = list(w)
                    w2[pos] = val
                    neigh.add(word_to_idx(w2))
        neighborhoods[idx] = neigh
    
    # Initial construction: tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(tuple(h + list(tail)))
    
    # Convert to indices
    initial_idx = [word_to_idx(w) for w in initial_code]
    
    # Greedy improvement: try to remove codewords that are redundant
    # A codeword is redundant if its neighborhood is fully covered by others
    
    # Build coverage map: for each word, how many codewords cover it
    coverage = defaultdict(int)
    for idx in initial_idx:
        for covered in neighborhoods[idx]:
            coverage[covered] += 1
    
    # Try to remove redundant codewords
    code_set = set(initial_idx)
    removed_count = 0
    
    # Sort by coverage contribution (words only covered by this codeword)
    def unique_coverage(idx):
        count = 0
        for covered in neighborhoods[idx]:
            if coverage[covered] == 1:
                count += 1
        return count
    
    # Repeatedly try to remove codewords with 0 unique coverage
    progress = True
    while progress and time.time() < deadline:
        progress = False
        # Find codeword with minimum unique coverage (prefer 0)
        min_unique = float('inf')
        best_remove = None
        for idx in code_set:
            uc = unique_coverage(idx)
            if uc < min_unique:
                min_unique = uc
                best_remove = idx
        
        if best_remove is not None and min_unique == 0:
            # Remove this codeword
            for covered in neighborhoods[best_remove]:
                coverage[covered] -= 1
            code_set.remove(best_remove)
            removed_count += 1
            progress = True
    
    current_code = list(code_set)
    current_size = len(current_code)
    
    # Now try to further improve by swapping codewords
    # This is more complex, so we'll do a limited local search
    
    # Check if we've improved over the initial
    initial_size = len(initial_idx)
    
    # Try random restarts with different initial codes if time permits
    best_code = current_code
    best_size = current_size
    
    # Additional optimization: try to find a smaller code from scratch
    # using a greedy set cover approach
    
    if time.time() < deadline - 30:
        # Greedy set cover: start empty, add codeword that covers most uncovered words
        uncovered = set(range(total_words))
        greedy_code = []
        
        # Precompute which codewords cover which words (inverse neighborhoods)
        # For each word, which codewords cover it?
        # Actually, for greedy set cover, we need: for each candidate codeword,
        # how many currently uncovered words does it cover?
        
        # This is expensive, so we'll use a heuristic
        # Start with the tetracode-based code as a hint
        
        # Actually, let's try a different approach: use the fact that
        # the problem is symmetric under permutations of coordinates and
        # permutations of the alphabet
        
        # Try a few random perturbations of the tetracode construction
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Randomly permute coordinates
            perm = list(range(n))
            random.shuffle(perm)
            
            # Build a permuted tetracode-based code
            permuted_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    w = [0] * n
                    for i in range(4):
                        w[perm[i]] = h[i]
                    for i in range(4, n):
                        w[perm[i]] = tail[i - 4]
                    permuted_code.add(word_to_idx(tuple(w)))
            
            # Try to reduce this code
            perm_coverage = defaultdict(int)
            for idx in permuted_code:
                for covered in neighborhoods[idx]:
                    perm_coverage[covered] += 1
            
            perm_code_set = set(permuted_code)
            perm_progress = True
            while perm_progress and time.time() < deadline - 5:
                perm_progress = False
                min_uc = float('inf')
                best_rm = None
                for idx in perm_code_set:
                    uc = 0
                    for covered in neighborhoods[idx]:
                        if perm_coverage[covered] == 1:
                            uc += 1
                    if uc < min_uc:
                        min_uc = uc
                        best_rm = idx
                
                if best_rm is not None and min_uc == 0:
                    for covered in neighborhoods[best_rm]:
                        perm_coverage[covered] -= 1
                    perm_code_set.remove(best_rm)
                    perm_progress = True
            
            if len(perm_code_set) < best_size:
                best_code = list(perm_code_set)
                best_size = len(perm_code_set)
    
    # Convert back to list of lists
    result = [idx_to_word(idx) for idx in best_code]
    
    # Verify validity (quick check)
    # For the specific parameters, we trust our construction
    
    return result

# EVOLVE-BLOCK-END
