"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code for {0,1,2}^n with covering radius 1.
    Starts from the known 486-word direct-sum construction and applies
    a local-search heuristic to reduce the number of codewords.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: use direct sum if possible, else brute force small
        if n <= 4:
            # For small n, we can try to find a perfect code or use all words
            # For ternary, perfect codes exist for n=1,2,4 (tetracode)
            # For n=1: 3 words
            # For n=2: 9 words (perfect)
            # For n=4: 9 words (tetracode, perfect)
            # For other n, use direct sum if n>=4
            if n == 1:
                return [[i] for i in range(q)]
            elif n == 2:
                return [list(w) for w in itertools.product(range(q), repeat=2)]
            elif n == 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return tetracode
            else:
                # General fallback: direct sum
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Initialize with the 486-word direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # Convert to a set for fast lookup
    codeword_set = set(codewords)
    
    # Precompute neighbors for each codeword (the codeword itself + all words at distance 1)
    # A word w is covered if it is within Hamming distance 1 of some codeword
    # For each codeword c, the covered words are: c itself, and all words differing in exactly one position
    
    def get_covered_words(c):
        """Return the set of words covered by codeword c (distance <= 1)."""
        covered = {c}
        for i in range(n):
            for val in range(q):
                if val != c[i]:
                    new_word = list(c)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    # Build coverage counts: for each word in the space, count how many codewords cover it
    # Space is 3^8 = 6561 words, which is manageable
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    coverage_count = defaultdict(int)
    word_to_covering_codewords = defaultdict(list)
    
    for c in codewords:
        covered = get_covered_words(c)
        for w in covered:
            coverage_count[w] += 1
            word_to_covering_codewords[w].append(c)
    
    # Identify critical words: words covered by exactly one codeword
    critical_words = [w for w in all_words if coverage_count[w] == 1]
    
    # Local search: try to remove codewords that are redundant
    # A codeword can be removed if all words it covers are still covered by other codewords
    # i.e., if for all w in covered(c), coverage_count[w] > 1
    
    def can_remove(c, codeword_set, coverage_count, word_to_covering_codewords):
        """Check if codeword c can be removed without uncovering any word."""
        covered = get_covered_words(c)
        for w in covered:
            if coverage_count[w] <= 1:
                return False
        return True
    
    # Try to remove codewords iteratively
    removed_count = 0
    max_iterations = 10000
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        found_removal = False
        
        # Shuffle codewords to avoid getting stuck in local minima
        shuffled_cw = list(codeword_set)
        random.shuffle(shuffled_cw)
        
        for c in shuffled_cw:
            if c not in codeword_set:
                continue
            
            if can_remove(c, codeword_set, coverage_count, word_to_covering_codewords):
                # Remove this codeword
                covered = get_covered_words(c)
                for w in covered:
                    coverage_count[w] -= 1
                    word_to_covering_codewords[w].remove(c)
                
                codeword_set.remove(c)
                removed_count += 1
                found_removal = True
                break
        
        if not found_removal:
            break
    
    # Now try swaps: replace a codeword with another that covers the same critical words
    # and potentially reduces the total coverage redundancy
    
    # Re-identify critical words after removals
    critical_words = [w for w in all_words if coverage_count[w] == 1]
    
    # Try to improve by swapping codewords
    # For each codeword, try to find an alternative codeword that:
    # 1. Covers all critical words that the original codeword covers
    # 2. Results in a net reduction in the number of codewords or same number but better coverage
    
    # Actually, let's try a different approach: try to find codewords that can be replaced
    # by a single codeword that covers more words, allowing removal of other codewords
    
    # Simpler approach: try random perturbations
    best_codewords = list(codeword_set)
    best_size = len(best_codewords)
    
    # Try to add a random codeword and see if we can then remove more
    for attempt in range(200):
        # Pick a random word not in the current codeword set
        random_word = tuple(random.choice(range(q)) for _ in range(n))
        if random_word in codeword_set:
            continue
        
        # Temporarily add it
        codeword_set.add(random_word)
        covered_new = get_covered_words(random_word)
        
        temp_coverage = dict(coverage_count)
        temp_word_to_cov = {w: list(covs) for w, covs in word_to_covering_codewords.items()}
        
        for w in covered_new:
            temp_coverage[w] = temp_coverage.get(w, 0) + 1
            if w not in temp_word_to_cov:
                temp_word_to_cov[w] = []
            temp_word_to_cov[w].append(random_word)
        
        # Try to remove codewords
        temp_removed = 0
        temp_cw_set = set(codeword_set)
        
        shuffled_cw = list(temp_cw_set)
        random.shuffle(shuffled_cw)
        
        for c in shuffled_cw:
            if c not in temp_cw_set:
                continue
            
            # Check if c can be removed with the temporary coverage
            covered_c = get_covered_words(c)
            removable = True
            for w in covered_c:
                if temp_coverage.get(w, 0) <= 1:
                    removable = False
                    break
            
            if removable:
                for w in covered_c:
                    temp_coverage[w] -= 1
                    if w in temp_word_to_cov:
                        temp_word_to_cov[w].remove(c)
                
                temp_cw_set.remove(c)
                temp_removed += 1
        
        new_size = len(temp_cw_set)
        
        if new_size < best_size:
            # Accept this change
            # Update state
            codeword_set = temp_cw_set
            coverage_count = temp_coverage
            word_to_covering_codewords = temp_word_to_cov
            best_size = new_size
            best_codewords = list(codeword_set)
        
        # Restore original state
        codeword_set.discard(random_word)
        for w in covered_new:
            coverage_count[w] -= 1
            if w in word_to_covering_codewords:
                if random_word in word_to_covering_codewords[w]:
                    word_to_covering_codewords[w].remove(random_word)
    
    # Final verification: ensure all words are covered
    final_codewords = list(codeword_set)
    
    # Verify coverage
    covered_all = set()
    for c in final_codewords:
        covered_all.update(get_covered_words(c))
    
    if len(covered_all) < q ** n:
        # Fallback to the original 486-word construction
        return [list(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    return [list(c) for c in final_codewords]

# EVOLVE-BLOCK-END
