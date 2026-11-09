"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code for {0,1,2}^n with covering radius 1.
    Uses a constraint satisfaction approach with critical word heuristic:
    1. Initialize with the 486-word direct sum construction
    2. Iteratively identify critical words (covered by exactly one codeword)
    3. Attempt to swap codewords to reduce total count while maintaining coverage
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n <= 4:
            if n == 1:
                return [[i] for i in range(q)]
            elif n == 2:
                return [list(w) for w in itertools.product(range(q), repeat=2)]
            elif n == 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return tetracode
            else:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Initialize with the 486-word direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            initial_codewords.add(tuple(h + list(tail)))
    
    codewords = set(initial_codewords)
    
    # Precompute all words in the space
    all_words = list(itertools.product(range(q), repeat=n))
    all_words_set = set(all_words)
    
    # Function to get covered words for a codeword (distance <= 1)
    def get_covered(c):
        """Return set of words covered by codeword c (Hamming distance <= 1)."""
        covered = {c}
        for i in range(n):
            for val in range(q):
                if val != c[i]:
                    new_word = list(c)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    # Build coverage structure
    # coverage_count[w] = number of codewords covering w
    # covering_set[w] = set of codewords covering w
    coverage_count = defaultdict(int)
    covering_set = defaultdict(set)
    
    for c in codewords:
        for w in get_covered(c):
            coverage_count[w] += 1
            covering_set[w].add(c)
    
    # Track time
    start_time = time.time()
    time_limit = 110  # Leave some margin for verification
    
    # Phase 1: Greedy removal - remove codewords that don't cover any critical words
    # A codeword is removable if all words it covers are covered by at least 2 codewords
    def can_remove(c, cov_count, cov_set):
        """Check if codeword c can be removed without uncovering any word."""
        for w in get_covered(c):
            if cov_count[w] <= 1:
                return False
        return True
    
    def remove_codeword(c, codeword_set, cov_count, cov_set):
        """Remove codeword c and update coverage structures."""
        for w in get_covered(c):
            cov_count[w] -= 1
            cov_set[w].discard(c)
        codeword_set.discard(c)
    
    # Phase 1: Iterative greedy removal
    max_rounds = 100
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        shuffled = list(codewords)
        random.shuffle(shuffled)
        
        found_removal = False
        for c in shuffled:
            if c not in codewords:
                continue
            if can_remove(c, coverage_count, covering_set):
                remove_codeword(c, codewords, coverage_count, covering_set)
                found_removal = True
        
        if not found_removal:
            break
    
    # Phase 2: Critical word swap heuristic
    # For each critical word, try to find an alternative codeword that covers it
    # and allows removal of the current covering codeword
    
    def try_swap_for_critical_word(w, codeword_set, cov_count, cov_set):
        """
        Try to swap the codeword covering critical word w with an alternative
        that allows removal of the original codeword.
        Returns True if a beneficial swap was made.
        """
        if w not in cov_set or len(cov_set[w]) == 0:
            return False
        
        # Get the codeword(s) covering this critical word
        current_cw = list(cov_set[w])[0]
        
        if current_cw not in codeword_set:
            return False
        
        # Find all words that are critical and covered by current_cw
        critical_covered = []
        for word in get_covered(current_cw):
            if cov_count[word] == 1:
                critical_covered.append(word)
        
        if not critical_covered:
            return False
        
        # Generate candidate replacement codewords
        # Candidates are words at distance 1 from any critical word that current_cw covers
        candidates = set()
        for cw_word in critical_covered:
            for i in range(n):
                for val in range(q):
                    new_word = list(cw_word)
                    new_word[i] = val
                    candidate = tuple(new_word)
                    if candidate not in codeword_set:
                        candidates.add(candidate)
        
        # Also add random candidates
        for _ in range(50):
            rand_word = tuple(random.choice(range(q)) for _ in range(n))
            if rand_word not in codeword_set:
                candidates.add(rand_word)
        
        # Try each candidate
        for candidate in list(candidates):
            if time.time() - start_time > time_limit:
                return False
            
            # Check if candidate covers all critical words that current_cw covers
            candidate_covered = get_covered(candidate)
            
            covers_all_critical = True
            for cw_word in critical_covered:
                if cw_word not in candidate_covered:
                    covers_all_critical = False
                    break
            
            if not covers_all_critical:
                continue
            
            # Simulate: add candidate, remove current_cw
            # Check if this is valid (no word becomes uncovered)
            
            # Words that would lose coverage: words covered by current_cw but not by candidate
            lost_words = set()
            for word in get_covered(current_cw):
                if word not in candidate_covered:
                    # Check if word is covered by other codewords
                    other_coverers = cov_set[word] - {current_cw}
                    if not other_coverers:
                        lost_words.add(word)
            
            if lost_words:
                continue
            
            # This swap is valid. Perform it.
            # Add candidate
            for word in candidate_covered:
                cov_count[word] += 1
                cov_set[word].add(candidate)
            
            # Remove current_cw
            for word in get_covered(current_cw):
                cov_count[word] -= 1
                cov_set[word].discard(current_cw)
            
            codeword_set.discard(current_cw)
            codeword_set.add(candidate)
            
            return True
        
        return False
    
    # Phase 2: Iteratively apply critical word swaps
    swap_rounds = 50
    for round_num in range(swap_rounds):
        if time.time() - start_time > time_limit:
            break
        
        # Identify all critical words
        critical_words = [w for w in all_words if coverage_count[w] == 1]
        
        if not critical_words:
            break
        
        random.shuffle(critical_words)
        
        found_swap = False
        for w in critical_words:
            if time.time() - start_time > time_limit:
                break
            if try_swap_for_critical_word(w, codewords, coverage_count, covering_set):
                found_swap = True
        
        if not found_swap:
            break
    
    # Phase 3: Final greedy removal pass
    for round_num in range(50):
        if time.time() - start_time > time_limit:
            break
        
        shuffled = list(codewords)
        random.shuffle(shuffled)
        
        found_removal = False
        for c in shuffled:
            if c not in codewords:
                continue
            if can_remove(c, coverage_count, covering_set):
                remove_codeword(c, codewords, coverage_count, covering_set)
                found_removal = True
        
        if not found_removal:
            break
    
    # Phase 4: Random perturbation - add random codewords and try to remove more
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Add a random codeword
        random_word = tuple(random.choice(range(q)) for _ in range(n))
        if random_word in codewords:
            continue
        
        # Save state
        old_codewords = set(codewords)
        old_cov_count = dict(coverage_count)
        old_cov_set = {w: set(s) for w, s in covering_set.items()}
        
        # Add random word
        codewords.add(random_word)
        for word in get_covered(random_word):
            coverage_count[word] += 1
            covering_set[word].add(random_word)
        
        # Try to remove codewords
        shuffled = list(codewords)
        random.shuffle(shuffled)
        
        for c in shuffled:
            if c not in codewords:
                continue
            if can_remove(c, coverage_count, covering_set):
                remove_codeword(c, codewords, coverage_count, covering_set)
        
        # Check if we improved
        if len(codewords) < len(old_codewords):
            # Keep the improvement
            pass
        else:
            # Revert
            codewords = old_codewords
            coverage_count = old_cov_count
            covering_set = old_cov_set
    
    # Final verification
    covered_all = set()
    for c in codewords:
        covered_all.update(get_covered(c))
    
    if len(covered_all) < q ** n:
        # Fallback to the original 486-word construction
        return [list(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    return [list(c) for c in codewords]

# EVOLVE-BLOCK-END
