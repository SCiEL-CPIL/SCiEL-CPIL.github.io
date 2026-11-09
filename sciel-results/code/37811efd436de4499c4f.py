"""Program for ternary covering code K_3(6,1) using greedy set cover with efficient coverage counting."""
import itertools
import numpy as np

# Precompute all words in {0,1,2}^6 as a flat array for efficiency
# There are 3^6 = 729 words.
# Represent each word as an integer in base 3 for fast indexing.

def _word_to_int(w):
    """Convert a tuple/list of digits to a base-3 integer."""
    val = 0
    for d in w:
        val = val * 3 + d
    return val

def _int_to_word(val, n):
    """Convert a base-3 integer to a tuple of n digits."""
    digits = []
    for _ in range(n):
        digits.append(val % 3)
        val //= 3
    return tuple(reversed(digits))

# Precompute for n=6, q=3, radius=1:
# For each word, compute the set of words within Hamming distance 1.
# We'll use a coverage array approach.

def construct(q, n, radius):
    """
    Construct a covering code for {0..q-1}^n with covering radius.
    Uses a greedy set cover approach with efficient coverage tracking.
    """
    # For the specific case q=3, n=6, radius=1, we use a specialized algorithm.
    # For general cases, fall back to a simple construction.
    
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    
    # General fallback: use the tetracode-based construction if n >= 4, else brute force
    if n >= 4:
        # Tetracode for the first 4 coordinates
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # For remaining coordinates, use all words
        result = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        # Small n: just use all words (brute force)
        return [list(w) for w in itertools.product(range(3), repeat=n)]


def _construct_ternary_6_1():
    """
    Specialized construction for K_3(6,1).
    Uses greedy set cover with efficient coverage counting using numpy.
    """
    total_words = 3 ** 6  # 729
    radius = 1
    q = 3
    n = 6
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # For each word, precompute the set of words within distance 1 (including itself)
    # This is 1 + 6*2 = 13 words per codeword candidate
    # We'll represent coverage as a boolean array of size total_words
    
    # Precompute for each candidate word index, the list of indices it covers
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # For each word, compute the neighbors within Hamming distance 1
    neighbors = []
    for w in all_words:
        covered = [w]
        for pos in range(n):
            for d in range(q):
                if d == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = d
                new_w = tuple(new_w)
                covered.append(new_w)
        neighbors.append(covered)
    
    # Convert neighbors to index lists
    neighbor_indices = []
    for covered in neighbors:
        idx_list = [word_to_idx[w] for w in covered]
        neighbor_indices.append(idx_list)
    
    # Greedy set cover: repeatedly pick the word that covers the most uncovered words
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Precompute coverage counts for each candidate
    # coverage_count[i] = number of currently uncovered words that candidate i covers
    
    # Initial coverage counts
    coverage_counts = np.array([len(neighbor_indices[i]) for i in range(total_words)], dtype=np.int32)
    
    # We need to update coverage counts efficiently.
    # When we select codeword c, we mark its neighbors as covered.
    # For each other candidate j, we need to decrement coverage_counts[j] by the number
    # of newly covered words that j also covers.
    
    # This is expensive. Let's use a different approach:
    # At each step, find the candidate with maximum coverage of uncovered words.
    # To do this efficiently, we can maintain the coverage array and compute
    # the coverage of each candidate by summing over its neighbors.
    
    # Given the small size (729 words), let's just do a direct computation at each step.
    # Number of steps is at most 73, and each step involves checking 729 candidates,
    # each with 13 neighbors. That's 729 * 13 = 9477 operations per step, times 73 steps = ~690k operations.
    # This is very fast.
    
    while not covered.all():
        best_idx = -1
        best_count = -1
        for i in range(total_words):
            count = 0
            for j in neighbor_indices[i]:
                if not covered[j]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
            if best_count == len(neighbor_indices[i]):
                # Can't do better than this
                break
        
        if best_idx == -1:
            break
        
        codewords.append(list(all_words[best_idx]))
        for j in neighbor_indices[best_idx]:
            covered[j] = True
    
    return codewords
