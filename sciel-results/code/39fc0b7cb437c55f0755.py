"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

# Use a fixed seed for reproducibility within a run, but allow variation
random.seed(42)


def _neighbors(word, q, n):
    """Yield all words at Hamming distance 1 from word."""
    for pos in range(n):
        for val in range(q):
            if val != word[pos]:
                yield pos, val


def _all_neighbors_set(word, q, n):
    """Return set of tuples (pos, val) for neighbors."""
    return set((pos, val) for pos in range(n) for val in range(q) if val != word[pos])


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Total space size
    total = q ** n

    # Convert word to integer index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbor index offsets for each position and value change
    # For position pos, changing from old_val to new_val changes index by (new_val - old_val) * q^(n-1-pos)
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    # For each position and each pair (old_val, new_val), the index delta
    # idx_delta[pos][old_val][new_val]
    idx_deltas = [[[0] * q for _ in range(q)] for _ in range(n)]
    for pos in range(n):
        pow_val = q_powers[n - 1 - pos]
        for old_val in range(q):
            for new_val in range(q):
                if old_val != new_val:
                    idx_deltas[pos][old_val][new_val] = (new_val - old_val) * pow_val

    # Greedy construction: start with empty code, repeatedly add codeword that covers most uncovered words
    # We'll track coverage using a boolean array
    covered = bytearray(total)  # 1 if covered, 0 if not
    uncovered_count = total

    # For efficiency, we'll use a list of uncovered words and a mapping from word to its neighbors
    # Actually, let's think differently. We want to find codewords such that every word is within radius 1.
    
    # Strategy: Use a greedy set cover approach with randomization and local search
    
    # First, let's build an initial code using a structured approach
    # The tetracode direct sum gives 9 * 3^(n-4) = 9 * 81 = 729 for n=8
    # We want to do better.
    
    # Let's use a greedy approach:
    # 1. Start with all words uncovered
    # 2. Repeatedly pick the word that covers the most uncovered words
    # 3. Mark all words within radius 1 as covered
    # 4. Continue until all are covered
    
    # But this is O(total * total) which is too slow for 6561^2.
    
    # Better approach: Use the structure. Each codeword covers 1 + n*(q-1) = 1 + 8*2 = 17 words.
    # Minimum number of codewords is at least ceil(6561/17) = 387.
    
    # Let's use a different strategy:
    # 1. Start with a random or structured initial code
    # 2. Use local search to remove redundant codewords and add new ones
    
    # For now, let's use the tetracode as a starting point and try to improve
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to set of indices for fast lookup
    code_indices = set(word_to_idx(w) for w in initial_code)
    code_list = list(initial_code)
    
    # Build coverage: for each word, which codeword(s) cover it?
    # covered_by[word_idx] = set of codeword indices that cover it
    
    # Actually, let's track: for each codeword, the set of words it covers
    # And for each word, the count of codewords that cover it
    
    # Let's use a different approach: 
    # covered_count[word_idx] = number of codewords within radius 1 of word_idx
    
    # Build initial coverage
    covered_count = bytearray(total)  # Use bytearray for memory efficiency, but we need counts > 1
    
    # Actually, let's use a list of ints
    covered_count = [0] * total
    
    # For each codeword, increment covered_count for all words within radius 1
    for cw in code_list:
        cw_idx = word_to_idx(cw)
        covered_count[cw_idx] += 1
        for pos in range(n):
            for val in range(q):
                if val != cw[pos]:
                    # Compute neighbor index
                    nb_idx = cw_idx + idx_deltas[pos][cw[pos]][val]
                    covered_count[nb_idx] += 1
    
    # Now, try to remove codewords that are redundant (all words they cover are covered by others)
    # A codeword is redundant if for every word it covers, covered_count[word] > 1
    
    # Let's iterate: try to remove each codeword, check if still valid
    # This is O(|code| * 17) per check, which is fine for ~700 codewords
    
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 30:
                break
            cw = code_list[i]
            cw_idx = word_to_idx(cw)
            
            # Check if removing this codeword would leave any word uncovered
            # i.e., if there's any word w where covered_count[w] == 1 and w is covered by cw
            redundant = True
            # Check the codeword itself
            if covered_count[cw_idx] == 1:
                redundant = False
            else:
                # Check all neighbors
                for pos in range(n):
                    if not redundant:
                        break
                    for val in range(q):
                        if val != cw[pos]:
                            nb_idx = cw_idx + idx_deltas[pos][cw[pos]][val]
                            if covered_count[nb_idx] == 1:
                                redundant = False
                                break
            
            if redundant:
                # Remove this codeword
                # Decrement covered_count for all words it covers
                covered_count[cw_idx] -= 1
                for pos in range(n):
                    for val in range(q):
                        if val != cw[pos]:
                            nb_idx = cw_idx + idx_deltas[pos][cw[pos]][val]
                            covered_count[nb_idx] -= 1
                code_list.pop(i)
                improved = True
    
    # Now try to add codewords to allow further removals (2-opt style)
    # Or just return what we have
    
    return code_list
# EVOLVE-BLOCK-END
