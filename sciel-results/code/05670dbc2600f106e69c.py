"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(q, n, radius):
    deadline = time.time() + 110.0  # safety margin before 120s kill
    start_time = time.time()
    
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Convert to tuples for hashing
    cw_tuples = [tuple(cw) for cw in codewords]
    
    # Build coverage: for each codeword, compute all words within Hamming distance <= radius
    # For radius=1, that's 1 + n*(q-1) words
    # We'll track coverage counts and which codewords cover each word
    
    # For efficiency, use a dictionary mapping word -> list of codeword indices that cover it
    # But with 729 words and 81 codewords, this is manageable
    
    # Build coverage map
    coverage_map = defaultdict(list)
    
    def get_neighbors(word, q, n, radius):
        """Get all words within Hamming distance <= radius from word."""
        if radius == 0:
            return [word]
        elif radius == 1:
            neighbors = [word]
            for pos in range(n):
                for val in range(q):
                    if val != word[pos]:
                        new_word = list(word)
                        new_word[pos] = val
                        neighbors.append(tuple(new_word))
            return neighbors
        else:
            # General case: use itertools.product for positions to change
            # For small radius, enumerate combinations of positions
            neighbors = [word]
            for r in range(1, radius + 1):
                for positions in itertools.combinations(range(n), r):
                    for changes in itertools.product(range(q), repeat=r):
                        if any(ch == word[p] for ch, p in zip(changes, positions)):
                            continue
                        new_word = list(word)
                        for pos, val in zip(positions, changes):
                            new_word[pos] = val
                        neighbors.append(tuple(new_word))
            return neighbors
    
    # Build coverage map: word -> set of codeword indices
    for i, cw in enumerate(cw_tuples):
        for neighbor in get_neighbors(cw, q, n, radius):
            coverage_map[neighbor].append(i)
    
    # Now greedily remove redundant codewords
    # A codeword is redundant if every word it covers is also covered by another codeword
    # We can remove it if removing it doesn't leave any word uncovered
    
    # Count how many times each word is covered
    word_coverage_count = {w: len(indices) for w, indices in coverage_map.items()}
    
    # Track which codewords are still active
    active = [True] * len(cw_tuples)
    
    # Greedy removal: repeatedly find a removable codeword and remove it
    # A codeword i is removable if for every word w covered by i, 
    # there exists another active codeword j != i that also covers w
    
    removed_count = 0
    while time.time() < deadline:
        # Find a removable codeword
        removable_idx = -1
        for i in range(len(cw_tuples)):
            if not active[i]:
                continue
            # Check if codeword i is removable
            neighbors_i = get_neighbors(cw_tuples[i], q, n, radius)
            is_removable = True
            for w in neighbors_i:
                # Check if w is covered by another active codeword
                covered_by_other = False
                for j in coverage_map[w]:
                    if j != i and active[j]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    is_removable = False
                    break
            if is_removable:
                removable_idx = i
                break
        
        if removable_idx == -1:
            break  # No more removable codewords
        
        # Remove this codeword
        active[removable_idx] = False
        removed_count += 1
    
    # Build final list of codewords
    result = [list(cw_tuples[i]) for i in range(len(cw_tuples)) if active[i]]
    
    return result
# EVOLVE-BLOCK-END
