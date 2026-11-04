"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode: 9 * q^(n-4) codewords."""
    if q != 3:
        # Fallback: all-zero words (valid but large)
        return [[0] * n]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _neighbors_indices(word, q, n, radius):
    """Return the set of indices of all words within Hamming distance <= radius from word."""
    covered = set()
    # Generate all words within radius by choosing positions to change
    # For radius=1: the word itself + n*(q-1) neighbors
    if radius == 0:
        idx = 0
        for d in word:
            idx = idx * q + d
        covered.add(idx)
    elif radius == 1:
        # The word itself
        idx = 0
        for d in word:
            idx = idx * q + d
        covered.add(idx)
        # All single-position changes
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                new_word = list(word)
                new_word[pos] = val
                idx2 = 0
                for d in new_word:
                    idx2 = idx2 * q + d
                covered.add(idx2)
    else:
        # General radius: enumerate all words within radius
        # For small n and radius, enumerate all subsets of positions
        from itertools import combinations, product
        for r in range(radius + 1):
            for positions in combinations(range(n), r):
                if r == 0:
                    idx = 0
                    for d in word:
                        idx = idx * q + d
                    covered.add(idx)
                else:
                    for changes in product(range(q), repeat=r):
                        new_word = list(word)
                        for i, pos in enumerate(positions):
                            if changes[i] != word[pos]:
                                new_word[pos] = changes[i]
                            else:
                                # Skip if no actual change (would be counted in smaller r)
                                # Actually we need to allow it but it duplicates
                                pass
                        idx2 = 0
                        for d in new_word:
                            idx2 = idx2 * q + d
                        covered.add(idx2)
    return covered


def _greedy_prune(codewords, q, n, radius, deadline):
    """Greedily remove redundant codewords while maintaining coverage."""
    if not codewords:
        return codewords
    
    total_words = q ** n
    
    # Map each codeword to its index
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    
    # Precompute coverage sets for each codeword
    coverage = [_neighbors_indices(w, q, n, radius) for w in codewords]
    
    # Union of all covered words
    all_covered = set()
    for c in coverage:
        all_covered.update(c)
    
    # If not all words are covered, we can't prune
    if len(all_covered) < total_words:
        return codewords
    
    # Greedy: repeatedly remove the codeword whose removal doesn't break coverage
    # Start from the end to preserve order somewhat
    n_codes = len(codewords)
    active = list(range(n_codes))
    
    # For efficiency, track which words are covered by which active codewords
    # word_to_covers[word_idx] = set of active codeword indices that cover it
    
    word_to_covers = {}
    for i, c in enumerate(coverage):
        for w in c:
            if w not in word_to_covers:
                word_to_covers[w] = set()
            word_to_covers[w].add(i)
    
    removed = [False] * n_codes
    result = list(codewords)
    
    # Try to remove each codeword
    for i in range(n_codes - 1, -1, -1):
        if time.time() > deadline:
            break
        if removed[i]:
            continue
        
        # Check if removing codeword i still covers all words
        # A word is still covered if it's covered by some other active codeword
        can_remove = True
        for w in coverage[i]:
            covers = word_to_covers[w]
            # Count how many active codewords cover w
            other_count = sum(1 for j in covers if not removed[j] and j != i)
            if other_count == 0:
                can_remove = False
                break
        
        if can_remove:
            removed[i] = True
            for w in coverage[i]:
                word_to_covers[w].discard(i)
    
    return [result[i] for i in range(n_codes) if not removed[i]]


def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover: start with no codewords, add the one covering most uncovered words."""
    total_words = q ** n
    
    # Precompute all word indices
    all_words = list(range(total_words))
    
    # We'll work with the full space
    # For each candidate codeword, compute which words it covers
    # But there are q^n possible codewords, so we need a smarter approach
    
    # Start with a random or structured initial set, then improve
    # Use the tetracode solution as starting point
    initial = _tetracode_solution(q, n, radius)
    
    # Convert to indices
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    # Greedy construction from scratch
    covered = set()
    codewords = []
    
    # Candidate pool: use structured candidates
    # For efficiency, consider candidates from the tetracode family and random samples
    candidates = set()
    for w in initial:
        candidates.add(word_to_idx(w))
    
    # Add some random candidates
    for _ in range(min(1000, total_words)):
        candidates.add(random.randint(0, total_words - 1))
    
    candidate_list = list(candidates)
    random.shuffle(candidate_list)
    
    while len(covered) < total_words and time.time() < deadline:
        best_idx = -1
        best_gain = -1
        best_word = None
        
        for cand in candidate_list:
            if time.time() > deadline:
                break
            # Compute coverage of this candidate
            nbrs = _neighbors_indices(idx_to_word(cand), q, n, radius)
            gain = len(nbrs - covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = cand
                best_word = idx_to_word(cand)
        
        if best_gain <= 0:
            break
        
        covered.update(_neighbors_indices(best_word, q, n, radius))
        codewords.append(best_word)
        candidate_list.remove(best_idx)
    
    if len(covered) < total_words:
        # Fall back to tetracode
        return initial
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Get initial solution
    solution = _tetracode_solution(q, n, radius)
    
    if time.time() > deadline - 5.0:
        return solution
    
    # Try greedy pruning
    pruned = _greedy_prune(solution, q, n, radius, deadline - 10.0)
    if len(pruned) < len(solution):
        solution = pruned
    
    if time.time() > deadline - 5.0:
        return solution
    
    # Try greedy cover from scratch (only if we have time)
    if time.time() < deadline - 30.0:
        greedy = _greedy_cover(q, n, radius, deadline - 5.0)
        if len(greedy) < len(solution):
            solution = greedy
    
    return solution
# EVOLVE-BLOCK-END
