"""Football pool problem: ternary covering code K_3(8,1).
Goal: minimize number of codewords of length 8 over {0,1,2} such that every word
is within Hamming distance 1 of some codeword.
"""

import time
import random
import numpy as np
from collections import defaultdict

# Deadline guard
_DEADLINE = 150.0  # seconds wall-clock limit; use 140s for safety margin

def _time_remaining():
    """Return seconds remaining before the deadline."""
    # We'll set a global start time
    if not hasattr(_time_remaining, '_start'):
        _time_remaining._start = time.time()
    return _DEADLINE - (time.time() - _time_remaining._start)


def _encode_word(word):
    """Encode a word (list of ints 0..q-1) as a single integer index."""
    idx = 0
    for i, v in enumerate(word):
        idx = idx * 3 + v
    return idx


def _decode_word(idx, n=8):
    """Decode an integer index back to a word of length n over {0,1,2}."""
    word = []
    for _ in range(n):
        word.append(idx % 3)
        idx //= 3
    word.reverse()
    return word


def _neighbors(word, q=3, n=8):
    """Generate all words within Hamming distance 1 of the given word.
    Includes the word itself and all words differing in exactly one position.
    """
    neighbors = [tuple(word)]
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                new_word = list(word)
                new_word[i] = v
                neighbors.append(tuple(new_word))
    return neighbors


def _build_neighbor_table(q=3, n=8):
    """Precompute neighbors for all words. Returns dict: word_tuple -> list of neighbor tuples."""
    # Total words: q^n = 3^8 = 6561
    # Each word has 1 + n*(q-1) = 1 + 8*2 = 17 neighbors
    # Total edges: 6561 * 17 = 111537
    table = {}
    for idx in range(q ** n):
        word = tuple(_decode_word(idx, n))
        table[word] = _neighbors(word, q, n)
    return table


def _greedy_cover(q=3, n=8, radius=1):
    """Fast greedy heuristic to find a feasible covering.
    Repeatedly pick the word that covers the most uncovered words.
    """
    all_words = [tuple(_decode_word(i, n)) for i in range(q ** n)]
    neighbor_table = _build_neighbor_table(q, n)
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        # Pick the word that covers the most uncovered words
        best_word = None
        best_count = -1
        for w in list(uncovered):
            count = 0
            for nb in neighbor_table[w]:
                if nb in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
            if count == len(uncovered):
                break
        if best_word is None:
            break
        code.append(list(best_word))
        # Remove all covered words
        for nb in neighbor_table[best_word]:
            uncovered.discard(nb)
    
    return code


def _compute_coverage(code, q=3, n=8, radius=1):
    """Compute which words are covered by the given code.
    Returns a set of covered word tuples.
    """
    neighbor_table = _build_neighbor_table(q, n)
    covered = set()
    for w in code:
        for nb in neighbor_table[tuple(w)]:
            covered.add(nb)
    return covered


def _local_search(code, q=3, n=8, radius=1, time_limit=100.0):
    """Local search to reduce the number of codewords.
    Strategy: try removing each codeword; if coverage is still complete, keep it removed.
    Also try swapping one codeword for another to enable further removals.
    """
    start_time = time.time()
    all_words = [tuple(_decode_word(i, n)) for i in range(q ** n)]
    neighbor_table = _build_neighbor_table(q, n)
    
    # Convert to list of tuples for efficiency
    code = [tuple(w) for w in code]
    code_set = set(code)
    
    # Build coverage count: for each word, how many codewords cover it
    cover_count = defaultdict(int)
    for w in code:
        for nb in neighbor_table[w]:
            cover_count[nb] += 1
    
    # Check if fully covered
    def is_fully_covered():
        for w in all_words:
            if cover_count[w] == 0:
                return False
        return True
    
    if not is_fully_covered():
        return [list(w) for w in code]
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try removing each codeword
        for i in range(len(code)):
            if (time.time() - start_time) >= time_limit:
                break
            w = code[i]
            # Tentatively remove
            for nb in neighbor_table[w]:
                cover_count[nb] -= 1
            
            if is_fully_covered():
                # Keep it removed
                code.pop(i)
                code_set.discard(w)
                improved = True
                break
            else:
                # Restore
                for nb in neighbor_table[w]:
                    cover_count[nb] += 1
    
    # Now try swaps: replace one codeword with another to potentially enable more removals
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try all pairs (i, candidate) where candidate is not in code
        # To limit search, only try a subset
        candidates = random.sample(all_words, min(500, len(all_words)))
        
        for i in range(len(code)):
            if (time.time() - start_time) >= time_limit:
                break
            old_w = code[i]
            for new_w in candidates:
                if new_w in code_set:
                    continue
                
                # Tentatively swap
                # Remove old
                for nb in neighbor_table[old_w]:
                    cover_count[nb] -= 1
                # Add new
                for nb in neighbor_table[new_w]:
                    cover_count[nb] += 1
                
                # Check if this enables any removals
                removals_possible = 0
                removable_indices = []
                for j in range(len(code)):
                    if j == i:
                        continue
                    w_j = code[j]
                    # Check if removing w_j still leaves full coverage
                    for nb in neighbor_table[w_j]:
                        cover_count[nb] -= 1
                    removable = True
                    for w in all_words:
                        if cover_count[w] == 0:
                            removable = False
                            break
                    if removable:
                        removals_possible += 1
                        removable_indices.append(j)
                    else:
                        for nb in neighbor_table[w_j]:
                            cover_count[nb] += 1
                
                if removals_possible > 0:
                    # Accept the swap and do the removals
                    code[i] = new_w
                    code_set.discard(old_w)
                    code_set.add(new_w)
                    # Do the removals
                    for j in sorted(removable_indices, reverse=True):
                        w_j = code[j]
                        for nb in neighbor_table[w_j]:
                            cover_count[nb] -= 1
                        code.pop(j)
                        code_set.discard(w_j)
                    improved = True
                    break
                else:
                    # Restore
                    for nb in neighbor_table[old_w]:
                        cover_count[nb] += 1
                    for nb in neighbor_table[new_w]:
                        cover_count[nb] -= 1
            
            if improved:
                break
    
    return [list(w) for w in code]


def _random_restart_greedy(q=3, n=8, radius=1, num_restarts=5, time_per_restart=20.0):
    """Try multiple random restarts of greedy + local search, keep the best."""
    best_code = None
    best_size = float('inf')
    
    start_time = time.time()
    total_time = time_per_restart * num_restarts
    
    for r in range(num_restarts):
        if (time.time() - start_time) >= total_time - 5:
            break
        
        # Randomize the greedy order by shuffling
        random.seed(r * 12345 + 67890)
        code = _greedy_cover(q, n, radius)
        code = _local_search(code, q, n, radius, time_limit=time_per_restart - 5)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
    
    return best_code if best_code else _greedy_cover(q, n, radius)


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet of size q with covering radius radius.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    _time_remaining._start = time.time()
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: use direct product construction
        # For general case, use a simple greedy
        code = _greedy_cover(q, n, radius)
        return code
    
    # For the specific case q=3, n=8, radius=1:
    # Start with greedy + local search
    code = _greedy_cover(q, n, radius)
    code = _local_search(code, q, n, radius, time_limit=60.0)
    
    # Try to improve with more aggressive local search if time permits
    if _time_remaining() > 40:
        code = _local_search(code, q, n, radius, time_limit=_time_remaining() - 10)
    
    # Final check: ensure validity
    all_words = [tuple(_decode_word(i, n)) for i in range(q ** n)]
    neighbor_table = _build_neighbor_table(q, n)
    covered = set()
    for w in code:
        for nb in neighbor_table[tuple(w)]:
            covered.add(nb)
    
    if len(covered) != q ** n:
        # Fallback to known construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return code
