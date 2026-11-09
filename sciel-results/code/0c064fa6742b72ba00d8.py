import itertools
import time
import random
from collections import deque

def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0  # 10 seconds safety margin
    
    if n == 6 and q == 3 and radius == 1:
        # Use a specialized approach for the known case
        return _cover_ternary_6_1(start_time, deadline)
    
    # General fallback: greedy set cover
    return _greedy_cover(q, n, radius, start_time, deadline)

def _word_to_idx(word, q, n):
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx

def _idx_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word

def _neighbors(word, q, n, radius):
    """Generate all words within Hamming distance <= radius from word."""
    if radius == 0:
        yield tuple(word)
        return
    
    # Use BFS or direct generation for small radius
    for d in range(radius + 1):
        if d == 0:
            yield tuple(word)
        else:
            # Choose d positions to change
            for positions in itertools.combinations(range(n), d):
                # For each position, choose a different value
                for changes in itertools.product(range(q), repeat=d):
                    new_word = list(word)
                    valid = True
                    for pos, val in zip(positions, changes):
                        if val == word[pos]:
                            valid = False
                            break
                        new_word[pos] = val
                    if valid:
                        yield tuple(new_word)

def _greedy_cover(q, n, radius, start_time, deadline):
    """Greedy set cover: repeatedly pick the codeword that covers the most uncovered words."""
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # For each possible codeword, precompute which words it covers
    # This is expensive but necessary for greedy
    cover_sets = {}
    for cw in all_words:
        covered = set()
        for w in all_words:
            # Check if w is within radius of cw
            dist = sum(1 for a, b in zip(cw, w) if a != b)
            if dist <= radius:
                covered.add(w)
        cover_sets[cw] = covered
    
    # Greedy: pick codeword covering most uncovered
    uncovered = set(all_words)
    selected = []
    
    while uncovered and (time.time() - start_time) < deadline:
        best_cw = None
        best_count = -1
        for cw in all_words:
            if cw in selected:
                continue
            count = len(cover_sets[cw] & uncovered)
            if count > best_count:
                best_count = count
                best_cw = cw
            if best_count == len(uncovered):
                break
        
        if best_cw is None or best_count <= 0:
            break
        
        selected.append(best_cw)
        uncovered -= cover_sets[best_cw]
    
    # Convert to list of lists
    return [list(cw) for cw in selected]

def _cover_ternary_6_1(start_time, deadline):
    """Specialized covering code for ternary length 6 radius 1."""
    q = 3
    n = 6
    radius = 1
    
    # Start with the known 73-word construction and try to improve
    # Use a local search / greedy refinement approach
    
    # First, generate the base construction (direct sum of tetracode)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Try to find a smaller code using greedy + local search
    # Represent code as set of tuples for fast lookup
    
    def code_to_set(code):
        return set(tuple(cw) for cw in code)
    
    def is_covering(code_set):
        """Check if code_set covers all words within radius 1."""
        for w in itertools.product(range(3), repeat=6):
            w = tuple(w)
            # Check if any codeword is within distance 1
            found = False
            for cw in code_set:
                dist = sum(1 for a, b in zip(w, cw) if a != b)
                if dist <= 1:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Greedy construction from scratch
    # Start empty, add codewords that cover most uncovered words
    
    all_words = [tuple(w) for w in itertools.product(range(3), repeat=6)]
    
    # Precompute coverage for each potential codeword
    # cover[cw] = set of words covered by cw
    cover = {}
    for cw in all_words:
        covered = set()
        for w in all_words:
            dist = sum(1 for a, b in zip(cw, w) if a != b)
            if dist <= 1:
                covered.add(w)
        cover[cw] = covered
    
    # Greedy set cover
    uncovered = set(all_words)
    selected = []
    
    while uncovered and (time.time() - start_time) < deadline - 5:
        best_cw = None
        best_count = -1
        for cw in all_words:
            count = len(cover[cw] & uncovered)
            if count > best_count:
                best_count = count
                best_cw = cw
        
        if best_cw is None or best_count <= 0:
            break
        
        selected.append(best_cw)
        uncovered -= cover[best_cw]
    
    # Convert to list of lists
    result = [list(cw) for cw in selected]
    
    # Try local search: see if we can remove any codeword
    # A codeword can be removed if all words it uniquely covers are still covered by others
    improved = True
    while improved and (time.time() - start_time) < deadline - 2:
        improved = False
        current_code = code_to_set(result)
        for i, cw in enumerate(result):
            # Check if removing cw still covers everything
            test_code = current_code - {tuple(cw)}
            # Quick check: does test_code cover all words?
            # Instead of full check, check if any word was only covered by cw
            # For efficiency, just do full check since n=6 is small
            if is_covering(test_code):
                result.pop(i)
                improved = True
                break
    
    return result
