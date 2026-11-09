"""Football pool problem for 8 matches: ternary covering code K_3(8,1).
Uses tetracode construction + deletion + guided local search."""
import itertools
import time
import random

def _tetracode():
    """The ternary tetracode: 9 codewords of length 4, perfect covering radius 1."""
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

def _hamming_dist(a, b):
    """Hamming distance between two tuples/lists."""
    return sum(1 for x, y in zip(a, b) if x != y)

def _build_initial_code(q, n, radius):
    """Direct sum of tetracode with all words on remaining coordinates."""
    tetracode = _tetracode()
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h + list(tail)))
    return list(set(code))

def _compute_covered(code, q, n):
    """Return a set of all words covered by the code (within Hamming distance <= 1)."""
    covered = set()
    for cw in code:
        covered.add(cw)
        for i in range(n):
            for v in range(q):
                if v != cw[i]:
                    word = list(cw)
                    word[i] = v
                    covered.add(tuple(word))
    return covered

def _deletion_pass(code, q, n, deadline):
    """Try to remove codewords that are still covered by others."""
    code = list(code)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        covered = _compute_covered(code, q, n)
        for idx in range(len(code)):
            if time.time() > deadline:
                break
            candidate = code[idx]
            # Check if removing candidate leaves all its covered words still covered
            # A codeword can be removed if every word within distance 1 of it
            # is also within distance 1 of some other codeword
            removable = True
            for i in range(n):
                if time.time() > deadline:
                    removable = False
                    break
                for v in range(q):
                    word = list(candidate)
                    word[i] = v
                    word = tuple(word)
                    # Check if word is covered by another codeword
                    found = False
                    for j, other in enumerate(code):
                        if j == idx:
                            continue
                        d = _hamming_dist(word, other)
                        if d <= 1:
                            found = True
                            break
                    if not found:
                        removable = False
                        break
                if not removable:
                    break
            if removable:
                code.pop(idx)
                improved = True
    return code

def _guided_local_search(code, q, n, radius, deadline, rng):
    """Guided local search: find critical words (uncovered) and use them to guide swaps."""
    code = list(code)
    best_code = list(code)
    best_len = len(code)
    
    while time.time() < deadline and len(code) > 403:
        # Find uncovered words (critical words)
        covered = _compute_covered(code, q, n)
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = [w for w in all_words if w not in covered]
        
        if not uncovered:
            # No uncovered words, try deletion
            code = _deletion_pass(code, q, n, deadline)
            if len(code) < best_len:
                best_len = len(code)
                best_code = list(code)
            continue
        
        # Pick a random uncovered word
        crit = rng.choice(uncovered)
        
        # Find codewords that are "close" to the critical word (distance 2)
        # and codewords that could be swapped
        close_cws = [cw for cw in code if _hamming_dist(cw, crit) == 2]
        
        if not close_cws:
            # No codeword at distance 2, try random swap
            if len(code) > 2:
                i = rng.randrange(len(code))
                j = rng.randrange(len(code) - 1)
                if j >= i:
                    j += 1
                # Swap: remove code[i], add a new word at distance 1 from crit
                new_word = list(crit)
                # Ensure it's different from crit
                pos = rng.randrange(n)
                new_word[pos] = (new_word[pos] + 1) % q
                new_word = tuple(new_word)
                if new_word not in code:
                    code[i] = new_word
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = list(code)
            continue
        
        # Try to replace a codeword with one that covers the critical word better
        # Strategy: pick a codeword at distance 2 from crit, replace it with a word
        # at distance 1 from crit that also covers other important words
        cw = rng.choice(close_cws)
        cw_idx = code.index(cw)
        
        # Generate candidate replacements: words at distance 1 from crit
        candidates = []
        for i in range(n):
            for v in range(q):
                if v != crit[i]:
                    word = list(crit)
                    word[i] = v
                    word = tuple(word)
                    if word not in code:
                        candidates.append(word)
        
        if not candidates:
            continue
        
        # Try each candidate and check if it maintains coverage
        best_candidate = None
        best_score = float('inf')
        
        for cand in candidates:
            if time.time() > deadline:
                break
            # Tentatively swap
            old_cw = code[cw_idx]
            code[cw_idx] = cand
            
            # Check if all words that old_cw covered are still covered
            ok = True
            for i in range(n):
                for v in range(q):
                    word = list(old_cw)
                    word[i] = v
                    word = tuple(word)
                    found = False
                    for other in code:
                        if _hamming_dist(word, other) <= 1:
                            found = True
                            break
                    if not found:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                # Count how many previously-uncovered words are now covered
                new_covered = 0
                for i in range(n):
                    for v in range(q):
                        word = list(cand)
                        word[i] = v
                        word = tuple(word)
                        if word not in covered:
                            new_covered += 1
                if new_covered < best_score:
                    best_score = new_covered
                    best_candidate = cand
            else:
                code[cw_idx] = old_cw
        
        if best_candidate is not None:
            code[cw_idx] = best_candidate
            if len(code) < best_len:
                best_len = len(code)
                best_code = list(code)
        else:
            # No good swap found, try a random perturbation
            if len(code) > 2:
                i = rng.randrange(len(code))
                new_word = list(crit)
                pos = rng.randrange(n)
                new_word[pos] = (new_word[pos] + 1) % q
                new_word = tuple(new_word)
                if new_word not in code:
                    code[i] = new_word
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = list(code)
    
    return best_code if len(best_code) < len(code) else code

def construct(q, n, radius):
    """Construct a covering code of length n over alphabet {0..q-1} with covering radius."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    rng = random.Random(42)
    
    # Build initial code using tetracode construction
    code = _build_initial_code(q, n, radius)
    
    # Deletion pass to remove redundant codewords
    code = _deletion_pass(code, q, n, deadline)
    
    # Guided local search to further reduce
    if time.time() < deadline:
        code = _guided_local_search(code, q, n, radius, deadline, rng)
    
    # Final deletion pass
    code = _deletion_pass(code, q, n, deadline)
    
    # Convert to list of lists
    return [list(cw) for cw in code]
