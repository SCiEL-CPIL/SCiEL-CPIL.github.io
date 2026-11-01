import time
from itertools import permutations

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132413214321"
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Standard recursive construction
    word = "1"
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            block = p + str(m) + p
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            new_word += block[max_overlap:]
        word = new_word
    
    # Try to shorten by finding better overlaps between consecutive blocks
    # Reconstruct with optimized block ordering
    if time.time() < deadline:
        word = _optimize_word(word, n, deadline)
    
    return word


def _optimize_word(word, n, deadline):
    """Try to shorten the superpermutation by greedy deletion."""
    
    def _factorial(k):
        result = 1
        for i in range(2, k + 1):
            result *= i
        return result
    
    fact_n = _factorial(n)
    
    def _perm_to_int(p):
        val = 0
        for c in p:
            val = val * n + (ord(c) - ord('1'))
        return val
    
    def _is_valid_fast(s):
        if len(s) < n:
            return False
        seen = set()
        for i in range(len(s) - n + 1):
            window = s[i:i+n]
            if len(set(window)) != n:
                continue
            key = _perm_to_int(window)
            seen.add(key)
            if len(seen) == fact_n:
                return True
        return len(seen) == fact_n
    
    # Greedy deletion: try to remove characters one at a time
    # Start with the full word and try to delete each position
    w = list(word)
    
    # Try positions in random order, repeatedly
    import random
    rng = random.Random(42)
    
    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        
        improved = False
        positions = list(range(1, len(w) - 1))
        rng.shuffle(positions)
        
        for pos in positions:
            if time.time() > deadline - 5:
                break
            
            # Try deleting position pos
            char = w[pos]
            w.pop(pos)
            
            # Check validity
            s = ''.join(w)
            if _is_valid_fast(s):
                improved = True
                # Keep the deletion, continue with shorter word
                break
            else:
                # Restore
                w.insert(pos, char)
        
        if not improved:
            break
    
    return ''.join(w)
