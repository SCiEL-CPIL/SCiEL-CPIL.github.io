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
    """Try to shorten the superpermutation by finding better overlaps."""
    # For now, return the standard construction
    # A more sophisticated approach would try different block orderings
    return word
