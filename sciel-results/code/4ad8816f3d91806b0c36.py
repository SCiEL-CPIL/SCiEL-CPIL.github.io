import time
import itertools

def _build_level(word, m):
    """
    Given a superpermutation for symbols 1..(m-1) stored in `word` (list of ints),
    build a superpermutation for symbols 1..m by inserting m.
    Returns the new word as a list of ints.
    """
    w_size = m - 1
    # Extract all distinct (m-1)-permutations in order of appearance
    perms, seen = [], set()
    for i in range(len(word) - w_size + 1):
        window = tuple(word[i:i + w_size])
        if len(set(window)) == w_size:
            if window not in seen:
                seen.add(window)
                perms.append(window)
    
    if not perms:
        # Should not happen for valid superpermutations
        return word + [m]
    
    def construct_with_order(perms_order):
        new_word = []
        for p in perms_order:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap between end of new_word and start of block
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        return new_word
    
    # Try multiple orderings and pick the shortest result
    candidates = []
    
    # Original order
    candidates.append(construct_with_order(perms))
    
    # Reverse order
    candidates.append(construct_with_order(list(reversed(perms))))
    
    # Sort by first element, then second, etc.
    candidates.append(construct_with_order(sorted(perms)))
    
    # Sort by last element descending
    candidates.append(construct_with_order(sorted(perms, key=lambda x: -x[-1])))
    
    # Pick the shortest
    best = min(candidates, key=len)
    return best


def construct(n):
    """
    Constructs a superpermutation for n symbols using the recursive construction
    with greedy ordering optimization at each level.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (150s hard kill, leave room)
    
    # Handle small n
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Recursive construction
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        word = _build_level(word, m)
    
    return "".join(map(str, word))
