import time
import itertools

def construct(n):
    """
    Constructs a superpermutation for n symbols.
    Uses the standard recursive construction which yields length sum_{k=1}^n k!.
    For n=7, this length is 5913.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Handle small n with hardcoded optimal solutions
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Standard construction yields 33, which is optimal
        pass

    # Recursive construction
    # Start with S_1
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Find all distinct permutations of size m-1 in the current word, in order of appearance
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            # Block is p + [m] + p
            # This allows for overlaps with the previous permutation
            block = list(p) + [m] + list(p)
            
            # Find the maximum overlap between the end of new_word and the start of block
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            # We can optimize this by only checking up to w_size - 1
            # because the new symbol m cannot be in the overlap
            # unless the previous word ended with m (which it doesn't, as m is new)
            # So overlap is determined by the prefix of p matching the suffix of new_word
            # Max possible overlap is w_size - 1 (if p is a shift of prev p)
            # But we check all just in case
            
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
    return "".join(map(str, word))
