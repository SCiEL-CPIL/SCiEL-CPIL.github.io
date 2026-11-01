import itertools
import time

def standard_construct(n):
    """
    Generates the standard superpermutation of length sum(k! for k in 1..n).
    Returns a list of integers.
    """
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all (m-1)-permutations from word in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation of 1..m-1
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Construct block: P m P
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with end of new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        word = new_word
    return word

def construct(n):
    # Handle small n where standard is optimal
    if n <= 6:
        return "".join(map(str, standard_construct(n)))
    
    # For n > 7, local search is too slow, return standard
    if n > 7:
        return "".join(map(str, standard_construct(n)))
    
    # For n == 7, use local search
    word = standard_construct(n)
    
    # Pre-compute valid permutation indices for fast validity check
    # Map each permutation of 1..n to an integer
    valid_indices = set()
    for p in itertools.permutations(range(1, n + 1)):
        # Encode permutation as integer
        # Use base 8 (since max symbol is 7)
        val = 0
        for x in p:
            val = val * 8 + x
        valid_indices.add(val)
    
    def encode_window(w):
        val = 0
        for x in w:
            val = val * 8 + x
        return val

    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        # Sliding window
        for i in range(len(w) - n + 1):
            seen.add(encode_window(w[i:i + n]))
        # Check if all valid permutations are present
        # Using issubset is efficient
        return valid_indices.issubset(seen)

    # Deadline guard
    start_time = time.time()
    deadline = 110  # 10s margin
    
    # Greedy deletion
    while time.time() - start_time < deadline:
        changed = False
        i = 0
        while i < len(word):
            # Try deleting word[i]
            # Optimization: skip if deletion would break validity immediately?
            # Hard to know without checking.
            
            # Create new word
            # Using list slicing is O(N), which is fine for N=6000
            new_word = word[:i] + word[i + 1:]
            
            if is_valid(new_word):
                word = new_word
                changed = True
                # Do not increment i, check the new element at position i
            else:
                i += 1
        
        if not changed:
            break

    return "".join(map(str, word))
