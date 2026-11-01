def construct(n):
    # Hardcoded optimal/standard for small n
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123121321412314123214121431214231214321"
    if n == 5:
        # Standard construction for n=5 is length 153.
        # We can compute it or hardcode. Computing is fast.
        pass

    # Use list of ints for efficiency
    word = [1]
    
    # If n is small, we can just run the loop
    # But for n=5,6,7 we run the loop from current state.
    # To be safe and fast, we start from the hardcoded n=4 result if n>=5?
    # Or just run from n=1. It's fast enough.
    
    # Let's restart from n=1 for simplicity, but use lists.
    word = [1]
    
    for m in range(2, n + 1):
        # Extract unique (m-1) permutations in order of appearance
        perms = []
        seen = set()
        
        # Optimization: Convert word to string for slicing? 
        # No, list slicing is fine.
        # word is a list of ints.
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            # Since we build it recursively, it should be, but let's verify uniqueness.
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            # Block is P + [m] + P
            block = list(p) + [m] + list(p)
            
            # Find max overlap
            # We want to append block to new_word
            # Check suffix of new_word vs prefix of block
            max_overlap = 0
            limit = min(len(new_word), len(block))
            
            # Optimization: 
            # The overlap must be a suffix of new_word and prefix of block.
            # Since block starts with p (a permutation of 1..m-1), 
            # and new_word ends with a previous block's suffix,
            # we can limit the search.
            # Actually, just checking from limit down to 0 is fine for small m.
            
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word += block[max_overlap:]
        
        word = new_word
        
    return "".join(map(str, word))
