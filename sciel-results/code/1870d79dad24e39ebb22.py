def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Standard construction is optimal
        # 123412314231243121342132413214321
        pass
    
    # General recursive construction
    word = "1"
    for m in range(2, n + 1):
        # Extract unique (m-1) permutations in order of appearance
        perms = []
        seen = set()
        # word is string, convert to list for easier slicing? 
        # Or just use string slicing.
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            # Check if all chars are unique
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            # Block is P + str(m) + P
            block = p + str(m) + p
            
            # Find max overlap
            # We want to append block to new_word
            # Check suffix of new_word vs prefix of block
            max_overlap = 0
            # Theoretical max overlap is len(block)-1 or len(new_word)-1
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word += block[max_overlap:]
        
        word = new_word
        
    return word
