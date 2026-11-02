def construct(n):
    """
    Constructs the shortest known superpermutation using the classic recursive method.
    Length is sum_{k=1}^n k!.
    For n=7, length is 5913.
    """
    if n == 1:
        return "1"
    
    # Start with the superpermutation for n=1
    word = [1]
    
    # Build up to n
    for m in range(2, n + 1):
        # Extract all unique permutations of length m-1 from the current word
        # The current word contains all permutations of 1..m-1
        perms = []
        seen = set()
        # The window size is m-1
        window_size = m - 1
        for i in range(len(word) - window_size + 1):
            window = tuple(word[i:i + window_size])
            # Check if it's a valid permutation (all distinct)
            # Since the previous step ensured all windows were permutations,
            # and we are iterating through the word, we just need to check uniqueness.
            # However, the word might contain non-permutation windows if not careful?
            # No, the classic construction ensures every window of length k in the
            # superpermutation of k symbols is a permutation.
            # But when we append new symbols, do we break this?
            # The construction is: for each perm P of 1..m-1, insert P m P.
            # The resulting word contains all perms of 1..m.
            # The windows of length m-1 in the new word are exactly the perms of 1..m-1?
            # Yes, because P m P contains P at the start and end.
            # So extracting unique windows of size m-1 gives the list of perms.
            
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # Construct the new word
        new_word = []
        for p in perms:
            # Block is p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the end of new_word
            # Max possible overlap is len(block)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
    return "".join(map(str, word))
