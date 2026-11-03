import time

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols using the classic recursive method.
    For n=7, this produces a word of length 5913.
    """
    start_time = time.time()
    deadline = 140.0  # 10s margin from 150s limit

    if n <= 1:
        return "1"

    word = [1]

    for m in range(2, n + 1):
        # Time check
        if time.time() > deadline:
            # If we run out of time, we return the current word.
            # Note: This is only valid if m == n. 
            # If m < n, the word is not a superpermutation of 1..n.
            # However, the loop is fast enough for n=7.
            # 1! + ... + 7! is small. The overlap check is the bottleneck.
            pass

        perms = []
        seen = set()
        
        # Extract unique permutations of length m-1 from the current word
        # The current word is a superpermutation of 1..m-1
        # We need to find the order in which the (m-1)-permutations appear
        # to guide the insertion of m
        
        # Optimization: Use a list for speed
        w_len = len(word)
        for i in range(w_len - m + 2):
            # Slice and tuple
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            # Since m is small, set is fine
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)

        new_word = []
        for p in perms:
            # Construct the block: P + [m] + P
            # e.g., if P=(1,2), m=3, block = 1,2,3,1,2
            block = list(p) + [m] + list(p)
            
            # Calculate overlap with new_word
            # We want the maximum o such that new_word[-o:] == block[:o]
            # o cannot exceed len(block) or len(new_word)
            max_o = min(len(new_word), len(block))
            overlap = 0
            
            # Check overlaps from largest to smallest
            # Since m is small, len(block) is 2*m-1 <= 13.
            # So we check at most 13 overlaps.
            for o in range(max_o, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
                    
            # Append the non-overlapping part of block
            new_word.extend(block[overlap:])
            
        word = new_word

    return "".join(map(str, word))
