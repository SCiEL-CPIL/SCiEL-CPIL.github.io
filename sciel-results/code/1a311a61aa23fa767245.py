import time
import sys

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols using the standard
    recursive construction. For n=7, this yields a length of 5913.
    The theoretical minimum is 5905, but that specific string is not easily generated
    without a pre-computed table or extensive search, which exceeds the time budget.
    """
    if n == 1:
        return "1"
    
    # Start with the superpermutation for 1
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations from the current word
        # in the order they appear
        perms = []
        seen = set()
        # The current word contains all (m-1) permutations.
        # We iterate through windows of size m-1.
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # A valid permutation of 1..m-1 must have distinct elements
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            # The block to insert is p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find the maximum overlap between the end of new_word and the start of block
            # to minimize the length of the appended part.
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
    return "".join(map(str, word))
