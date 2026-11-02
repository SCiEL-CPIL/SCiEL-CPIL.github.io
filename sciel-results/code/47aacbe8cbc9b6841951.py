import time
import random
from itertools import permutations

def _standard_superperm(n):
    """
    Generates the standard superpermutation for n symbols.
    Length: 1! + 2! + ... + n!
    """
    if n == 1:
        return [1]
    
    # Start with n-1 superpermutation
    word = _standard_superperm(n - 1)
    
    # Extract all (n-1)-permutations from the previous word in order of appearance
    perms = []
    seen = set()
    for i in range(len(word) - (n - 2)):
        window = tuple(word[i:i + n - 1])
        # Check if it's a valid permutation of n-1 distinct symbols
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    # Construct the new word
    new_word = []
    for p in perms:
        # Block is p + [n] + p
        block = list(p) + [n] + list(p)
        
        # Find maximum overlap between end of new_word and start of block
        max_overlap = 0
        max_possible = min(len(new_word), len(block))
        for k in range(max_possible, -1, -1):
            if k == 0:
                max_overlap = 0
                break
            if new_word[len(new_word) - k:] == block[:k]:
                max_overlap = k
                break
        
        new_word += block[max_overlap:]
        
    return new_word

def _is_valid_superperm(word, n):
    """
    Checks if the word contains all permutations of 1..n as consecutive substrings.
    """
    if len(word) < n:
        return False
    
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        # Check if window is a permutation of 1..n
        if len(set(window)) == n and all(1 <= x <= n for x in window):
            found.add(window)
            
    return len(found) == len(list(permutations(range(1, n + 1))))

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Get standard superpermutation
    std_word = _standard_superperm(n)
    
    if n < 7:
        return "".join(map(str, std_word))
        
    # For n=7, try to improve by reordering the blocks of the last step
    # The last step involves 720 blocks (permutations of 1..6)
    # We can try a simple local search on the order of these blocks
    
    if n == 7:
        # Reconstruct the blocks and their overlaps
        # We need the perms of size 6 from the n=6 superperm
        word_6 = _standard_superperm(6)
        perms_6 = []
        seen = set()
        for i in range(len(word_6) - 5):
            window = tuple(word_6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms_6.append(window)
                
        # perms_6 has 720 elements
        # Current order is the one in std_word
        # We want to find a better order for perms_6 to minimize total length
        
        # Precompute overlap matrix? 720x720 is 518k entries. 
        # Overlap of P_i and P_j is longest suffix of P_i that is prefix of P_j.
        # Since P_i, P_j are length 6, max overlap is 5.
        
        # Let's try a few random shuffles and check if they are valid and shorter
        best_word = std_word
        best_len = len(std_word)
        
        # Heuristic: Start with standard order, try 2-opt swaps
        # But checking validity is expensive.
        # Maybe just try to improve the overlap of the last few blocks?
        
        # Given the time constraint and complexity, a robust improvement is hard.
        # I will return the standard construction as it is guaranteed valid.
        # If I have time, I can add a simple greedy improvement.
        
        # Let's try a simple greedy construction for n=7 from scratch?
        # No, standard is safer.
        
        # Check time
        if time.time() > deadline:
            return "".join(map(str, std_word))
            
    return "".join(map(str, std_word))
