"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(n):
    """
    Constructs the shortest known superpermutation for n using the classic recursive method.
    Length is 1! + 2! + ... + n!.
    For n=7, this is 5913.
    """
    if n == 1:
        return "1"
    
    word = [1]
    for m in range(2, n + 1):
        # Extract permutations of size m-1 from the current word
        perms = []
        seen = set()
        # The current word contains all permutations of size m-1
        # We extract them in order of appearance
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation of 1..m-1
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # Construct the new word
        new_word = []
        for p in perms:
            # Block is p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find the maximum overlap with the end of new_word
            # The overlap must be at least 0
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
