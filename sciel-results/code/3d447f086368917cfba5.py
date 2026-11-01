"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(n):
    # Classic recursive construction of length 1! + 2! + ... + n!
    # This is the only guaranteed valid solution for n=7 within time limits
    # without specific pre-computed knowledge of the 5905 solution.
    if n == 1:
        return "1"
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique permutations of length m-1 from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Only consider windows that are permutations (all distinct)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # The block to insert is p, m, p
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of the current new word
            # The overlap must be a suffix of new and prefix of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
