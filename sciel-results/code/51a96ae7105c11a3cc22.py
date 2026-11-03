"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(n):
    # Classic recursive construction of length 1! + 2! + ... + n!
    # This is the standard "interleaving" construction.
    # For n=7, this yields a length of 5913.
    # It is valid and fast.
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique permutations of length m-1 that appear in the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation (all distinct)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Construct the block: p + [m] + p
            block = list(p) + [m] + list(p)
            # Find the maximum overlap between the end of 'new' and the start of 'block'
            # We want the largest k such that new[-k:] == block[:k]
            overlap = 0
            max_overlap = min(len(new), len(block))
            for k in range(max_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new[len(new) - k:] == block[:k]:
                    overlap = k
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
