"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(n):
    if n == 1:
        return "1"
    
    def standard_superperm(k):
        """Standard recursive superpermutation construction."""
        if k == 1:
            return [1]
        
        prev = standard_superperm(k - 1)
        
        # Extract all (k-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(prev) - (k - 2)):
            window = tuple(prev[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word
        new_word = []
        for p in perms:
            block = list(p) + [k] + list(p)
            max_overlap = min(len(new_word), len(block))
            best_overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    best_overlap = o
                    break
            new_word.extend(block[best_overlap:])
        
        return new_word
    
    # For n <= 5, standard construction is optimal
    if n <= 5:
        word = standard_superperm(n)
        return "".join(map(str, word))
    
    # For n >= 6, try to optimize
    # Build up to n-1 using standard
    base = standard_superperm(n - 1)
    
    # Extract all (n-1)-permutations
    perms = []
    seen = set()
    for i in range(len(base) - (n - 2)):
        window = tuple(base[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Standard construction for the last layer
    def build_last_layer(perms, new_symbol):
        new_word = []
        for p in perms:
            block = list(p) + [new_symbol] + list(p)
            max_overlap = min(len(new_word), len(block))
            best_overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    best_overlap = o
                    break
            new_word.extend(block[best_overlap:])
        return new_word
    
    word = build_last_layer(perms, n)
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
