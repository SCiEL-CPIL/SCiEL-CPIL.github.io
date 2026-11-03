"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build standard recursive superpermutation
    def build_standard(m):
        if m == 1:
            return [1]
        if m == 2:
            return [1, 2, 1]
        
        prev = build_standard(m - 1)
        perms = []
        seen = set()
        for i in range(len(prev) - m + 2):
            window = tuple(prev[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        return new_word
    
    word = build_standard(n)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
