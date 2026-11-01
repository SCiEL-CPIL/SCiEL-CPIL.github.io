"""Superpermutation construction via standard recursive method with optimized overlaps."""
# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1"
    
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Standard recursive construction
    # S_n is built from S_{n-1}
    # For each permutation pi of 1..n-1 in S_{n-1}, we form a block pi n pi
    # We concatenate these blocks, using the maximum possible overlap
    # that preserves the validity of the superpermutation structure.
    
    # To ensure validity and optimal length for the recursive structure,
    # we extract the permutations in the order they appear in the previous superpermutation.
    
    word = [1]
    for m in range(2, n + 1):
        # Extract all unique permutations of length m-1 from the current word
        # in the order of their first appearance.
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for idx, p in enumerate(perms):
            block = list(p) + [m] + list(p)
            
            if idx == 0:
                new_word.extend(block)
            else:
                # Find the maximum overlap between the end of new_word and the start of block
                # The overlap must be at least m-2 to maintain the recursive structure's validity?
                # Actually, the standard construction relies on the fact that consecutive 
                # permutations in S_{m-1} overlap by m-2.
                # So the overlap between block_i and block_{i+1} should be at least m-2.
                
                max_overlap = min(len(new_word), len(block))
                found_overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        found_overlap = o
                        break
                
                # Heuristic: If the overlap is too small, we might break the structure.
                # However, for the standard recursive construction, the overlap is naturally large.
                # We use the maximal overlap found.
                new_word.extend(block[found_overlap:])
        
        word = new_word
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
