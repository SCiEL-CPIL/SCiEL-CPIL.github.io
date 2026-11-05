"""Shortest superpermutation via VNS on the recursive construction."""

def construct(n):
    # Classic recursive construction of length 1! + 2! + ... + n!: each permutation P of the
    # previous word, in order of appearance, becomes P m P, overlapped as far as possible.
    # We then apply a variable-neighborhood search (VNS) to improve the ordering of blocks.
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build the standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    # Now apply VNS to improve the ordering of (n-1)-permutation blocks
    # The word is constructed from blocks of form P m P where P is a permutation of 1..(n-1)
    # We need to extract these blocks and their order
    
    # First, let's reconstruct the block structure from the word
    # For the final word, the structure is: b_1, b_2, ..., b_k where each b_i is a block
    # Each block b_i corresponds to a permutation P_i of 1..(n-1), and b_i = P_i + [n] + P_i
    # The blocks are concatenated with maximum overlap
    
    # Let's extract the blocks by finding all occurrences of n in the word
    # Each occurrence of n is in the middle of a block P n P
    
    # Actually, let me think about this differently. The word is built from blocks.
    # Let's track the blocks during construction.
    
    # Rebuild with tracking
    word = [1]
    blocks_by_level = {}
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Store the perms for this level
        blocks_by_level[m] = perms
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    # The blocks at level n are the permutations of 1..(n-1)
    # Let's work with the list of permutations (blocks) at the top level
    # and try to reorder them to minimize the total length
    
    perms = blocks_by_level[n]
    k = len(perms)
    
    if k <= 1:
        return "".join(map(str, word))
    
    # Function to compute the total length given an ordering of permutations
    def compute_length(ordering):
        total = 0
        prev = None
        for idx in ordering:
            p = perms[idx]
            block = list(p) + [n] + list(p)
            if prev is None:
                total = len(block)
            else:
                prev_block = list(prev) + [n] + list(prev)
                # Find maximum overlap between prev_block suffix and block prefix
                max_overlap = min(len(prev_block), len(block))
                best = 0
                for o in range(max_overlap, -1, -1):
                    if prev_block[len(prev_block) - o:] == block[:o]:
                        best = o
                        break
                total += len(block) - best
            prev = p
        return total
    
    # Initial ordering
    current_ordering = list(range(k))
    current_length = compute_length(current_ordering)
    
    # VNS: try different neighborhood structures
    # N1: swap two adjacent blocks
    # N2: Or-opt (move a contiguous segment of blocks)
    # N3: 2-opt (reverse a contiguous segment of blocks)
    
    def try_swap(ordering, i, j):
        new_ordering = ordering[:]
        new_ordering[i], new_ordering[j] = new_ordering[j], new_ordering[i]
        return new_ordering
    
    def try_oropt(ordering, i, j, new_pos):
        """Move segment ordering[i:j+1] to position new_pos"""
        seg = ordering[i:j+1]
        rest = ordering[:i] + ordering[j+1:]
        if new_pos > j:
            new_pos -= (j - i + 1)
        new_ordering = rest[:new_pos] + seg + rest[new_pos:]
        return new_ordering
    
    def try_2opt(ordering, i, j):
        """Reverse segment ordering[i:j+1]"""
        new_ordering = ordering[:]
        new_ordering[i:j+1] = new_ordering[i:j+1][::-1]
        return new_ordering
    
    best_ordering = current_ordering[:]
    best_length = current_length
    
    # Local search with VNS
    improved = True
    while improved:
        improved = False
        # N1: adjacent swaps
        for i in range(k - 1):
            new_ordering = try_swap(best_ordering, i, i + 1)
            new_length = compute_length(new_ordering)
            if new_length < best_length:
                best_ordering = new_ordering
                best_length = new_length
                improved = True
                break
        if improved:
            continue
        
        # N2: Or-opt
        for seg_len in range(2, min(4, k)):
            for i in range(k - seg_len + 1):
                for j in range(k - seg_len + 1):
                    if j == i or (i + seg_len - 1 < j) or (j < i):
                        new_ordering = try_oropt(best_ordering, i, i + seg_len - 1, j)
                        if new_ordering != best_ordering:
                            new_length = compute_length(new_ordering)
                            if new_length < best_length:
                                best_ordering = new_ordering
                                best_length = new_length
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # N3: 2-opt (reverse segments)
        for i in range(k):
            for j in range(i + 2, k):
                new_ordering = try_2opt(best_ordering, i, j)
                if new_ordering != best_ordering:
                    new_length = compute_length(new_ordering)
                    if new_length < best_length:
                        best_ordering = new_ordering
                        best_length = new_length
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue
    
    # Reconstruct the word using the best ordering
    new_word = []
    for idx in best_ordering:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        overlap = 0
        if new_word:
            max_overlap = min(len(new_word), len(block))
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
        new_word += block[overlap:]
    
    result = "".join(map(str, new_word))
    
    # Verify: check that all permutations of 1..n appear as n consecutive symbols
    if n <= 7:
        from itertools import permutations
        all_perms = set(tuple(p) for p in permutations(range(1, n + 1)))
        found = set()
        for i in range(len(result) - n + 1):
            window = tuple(int(c) for c in result[i:i + n])
            found.add(window)
        if found != all_perms:
            # Fallback to the original word
            return "".join(map(str, word))
    
    return result
