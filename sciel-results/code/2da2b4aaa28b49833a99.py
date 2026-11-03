import itertools
import time

def construct(q, n, radius):
    # Only handling the specific case q=3, n=8, radius=1 for optimization
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed for other cases (though task implies specific instance)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # 1. Generate initial code: Tetracode x Full Space(4)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # Convert to tuples for hashing and set operations
    code = set(initial_code)
    
    # Precompute neighbors for all possible words in the space? 
    # Space is 3^8 = 6561. 
    # We can map word (tuple) to integer index for faster counting.
    # Or just use a dict for coverage counts.
    
    # Let's use integer representation for speed.
    # word_to_int: tuple of 8 ints -> int in 0..6560
    # int_to_word: int -> tuple
    
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * 3 + w[i]
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = val % 3
            val //= 3
        return tuple(w)

    # Map initial code to ints
    code_ints = set(word_to_int(w) for w in initial_code)
    
    # Precompute neighbors for all 6561 words
    # neighbors[i] is a list of ints within distance 1 of i
    all_words = [int_to_word(i) for i in range(3**n)]
    neighbors = [[] for _ in range(3**n)]
    for i in range(3**n):
        w = all_words[i]
        neighbors[i].append(i) # self
        for pos in range(n):
            for val in range(3):
                if w[pos] != val:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors[i].append(word_to_int(tuple(new_w)))
                    
    # coverage_count[x] = number of codewords in current code covering x
    coverage_count = [0] * (3**n)
    
    # Initialize coverage
    for c in code_ints:
        for nb in neighbors[c]:
            coverage_count[nb] += 1
            
    # Redundancy removal loop
    # We iterate through the current code. If a codeword is redundant, we remove it.
    # A codeword c is redundant if for all nb in neighbors[c], coverage_count[nb] > 1.
    # Note: If we remove c, we must decrement coverage_count for all its neighbors.
    
    # To avoid modifying list while iterating, we collect to remove.
    # However, removing one might make another redundant.
    # We can do passes until stable.
    
    # Optimization: If a codeword is the *only* cover for some neighbor, it's not redundant.
    # So we just check min(coverage_count[nb] for nb in neighbors[c]) > 1.
    
    # Start with a list of codewords to check
    # Using a set for code_ints is good for membership, but we need to iterate.
    # Let's convert to list for iteration, but keep set for O(1) removal?
    # Actually, just iterating a list and removing from it is O(N^2) in worst case, but N is small.
    
    current_code = list(code_ints)
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(current_code):
            if time.time() > deadline:
                break
            c = current_code[i]
            # Check if c is redundant
            redundant = True
            for nb in neighbors[c]:
                if coverage_count[nb] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                current_code.pop(i)
                for nb in neighbors[c]:
                    coverage_count[nb] -= 1
                changed = True
                # Do not increment i, as the next element shifted into position i
            else:
                i += 1
        
        if time.time() > deadline:
            break
            
    # Convert back to list of lists
    result = [list(int_to_word(c)) for c in current_code]
    return result
