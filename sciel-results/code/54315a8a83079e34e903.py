import time
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with radius 'radius'.
    For the specific case q=3, n=8, radius=1, returns the optimal known code of size 486.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Special case: Ternary code, length 8, radius 1.
    # The direct product of the ternary tetracode (length 4, size 9) and the full space of length 4 (size 3^4=81)
    # yields 9 * 81 = 486 codewords. This is the best known upper bound.
    if q == 3 and n == 8 and radius == 1:
        # Tetracode: 9 words of length 4
        # Words are (a, b, a+b, a+2b) mod 3 for a,b in {0,1,2}
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
        
        # Generate all tails of length 4
        tails = list(itertools.product(range(3), repeat=4))
        
        # Construct the full code
        code = []
        for head in tetracode:
            for tail in tails:
                code.append(head + list(tail))
        
        return code

    # Generic fallback for other parameters (not expected for this task, but good for robustness)
    # Simple greedy: add random words until covered (inefficient, but valid)
    import random
    import numpy as np
    
    # For small spaces, we can use a greedy set cover approach
    # Space size q**n. If q**n is too large, this will fail/timeout.
    # Given the constraints, this path is unlikely to be hit for the main task.
    
    # If n is small, we can try a greedy approach
    if q ** n <= 100000:
        # Generate all words
        all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
        uncovered = set(all_words)
        code = []
        
        # Greedy: pick word that covers most uncovered neighbors
        # Pre-compute neighbors for speed? Might be too slow for large n.
        # For small n, it's fine.
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered word or the one that covers most?
            # For simplicity and speed in fallback, pick random uncovered word
            # This might not be optimal but is valid.
            # Actually, let's just pick the first uncovered word to ensure progress.
            w = next(iter(uncovered))
            code.append(w)
            
            # Remove all words within radius of w from uncovered
            # This is O(|uncovered| * n) per step, which is slow.
            # Better: iterate uncovered and check distance.
            new_uncovered = []
            for u in uncovered:
                dist = sum(1 for i in range(n) if u[i] != w[i])
                if dist > radius:
                    new_uncovered.append(u)
            uncovered = set(new_uncovered)
            
        return code
    else:
        # If space is too large for greedy, return empty or raise?
        # The task requires valid code. If we can't construct one, we should try harder.
        # But for this specific task, the special case handles it.
        # Return a trivial code (just one word) is invalid for radius 1 unless n=1.
        # We'll raise an error if we can't handle it, but the special case should prevent this.
        raise ValueError("Constructing code for large space not implemented in fallback")
