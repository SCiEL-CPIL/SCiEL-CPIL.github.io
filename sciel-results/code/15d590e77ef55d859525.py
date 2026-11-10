<<<<<<< SEARCH
import time
import random
import numpy as np

def _try_greedy_fast(order, deadline):
    """Fast greedy using numpy for vectorized checks."""
    chosen = []
    # counts[s] = number of representations of s
    # Use a dict for sparse counts
    counts = {}
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Generate new sums: x + y for y in chosen, and 2*x
        if chosen:
            new_sums = [x + y for y in chosen]
        else:
            new_sums = []
        new_sums.append(2 * x)
        
        # Check validity: no sum can have count >= 2 already
        valid = True
        for s in new_sums:
            if counts.get(s, 0) >= 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return chosen
=======
import time
import random
import numpy as np

def _try_greedy_fast(order, deadline):
    """Fast greedy using numpy for vectorized checks."""
    chosen = []
    # counts[s] = number of representations of s
    # Use a dict for sparse counts
    counts = {}
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Generate new sums: x + y for y in chosen, and 2*x
        if chosen:
            new_sums = [x + y for y in chosen]
        else:
            new_sums = []
        new_sums.append(2 * x)
        
        # Check validity: no sum can have count >= 2 already
        valid = True
        for s in new_sums:
            if counts.get(s, 0) >= 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return chosen

def _structured_seed():
    """Generate a structured B2[2] seed using modular arithmetic patterns."""
    # Use a pattern based on quadratic residues or similar
    # Try: {i * k + offset for i in range(m)} with appropriate k
    # For B2[2], a good starting point is a Sidon-like set
    
    # Start with a small Sidon set and scale it
    # A known small Sidon set: {1, 2, 4, 8, 13, 21, ...}
    # But we want B2[2], not B2[1], so we can be denser
    
    # Try a grid-like pattern
    chosen = set()
    # Use blocks: in each block of size B, pick elements that don't conflict
    B = 32
    for block in range(1000 // B + 1):
        base = block * B
        # Pick elements spaced out within the block
        for i in range(0, B, 3):
            x = base + i + 1
            if x <= 1000:
                chosen.add(x)
    
    return sorted(chosen)
>>>>>>> REPLACE

