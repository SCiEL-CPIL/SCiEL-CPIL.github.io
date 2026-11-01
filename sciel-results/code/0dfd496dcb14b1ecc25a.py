"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Classic recursive construction
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
    
    # Verify and return if time is tight
    if time.time() > deadline - 5:
        return "".join(map(str, word))
    
    # Try to improve via local search: remove symbols where possible
    # A symbol at position i can be removed if all n-windows that include it still appear elsewhere
    # This is complex, so try a simpler approach: check if we can shorten by 1 at each position
    
    # For n=7, the known improvement is from 5913 to 5905 (remove 8 symbols)
    # Try greedy removal: for each position, check if removing it keeps all perms present
    
    # Build set of all required permutations
    all_perms = set(permutations(range(1, n + 1)))
    
    def get_windows(w):
        return set(tuple(w[i:i+n]) for i in range(len(w) - n + 1))
    
    current = list(word)
    current_windows = get_windows(current)
    
    # Greedy: try removing each symbol from left to right
    i = 0
    while i < len(current) - n + 1:
        if time.time() > deadline - 3:
            break
        # Try removing symbol at position i
        test = current[:i] + current[i+1:]
        test_windows = get_windows(test)
        if all_perms.issubset(test_windows):
            current = test
            current_windows = test_windows
            # Don't advance i, try removing at same position again
        else:
            i += 1
    
    return "".join(map(str, current))
# EVOLVE-BLOCK-END
