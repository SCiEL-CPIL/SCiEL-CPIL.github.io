import time
import random

def construct(n):
    # Step 1: Generate the standard recursive superpermutation
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
            # Find max overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Step 2: Optimization via Deletion
    # Target: Shorten the word while maintaining validity
    
    # Precompute all valid permutations for quick check?
    # For n=7, 5040 permutations.
    # We can store them in a set.
    
    import itertools
    all_perms = set(itertools.permutations(range(1, n+1)))
    
    def get_windows(w):
        return [tuple(w[i:i+n]) for i in range(len(w) - n + 1)]
    
    def is_valid(w):
        if len(w) < n: return False
        windows = set(get_windows(w))
        return all_perms.issubset(windows)
    
    # We want to delete characters.
    # To speed up, we can try to delete characters that are "safe".
    # A character is safe to delete if no permutation is lost.
    
    # Initial check
    if not is_valid(word):
        return "".join(map(str, word))
        
    # Heuristic: Try to delete characters.
    # We'll do a few passes.
    # To fit in 120s, we limit the number of attempts.
    
    start_time = time.time()
    max_time = 100 # Leave 20s margin
    
    # Strategy: 
    # 1. Count frequency of each permutation.
    # 2. If a permutation has freq > 1, it's "safe" to remove one of its instances 
    #    if that removal doesn't destroy other unique permutations.
    
    # Simpler: Just try to delete random characters or characters in the middle.
    # But we must ensure validity.
    
    # Let's try a "Local Search" approach.
    # We'll try to delete characters from index 1 to len-2.
    # If deletion is valid, we do it.
    
    # To make it faster, we can use a "Sliding Window" validity check.
    # But for simplicity and robustness within time limits, we will use a 
    # limited number of full checks.
    
    attempts = 0
    max_attempts = 2000 # Adjust based on time
    
    while time.time() - start_time < max_time:
        attempts += 1
        # Try to delete a random character (not first or last)
        # Or iterate through positions.
        # Iterating is more deterministic.
        
        # Let's try to delete from the middle outwards, or just iterate 1..L-2
        # But iterating 6000 times and doing a full check (6000*7 ops) is 42k ops per deletion.
        # 6000 deletions * 42k = 252M ops. Too slow for Python.
        
        # We need a faster validity check or a smarter heuristic.
        
        # Heuristic: Only try to delete characters that are part of "redundant" overlaps.
        # This is complex.
        
        # Alternative: Use the known fact that the standard construction has specific 
        # "waste". 
        # For n=7, the waste is 8 characters.
        # We can hardcode the positions to delete if we know them? 
        # No, we must compute.
        
        # Let's try a "Greedy Deletion" with a faster check.
        # We maintain a set of current windows.
        # When we delete index i:
        #   Remove windows starting at i-n+1 ... i
        #   Add windows starting at i-n+1 ... i-1 (shifted)
        
        # This update is O(n).
        # Then we check if any permutation count drops to 0.
        
        current_windows = set(get_windows(word))
        # We need counts, not just set.
        from collections import Counter
        window_counts = Counter(get_windows(word))
        
        # If any count is 1, that permutation is "critical".
        # We can only delete a character if it doesn't affect any critical permutation's count to 0.
        
        # Let's implement the incremental update.
        
        def try_delete(i):
            # Returns True if deletion is valid, and updates state if so.
            # i is index in word.
            # Windows affected:
            # Old windows: start j in [i-n+1, i]
            # New windows: start j in [i-n+1, i-1] (these are the old windows starting at j+1? No)
            
            # Word: ... W[i-1] W[i] W[i+1] ...
            # Delete W[i].
            # New Word: ... W[i-1] W[i+1] ...
            
            # Old windows containing W[i]:
            # Start indices: i-n+1, ..., i
            # These are:
            # W[i-n+1]...W[i]
            # ...
            # W[i-1]...W[i]
            # W[i]...W[i+n-1] -> No, window length is n.
            # Window starting at i: W[i]...W[i+n-1]
            
            # New windows:
            # The windows that *would have* started at i-n+1 ... i-1 now use different elements.
            # Actually, the set of windows changes.
            
            # Let's just do the full check for the first few iterations to be safe, 
            # and if it's too slow, we fall back to returning the standard word.
            
            # Given the 120s limit, and Python speed, full check is ~1ms for L=6000.
            # 1000 full checks = 1s.
            # We can do ~100,000 full checks in 100s.
            # This is plenty to find 8 deletions if we are smart.
            
            pass # Placeholder
            
        # Let's just do a simple loop of random deletions with full check.
        # Random is faster to code and might find the spots.
        
        while time.time() - start_time < max_time:
            # Pick a random index to delete
            if len(word) <= n: break
            i = random.randint(1, len(word) - 2)
            
            # Try delete
            new_word = word[:i] + word[i+1:]
            
            # Quick check: Length must be >= n
            if len(new_word) < n:
                continue
                
            # Full check
            if is_valid(new_word):
                word = new_word
                # Reset some counter? No, just continue.
            else:
                # Do nothing, try another index
                pass
                
            # Add a small sleep to avoid 100% CPU? No, we want to use time.
            # But Python GIL is not an issue here.
            
    return "".join(map(str, word))
