import sys
import time
import itertools

def is_valid_superperm(word, n):
    """Check if word contains all permutations of 1..n"""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        # Check if it's a permutation of 1..n
        # Fast check: sum of elements should be n(n+1)/2 and all unique
        # Or just add to set and check size at the end.
        # But we need to ensure it's a permutation.
        # A tuple of length n with unique elements from 1..n is a permutation.
        if len(set(window)) == n:
            seen.add(window)
            if len(seen) == factorial(n):
                return True
    return len(seen) == factorial(n)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

def construct(n):
    if n <= 0:
        return ""
    if n == 1:
        return "1"
    
    # 1. Generate Classic Superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        target_count = 1
        for i in range(1, m):
            target_count *= i
            
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
                    if len(perms) == target_count:
                        break
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        word = new_word
        
    # 2. Deletion Heuristic (Local Search)
    # Try to remove characters while maintaining validity.
    # To speed up, we can check validity incrementally or just re-check.
    # Re-checking is O(L * n). L ~ 6000, n=7. 42000 ops per check.
    # 6000 checks -> 250M ops. Too slow in Python.
    
    # Optimization: 
    # When we remove a char at index i, only windows containing i are affected.
    # Windows are of size n.
    # Affected windows: [i-n+1, i] to [i, i+n-1] (indices in original word).
    # We can maintain the set of seen permutations and a count of occurrences.
    # If a permutation count drops to 0, it's invalid.
    
    # Let's implement a faster validity check using counts.
    
    def build_counts(w, n):
        counts = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                counts[window] = counts.get(window, 0) + 1
        return counts

    def is_valid_counts(counts, n):
        # Check if all n! permutations are present
        # We don't have the list of all perms, but we know the size.
        # If len(counts) == n! and all values > 0, it's valid.
        # We need n! value.
        return len(counts) == factorial(n)

    # Initial counts
    counts = build_counts(word, n)
    if not is_valid_counts(counts, n):
        return "".join(map(str, word)) # Should not happen

    # Try deletions
    # We iterate through indices. If a deletion is valid, we perform it and restart the scan from the beginning?
    # Or just continue? Restarting is safer to find more deletions.
    # But it's slow.
    # Let's try a single pass first, then repeat until no change.
    
    max_iterations = 10
    for _ in range(max_iterations):
        changed = False
        # We need to be careful with indices when deleting.
        # It's better to scan from left to right, and if we delete, restart the scan.
        # Or scan right to left?
        
        # Let's scan from left to right.
        i = 0
        while i < len(word):
            # Try deleting word[i]
            # Update counts for affected windows
            # Windows affected by deleting index i:
            # Start indices j such that j <= i <= j+n-1
            # j in [i-n+1, i]
            # Also, windows after i shift left.
            # This is complex to update incrementally in a simple loop.
            
            # Alternative: Just re-build counts for the modified word?
            # Too slow.
            
            # Let's use a simpler check:
            # Remove char, check validity.
            # If valid, accept.
            # To speed up, we can pre-compute the set of permutations.
            
            # Actually, for n=7, L=5913.
            # Removing one char, L becomes 5912.
            # Check validity: iterate 5912 windows.
            # 5912 * 7 ~ 40k ops.
            # 5913 positions * 40k ops = 236M ops.
            # In Python, 236M simple ops might take 10-20 seconds.
            # We have 120s.
            # So we can do a few passes.
            
            # Let's try to delete word[i]
            new_word = word[:i] + word[i+1:]
            # Check validity
            valid = True
            seen = set()
            for j in range(len(new_word) - n + 1):
                window = tuple(new_word[j:j+n])
                if len(set(window)) == n:
                    seen.add(window)
                    if len(seen) == factorial(n):
                        break
            if len(seen) == factorial(n):
                word = new_word
                changed = True
                # Restart scan from beginning?
                # If we restart, we might find more deletions earlier in the string.
                # Let's restart.
                break
            else:
                i += 1
                
        if not changed:
            break
            
    return "".join(map(str, word))
