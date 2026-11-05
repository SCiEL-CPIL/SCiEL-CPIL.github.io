"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def construct(n):
    # Deadline guard: evaluator kills after 150s, we want a safety margin.
    start_time = time.time()
    DEADLINE = 140.0  # seconds
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214312"
    
    # For n >= 5, use the classic recursive construction which gives length 1!+2!+...+n!
    # This is the best known general construction. For n=5 it's 153, n=6 it's 873.
    # For n=7 the classic gives 5913, but the shortest known is 5905.
    # We'll use the classic construction as our baseline and try to improve with local search.
    
    def classic_superperm(n):
        """Classic recursive superpermutation construction."""
        if n == 1:
            return [1]
        
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
                # Find maximum overlap
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    # Get the classic construction
    word = classic_superperm(n)
    best_word = list(word)
    best_len = len(best_word)
    
    if n <= 4:
        return "".join(map(str, best_word))
    
    # Verify the word contains all permutations
    def count_permutations(word, n):
        """Count how many distinct n-permutations appear in the word."""
        if len(word) < n:
            return 0
        perms = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:
                perms.add(window)
        return len(perms)
    
    # For n=5, we can try local search to improve
    if n == 5:
        target = 120  # 5!
        current_count = count_permutations(best_word, n)
        
        # Try local search: swap adjacent elements, or insert/delete
        if current_count == target and best_len > 153:
            # Try to shorten
            pass  # 153 is optimal for n=5
        
        return "".join(map(str, best_word))
    
    # For n=6 and n=7, try to improve on the classic construction
    # The classic gives 873 for n=6 and 5913 for n=7
    # Best known: 873 for n=6, 5905 for n=7
    
    if n == 6:
        # Classic gives 873, which is optimal
        return "".join(map(str, best_word))
    
    if n == 7:
        # Classic gives 5913, best known is 5905
        # Try local search to find a shorter word
        target = 5040  # 7!
        
        # Check if our word is valid
        current_count = count_permutations(best_word, n)
        if current_count < target:
            # Something went wrong, return the classic
            return "".join(map(str, best_word))
        
        # Try to shorten by removing redundant characters
        # Strategy: try to find a character that can be removed without losing any permutation
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            return "".join(map(str, best_word))
        
        # Local search: try swapping, inserting, deleting
        # First, try to find deletions
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 10:
            improved = False
            for i in range(len(best_word)):
                # Try deleting position i
                candidate = best_word[:i] + best_word[i+1:]
                if len(candidate) < n:
                    continue
                # Quick check: does it still contain all permutations?
                # This is expensive, so only do it if we're close to the deadline
                if time.time() - start_time > DEADLINE - 15:
                    break
                c = count_permutations(candidate, n)
                if c == target:
                    best_word = candidate
                    best_len = len(best_word)
                    improved = True
                    break
            if time.time() - start_time > DEADLINE - 10:
                break
        
        return "".join(map(str, best_word))
    
    # For other n, just return the classic construction
    return "".join(map(str, best_word))

# EVOLVE-BLOCK-END
