import time
import random
from collections import deque

def construct(n):
    if n <= 1:
        return "1"
    
    # For small n, use known optimal constructions
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Precompute all permutations as tuples for quick lookup
    # For n=7, there are 5040 permutations - manageable
    from itertools import permutations as perms
    all_perms = list(perms(range(1, n+1)))
    perm_set = set(all_perms)
    num_perms = len(all_perms)
    
    # For n=7, we need to be efficient
    # Strategy: greedy with urgency
    # State: current word, set of covered permutations
    # At each step, choose next symbol to maximize:
    #   (new permutations completed) - urgency_penalty
    # where urgency penalizes symbols that lead to states with few remaining options
    
    best_word = None
    best_length = float('inf')
    
    # For n=7, the classic recursive construction gives 13699
    # We need much better. Let's try a different approach.
    # 
    # Key insight: A superpermutation can be viewed as a path in the de Bruijn graph
    # where vertices are (n-1)-tuples and edges add one symbol.
    # We need a trail that visits all n! vertices (each permutation is a vertex
    # in the (n-1)-de Bruijn graph... actually, each permutation of length n
    # corresponds to an edge in the (n-1)-de Bruijn graph).
    #
    # Actually, let me think differently. We want a word W of length L such that
    # every permutation of 1..n appears as a contiguous substring of length n.
    # 
    # For n=7, L should be around 5905. The trivial bound is 7*5040 = 35280.
    # The recursive construction gives sum(k!) for k=1..7 = 1+2+6+24+120+720+5040 = 5913.
    # Wait, that's actually very close to 5905!
    
    # Let me recalculate: 1!+2!+3!+4!+5!+6!+7! = 1+2+6+24+120+720+5040 = 5913
    # The known shortest is 5905, so we're only 8 off from the recursive construction!
    
    # So the recursive construction is actually near-optimal. Let me verify it works
    # and try to shave off a few characters.
    
    # Classic recursive construction:
    word = [1]
    for m in range(2, n + 1):
        perms_list, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_list.append(window)
        new = []
        for p in perms_list:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # This gives length 5913 for n=7
    # We need to verify it's valid and try to improve
    
    result = "".join(map(str, word))
    current_len = len(result)
    
    # Now try local optimization: remove redundant characters
    # A character at position i is redundant if removing it doesn't lose any permutation
    
    # For efficiency, let's track which permutations are covered
    def check_validity(w, n):
        """Check if word w contains all permutations of 1..n as substrings"""
        if len(w) < n:
            return False
        covered = set()
        for i in range(len(w) - n + 1):
            window = tuple(int(c) for c in w[i:i+n])
            covered.add(window)
        return len(covered) == len(all_perms)
    
    # Try to remove characters from the end or middle
    # Start from the recursive construction and try to shorten it
    
    w = list(result)
    
    # Try removing characters from various positions
    # Focus on positions where removal might not break coverage
    
    improvement_found = True
    while improvement_found and (time.time() - start_time) < deadline:
        improvement_found = False
        L = len(w)
        # Try removing each character (prioritize middle/end for efficiency)
        # For n=7, L~5905, trying all positions is feasible
        
        for i in range(L - 1, -1, -1):  # Try from end to start
            if (time.time() - start_time) > deadline:
                break
            if len(w) < n:
                break
            # Try removing position i
            w2 = w[:i] + w[i+1:]
            if len(w2) < n:
                continue
            # Quick check: does w2 still contain all perms?
            # This is expensive, so only do it if we're close
            if check_validity("".join(map(str, w2)), n):
                w = w2
                improvement_found = True
                break
    
    final_result = "".join(map(str, w))
    
    # Verify final result
    if not check_validity(final_result, n):
        # Fall back to the recursive construction
        final_result = result
    
    return final_result
