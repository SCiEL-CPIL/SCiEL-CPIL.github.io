"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import permutations
from math import factorial

# EVOLVE-BLOCK-START

def _classic_superpermutation(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
    return word

def _find_best_order(n, time_limit):
    """Find a good ordering of permutations for the recursive construction."""
    start_time = time.time()
    
    # For small n, we can search for better orderings
    if n <= 5:
        # Try different orderings and pick the one that gives shortest word
        base_perms = list(permutations(range(1, n)))
        best_len = float('inf')
        best_order = None
        
        # Try identity order first
        word = _classic_superpermutation(n)
        best_len = len(word)
        best_order = list(range(len(base_perms)))
        
        # Try a few random orderings if time permits
        import random
        attempts = 0
        while time.time() - start_time < time_limit * 0.5 and attempts < 100:
            attempts += 1
            order = list(range(len(base_perms)))
            random.shuffle(order)
            
            # Build word with this order
            word = [1]
            for m in range(2, n + 1):
                # Get current permutations in the word
                perms_in_word, seen = [], set()
                for i in range(len(word) - m + 2):
                    window = tuple(word[i:i + m - 1])
                    if len(set(window)) == m - 1 and window not in seen:
                        seen.add(window)
                        perms_in_word.append(window)
                
                # Reorder based on our heuristic
                # For now, just use the natural order
                new = []
                for p in perms_in_word:
                    block = list(p) + [m] + list(p)
                    overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                    new += block[overlap:]
                word = new
            
            if len(word) < best_len:
                best_len = len(word)
                best_order = order
        
        return best_len
    
    return None

def construct(n):
    """Construct a superpermutation of length <= n*n! for n symbols."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # For n=1
    if n == 1:
        return "1"
    
    # Use classic construction as baseline
    word = _classic_superpermutation(n)
    best_word = list(word)
    best_len = len(word)
    
    # For n >= 7, the classic construction is already quite good
    # Try to improve with local search if time permits
    if n >= 4 and time.time() < deadline:
        # Local search: try to find better transitions
        # This is a simplified approach - try swapping adjacent elements in the permutation list
        improved = True
        while improved and time.time() < deadline:
            improved = False
            word = list(best_word)
            
            # Extract the permutation sequence used in the last step
            m = n
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Try swapping adjacent permutations
            for i in range(len(perms) - 1):
                if time.time() >= deadline:
                    break
                
                # Swap perms[i] and perms[i+1]
                perms_swapped = perms[:]
                perms_swapped[i], perms_swapped[i+1] = perms_swapped[i+1], perms_swapped[i]
                
                # Rebuild word with swapped order
                new_word = [1]
                for mm in range(2, n + 1):
                    if mm < n:
                        # Use original order for earlier steps
                        p_list, seen_p = [], set()
                        for ii in range(len(new_word) - mm + 2):
                            w = tuple(new_word[ii:ii + mm - 1])
                            if len(set(w)) == mm - 1 and w not in seen_p:
                                seen_p.add(w)
                                p_list.append(w)
                    else:
                        p_list = perms_swapped
                    
                    new = []
                    for p in p_list:
                        block = list(p) + [mm] + list(p)
                        overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                        new += block[overlap:]
                    new_word = new
                
                if len(new_word) < best_len:
                    best_len = len(new_word)
                    best_word = new_word
                    improved = True
    
    return "".join(map(str, best_word))

# EVOLVE-BLOCK-END
