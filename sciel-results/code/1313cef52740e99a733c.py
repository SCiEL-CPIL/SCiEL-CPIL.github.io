import time
import sys
import random

def construct_standard(n):
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
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def compute_counts(word, n):
    counts = {}
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            counts[w] = counts.get(w, 0) + 1
    return counts

def is_valid_deletion(word, i, n, counts):
    # Check if removing word[i] is safe
    # Affected windows: j such that j <= i < j+n
    # j in [i-n+1, i]
    start = max(0, i - n + 1)
    end = min(i + 1, len(word) - n + 1)
    
    for j in range(start, end):
        w = tuple(word[j:j+n])
        if len(set(w)) == n:
            if counts.get(w, 0) <= 1:
                return False
    return True

def is_valid_swap(word, i, n, counts):
    # Swap word[i] and word[i+1]
    # Check if all affected windows remain permutations
    # Affected windows: j in [i-n+1, i+1]
    start = max(0, i - n + 1)
    end = min(i + 2, len(word) - n + 1)
    
    # Create a copy of the word with the swap to check?
    # Or just check the specific windows.
    # We need to ensure that NO window becomes invalid.
    # And that we don't lose any unique permutation?
    # Swapping doesn't change the set of characters in the global word, 
    # but it changes the windows.
    # A swap is valid if:
    # 1. Every window that was a permutation before is still a permutation?
    #    No, we just need the FINAL word to contain all permutations.
    #    It is safer to require that the swap does not destroy any unique permutation.
    #    But a swap might turn a non-perm window into a perm window, or vice versa.
    #    The condition "contains every permutation" must hold.
    #    
    #    Simplest robust check for swap:
    #    Perform swap on a copy.
    #    Compute counts on the copy.
    #    Check if all 5040 perms are present.
    #    This is O(L*N) per swap check.
    #    If we do 1000 swaps, it's 1000 * 42000 = 42M ops. Feasible.
    
    pass # We will implement a full check for swaps to be safe, or restrict swaps to "safe" ones.

def construct(n):
    start_time = time.time()
    deadline = start_time + 115 # 5s margin
    
    # Generate standard
    word = construct_standard(n)
    
    # Initial counts
    counts = compute_counts(word, n)
    
    # Phase 1: Greedy Deletion (Left to Right)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(word):
            if is_valid_deletion(word, i, n, counts):
                del word[i]
                counts = compute_counts(word, n)
                improved = True
            else:
                i += 1
                
    # Phase 2: Greedy Deletion (Right to Left)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = len(word) - 1
        while i >= 0:
            if is_valid_deletion(word, i, n, counts):
                del word[i]
                counts = compute_counts(word, n)
                improved = True
            else:
                i -= 1
                
    # Phase 3: Randomized Search
    # Use random deletions and swaps
    # We will use a "full validation" for swaps to ensure validity.
    # For deletions, we use the fast check.
    
    no_deletion_count = 0
    max_no_deletion = 100
    
    while time.time() < deadline:
        if no_deletion_count > max_no_deletion:
            # Try a swap to perturb
            if len(word) > 1:
                i = random.randint(0, len(word) - 2)
                # Check if swap is valid
                # Copy word
                w_copy = word[:]
                w_copy[i], w_copy[i+1] = w_copy[i+1], w_copy[i]
                # Check if w_copy is a valid superpermutation
                # This is expensive.
                # Optimization: Only check if the swap might be beneficial?
                # Let's just do it.
                c_copy = compute_counts(w_copy, n)
                # Check if all perms are present
                # We know the set of all perms.
                # If len(c_copy) == n! and all counts > 0, it's valid.
                if len(c_copy) == int(math.factorial(n)):
                    word = w_copy
                    counts = c_copy
                    no_deletion_count = 0 # Reset? Or keep?
                    # Reset to allow more swaps if needed
            else:
                break
        else:
            # Try random deletion
            i = random.randint(0, len(word) - 1)
            if is_valid_deletion(word, i, n, counts):
                del word[i]
                counts = compute_counts(word, n)
                no_deletion_count = 0
            else:
                no_deletion_count += 1
                if no_deletion_count > max_no_deletion:
                    # Force a swap attempt
                    if len(word) > 1:
                        i_swap = random.randint(0, len(word) - 2)
                        w_copy = word[:]
                        w_copy[i_swap], w_copy[i_swap+1] = w_copy[i_swap+1], w_copy[i_swap]
                        c_copy = compute_counts(w_copy, n)
                        if len(c_copy) == int(math.factorial(n)):
                            word = w_copy
                            counts = c_copy
                            no_deletion_count = 0

    return "".join(map(str, word))

# Need math import
import math
