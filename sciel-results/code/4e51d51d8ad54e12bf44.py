import itertools
import random
import math

def construct(n):
    # 1. Generate classic superpermutation sequence
    # This produces a list of permutations where consecutive ones overlap by n-1
    # and the resulting word is the classic superpermutation.
    
    # For n=1, return "1"
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build the classic sequence recursively
    # S(1) = [(1,)]
    # S(m) is built from S(m-1)
    # The word is constructed by interleaving.
    # Let's just generate the word first, then extract the permutation list.
    
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
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Extract permutation list from word
    # The list of permutations is just the sliding windows of size n
    perm_list = [tuple(word[i:i+n]) for i in range(len(word) - n + 1)]
    
    # Verify it's a valid walk
    # perm_list[i+1][0:n-1] == perm_list[i][1:n]
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Check coverage
    covered = set(perm_list)
    if len(covered) != len(all_perms):
        # Fallback
        return "".join(map(str, word))
        
    # Simulated Annealing
    # State: perm_list
    # Cost: len(perm_list) + 1000 * (len(all_perms) - len(set(perm_list)))
    
    current_list = perm_list
    current_cost = len(current_list)
    
    # Precompute neighbors for fast validity check?
    # Given a permutation P, the next permutation Q must satisfy Q[0:n-1] == P[1:n].
    # We can create a dictionary: prefix -> list of perms with that prefix.
    # prefix is P[1:n].
    # Actually, Q is determined by P[1:n] and the last element.
    # So for a given P, the valid next perms are those starting with P[1:n].
    
    # To check if A connects to B:
    # return A[1:] == B[:-1]
    
    def is_valid_walk(lst):
        for i in range(len(lst) - 1):
            if lst[i][1:] != lst[i+1][:-1]:
                return False
        return True
        
    # This is O(N) per check. N ~ 6000.
    # In SA, we do many checks. We need incremental updates.
    
    # Let's use a simpler heuristic: Hill Climbing with Random Restarts.
    # Or just a very aggressive local search on the list.
    
    # Move 1: Delete an element if it's a duplicate and the neighbors connect.
    # Move 2: Swap adjacent elements if the new sequence is valid.
    
    # We will try to minimize the length.
    
    best_list = current_list[:]
    best_len = len(best_list)
    
    # Time limit check
    import time
    start_time = time.time()
    timeout = 110 # Leave 10s for overhead
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        # Iterate through the list
        # We scan from left to right, trying to delete.
        # If we delete, we restart the scan from the previous position.
        
        i = 0
        while i < len(current_list) - 1:
            # Try to delete current_list[i]
            # Condition 1: current_list[i] appears elsewhere in the list
            # Condition 2: current_list[i-1] connects to current_list[i+1]
            
            # Check Condition 1: Is it a duplicate?
            # We can maintain a count dict.
            # But updating count dict on deletion is easy.
            
            # Let's just check if it's the only occurrence.
            # O(N) check is too slow if done for every i.
            # Let's maintain a frequency map.
            
            # We'll restructure the loop to maintain counts.
            pass # Placeholder for logic
            
        # If simple deletion fails, try swaps?
        # For now, let's just do a robust deletion pass with frequency map.
        
        # Reset for a new pass
        freq = {}
        for p in current_list:
            freq[p] = freq.get(p, 0) + 1
            
        i = 0
        while i < len(current_list) - 1:
            # Try to delete current_list[i]
            p_i = current_list[i]
            
            # Check if duplicate
            if freq[p_i] <= 1:
                i += 1
                continue
                
            # Check connectivity
            # If i == 0, we need to check if we can just drop it? 
            # No, the list must remain a valid walk.
            # If we drop index 0, the new start is current_list[1].
            # This is always valid as a walk (just shorter).
            # But does it break coverage? No, if duplicate.
            # So if i == 0 and freq > 1, we can delete.
            
            if i == 0:
                # Delete
                del current_list[0]
                freq[p_i] -= 1
                improved = True
                continue # Don't increment i, check new 0
                
            if i == len(current_list) - 1:
                # Delete
                del current_list[i]
                freq[p_i] -= 1
                improved = True
                break # End of list
                
            # Internal element
            p_prev = current_list[i-1]
            p_next = current_list[i+1]
            
            # Check if p_prev connects to p_next
            if p_prev[1:] == p_next[:-1]:
                # We can delete p_i
                del current_list[i]
                freq[p_i] -= 1
                improved = True
                # Decrement i to re-check the new element at position i (which was i+1)
                # Actually, the element that was at i+1 is now at i.
                # We should check it again? Or just continue?
                # If we delete, the new element at i is the old i+1.
                # We should check if *that* can be deleted.
                # So we do NOT increment i.
                continue
                
            i += 1
            
        if len(current_list) < best_len:
            best_list = current_list[:]
            best_len = len(best_list)
            
    # Convert best_list back to word
    # Word is first perm + rest of perms without first n-1 chars
    if not best_list:
        return ""
        
    word_chars = list(best_list[0])
    for p in best_list[1:]:
        word_chars.extend(p[1:])
        
    return "".join(map(str, word_chars))
