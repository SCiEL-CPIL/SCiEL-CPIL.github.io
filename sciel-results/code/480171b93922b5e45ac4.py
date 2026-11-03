import time
import random
from itertools import permutations
from collections import deque

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if n <= 1:
        return "1"
    
    # For small n, use known optimal constructions
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412132413214231342314321"
    if n == 5:
        # Classic is optimal for n=5 (153)
        return _classic_superperm(5)
    if n == 6:
        # Classic is optimal for n=6 (873)
        return _classic_superperm(6)
    
    # For n >= 7, start with classic and try to improve
    if n > 7:
        # Fallback to classic for larger n as search space explodes
        return _classic_superperm(n)

    # n = 7
    # Generate classic superpermutation
    word = _classic_superperm(n)
    
    # Local Search / Simulated Annealing to shorten
    # Goal: Remove characters while maintaining validity
    # Validity: All n! permutations appear as consecutive windows
    
    num_perms = factorial(n)
    all_perms_set = set(permutations(range(1, n + 1)))
    
    def is_valid_superperm(s):
        if len(s) < n:
            return False
        seen = set()
        for i in range(len(s) - n + 1):
            window = tuple(s[i:i + n])
            seen.add(window)
            if len(seen) == num_perms:
                return True
        return len(seen) == num_perms

    def get_missing_count(s):
        if len(s) < n:
            return num_perms
        seen = set()
        for i in range(len(s) - n + 1):
            window = tuple(s[i:i + n])
            seen.add(window)
        return num_perms - len(seen)

    # Start with the classic word
    current_word = word
    current_missing = get_missing_count(current_word)
    
    # If classic is already valid (it is), missing is 0
    # We want to remove chars. Removing a char increases missing count potentially.
    # We use SA to accept bad moves to escape local minima, then try to fix.
    
    # Actually, a simpler approach:
    # 1. Try to remove every character. If valid, keep it.
    # 2. If stuck, try to swap/insert to create new overlaps?
    #    Inserting makes it longer. We want shorter.
    #    So we only consider removals and maybe "sliding" windows?
    #    Sliding is equivalent to removal + insertion.
    
    # Let's try a heuristic:
    # Identify "redundant" characters.
    # A character is redundant if removing it doesn't destroy any unique permutation window.
    # Actually, if a permutation appears in multiple windows, removing a char might merge/shift them.
    
    # Simple Local Search:
    # Iterate through indices. Try to remove.
    # If valid, remove.
    # If invalid, try to "move" the character? No.
    
    # Let's use Simulated Annealing on the length.
    # State: The string.
    # Move: Remove a random character.
    # Cost: Number of missing permutations.
    # We want to reach 0 missing with minimal length.
    
    # Since we start at 0 missing, we want to keep it 0 while reducing length.
    # So, Move: Remove a char.
    # If new missing == 0, accept.
    # If new missing > 0, accept with prob exp(-delta / T).
    # If we accept a bad move, we must later try to "fix" it by inserting chars?
    # This is complex.
    
    # Alternative: 
    # The classic construction has a specific structure.
    # Maybe we can just return the classic one?
    # Score 1.0 is safe.
    # But we want > 1.0.
    
    # Let's try a limited search:
    # Try removing each character one by one from the end to the start.
    # If valid, keep removed.
    # This is a greedy reduction.
    
    improved_word = current_word
    improved_missing = 0
    
    # Greedy Reduction
    changed = True
    while changed:
        changed = False
        for i in range(len(improved_word) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing char at i
            new_word = improved_word[:i] + improved_word[i+1:]
            if is_valid_superperm(new_word):
                improved_word = new_word
                changed = True
                break # Restart scan from beginning? Or continue?
                # Restarting is safer to find more removals
        if time.time() > deadline:
            break
            
    # If we improved, return improved
    if len(improved_word) < len(current_word):
        return improved_word
        
    # If no improvement, return classic
    return current_word

def _classic_superperm(n):
    """
    Generates the classic superpermutation of length sum(k! for k=1..n)
    """
    if n == 1:
        return "1"
    
    # Recursive construction
    # S_1 = "1"
    # S_{n} is built from S_{n-1}
    # The standard construction:
    # Let S_{n-1} be the superperm for n-1.
    # For each permutation p of 1..n-1 that appears in S_{n-1} (in order of appearance):
    #   Append p, then n, then p again?
    # No, the standard construction is:
    # S_n = S_{n-1}
    # For each occurrence of a permutation of 1..n-1 in S_{n-1}:
    #   Append the missing symbol n?
    
    # Actually, the well-known construction is:
    # S_1 = "1"
    # S_2 = "121"
    # S_3 = "123121321"
    # S_4 = "123412132413214231342314321"
    
    # Algorithm:
    # Start with S = "1"
    # For m from 2 to n:
    #   new_S = []
    #   For each window of size m-1 in S that contains m-1 distinct elements:
    #      Append window, then m, then window?
    #   Wait, the standard recursive step is:
    #   Given S_{m-1}, to get S_m:
    #   For each permutation p of {1,...,m-1} in the order they appear in S_{m-1}:
    #       Append p, then m, then p?
    #   No, that doubles the length roughly.
    
    # Correct recursive construction:
    # S_1 = "1"
    # S_m is formed by taking S_{m-1} and for every occurrence of a permutation of {1..m-1},
    # inserting m before the last element? No.
    
    # Let's use the known list for small n and a generator for n=7.
    # Since we only need n=7, and we know the length is 5913,
    # we can generate it using the standard algorithm:
    
    # Algorithm from Wikipedia "Superpermutation":
    # The shortest known superpermutation for n=5 is 153.
    # Construction:
    # Let S be the superpermutation for n-1.
    # For each permutation p of {1, ..., n-1} that appears in S:
    #   Append p, then n, then p?
    
    # Actually, the construction is:
    # S_n = S_{n-1}
    # For each permutation p of {1..n-1} in order of appearance in S_{n-1}:
    #   Append p, then n, then p?
    
    # Let's implement the specific algorithm that yields the classic length.
    # Reference: 
    # S_1 = "1"
    # S_2 = "121"
    # S_3 = "123121321"
    # S_4 = "123412132413214231342314321"
    
    # The pattern is:
    # S_n is a sequence of blocks.
    # For n=3:
    # 123
    # 121
    # 321
    # Wait, "123121321"
    # Windows: 123, 231, 312, 121, 213, 132, 321.
    # Perms of 1,2,3: 123, 132, 213, 231, 312, 321. All present.
    
    # Let's use a known Python implementation of the classic superperm generator.
    
    def generate_classic(n):
        if n == 1:
            return [1]
        
        # Get classic for n-1
        prev = generate_classic(n - 1)
        
        # We need to insert n into the sequence.
        # The standard construction iterates through all permutations of 1..n-1 
        # in the order they appear in the previous superperm.
        
        # First, extract all permutations of 1..n-1 from 'prev' in order.
        perms = []
        seen = set()
        for i in range(len(prev) - (n - 1) + 1):
            window = tuple(prev[i:i + n - 1])
            if window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Now, construct the new superperm.
        # For each permutation p in perms:
        #   Append p, then n, then p?
        #   No, the overlap is handled by the structure.
        
        # The correct recursive step:
        # new_word = []
        # for p in perms:
        #     new_word.extend(p)
        #     new_word.append(n)
        #     new_word.extend(p) # This seems too long
        
        # Actually, the construction is:
        # S_n = S_{n-1}
        # For each permutation p of {1..n-1} in order of appearance:
        #   Insert n before the last element of p?
        
        # Let's try a different approach: 
        # The classic superperm for n is a concatenation of (n-1)! blocks of length n?
        
        # Let's just use the hardcoded strings for n<=6 and for n=7, 
        # if we can't generate it, we use the fallback.
        # But we need to generate it to start the search.
        
        # Here is a working algorithm for classic superperm:
        # https://paulfollmer.com/superpermutations.html
        # The construction is:
        # S_1 = "1"
        # S_n is obtained from S_{n-1} by:
        #   For each permutation p of {1..n-1} in the order they appear in S_{n-1}:
        #       Append p, then n, then p?
        
        # Let's verify for n=3 from n=2.
        # S_2 = "121". Perms of {1,2}: 12, 21.
        # Order in S_2: 12 (at 0), 21 (at 1).
        # Step 1: p=12. Append 1, 2, 3, 1, 2? -> "12312"
        # Step 2: p=21. Append 2, 1, 3, 2, 1? -> "21321"
        # Concatenate with overlap?
        # "12312" + "21321" -> "123121321" (Overlap "2"?)
        # 12312
        #   21321
        # Overlap is "2"? 
        # 12312 ends with 2. 21321 starts with 2.
        # So 123121321.
        # This matches S_3!
        
        # So the algorithm is:
        # 1. Get S_{n-1}.
        # 2. Extract unique permutations of {1..n-1} in order of appearance.
        # 3. For each such permutation p:
        #      Block = p + [n] + p
        # 4. Concatenate these blocks, maximizing overlap between consecutive blocks.
        
        result = []
        for p in perms:
            block = list(p) + [n] + list(p)
            # Find max overlap with current result
            max_overlap = 0
            for o in range(min(len(result), len(block)), -1, -1):
                if o == 0:
                    max_overlap = 0
                    break
                if result[len(result)-o:] == block[:o]:
                    max_overlap = o
                    break
            result.extend(block[max_overlap:])
            
        return result

    return "".join(map(str, generate_classic(n)))

from math import factorial

# Re-define construct to use the helper
def construct(n):
    start_time = time.time()
    deadline = start_time + 140
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412132413214231342314321"
    
    # Generate classic superperm
    word = _classic_superperm(n)
    
    if n < 7:
        return word
        
    # For n=7, try to improve
    num_perms = factorial(n)
    
    def is_valid_superperm(s):
        if len(s) < n:
            return False
        seen = set()
        for i in range(len(s) - n + 1):
            window = tuple(s[i:i + n])
            seen.add(window)
            if len(seen) == num_perms:
                return True
        return len(seen) == num_perms

    # Greedy Removal
    current = word
    changed = True
    while changed:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_word = current[:i] + current[i+1:]
            if is_valid_superperm(new_word):
                current = new_word
                changed = True
                break
        if time.time() > deadline:
            break
            
    return current
