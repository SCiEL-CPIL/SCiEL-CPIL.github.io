import time
import random
from itertools import permutations

def construct(n):
    # Generate the classic superpermutation
    word = generate_classic(n)
    
    # For n < 5, classic is optimal
    if n < 5:
        return "".join(map(str, word))
    
    # For n=5,6,7, try to optimize
    if n <= 7:
        # Precompute target set for validity check
        target_set = set()
        base = 1
        pow_n = [1] * (n + 1)
        for i in range(1, n + 1):
            pow_n[i] = pow_n[i-1] * n
            
        for p in permutations(range(1, n + 1)):
            val = 0
            for j in range(n):
                val += (p[j] - 1) * pow_n[j]
            target_set.add(val)
            
        word = optimize_superpermutation(word, n, target_set, pow_n, time_limit=110)
        
    return "".join(map(str, word))

def generate_classic(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract permutations of length m-1
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
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def is_valid_slide(word, n, target_set, pow_n):
    if len(word) < n:
        return False
    
    # Calculate first window hash
    curr_hash = 0
    for j in range(n):
        curr_hash += (word[j] - 1) * pow_n[j]
        
    found = set()
    found.add(curr_hash)
    if len(found) == len(target_set):
        return True
        
    for i in range(1, len(word) - n + 1):
        # Remove leftmost, add rightmost
        curr_hash = (curr_hash - (word[i-1] - 1)) // n + (word[i+n-1] - 1) * pow_n[n-1]
        found.add(curr_hash)
        if len(found) == len(target_set):
            return True
    return False

def optimize_superpermutation(word, n, target_set, pow_n, time_limit=100):
    start_time = time.time()
    best_word = word
    best_len = len(word)
    
    while time.time() - start_time < time_limit:
        current_word = best_word[:]
        current_len = best_len
        improved = False
        attempts = 0
        # Dynamic max attempts? Just run until time limit or improvement
        # But we need to check time inside the loop
        
        while not improved:
            if time.time() - start_time >= time_limit:
                break
                
            idx = random.randint(0, current_len - 1)
            # Try removing
            new_word = current_word[:idx] + current_word[idx+1:]
            if is_valid_slide(new_word, n, target_set, pow_n):
                current_word = new_word
                current_len -= 1
                improved = True
            attempts += 1
            
            # If we've tried a lot without success, break to restart outer loop
            # This prevents getting stuck in a bad random sequence
            if attempts > 50000:
                break
                
        if current_len < best_len:
            best_word = current_word
            best_len = current_len
            # If we found an improvement, we might want to restart the search from this new state
            # The loop naturally does this by resetting current_word to best_word at the top
            
    return best_word
