import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123121321412314123214121431214231214321"
    
    # General recursive construction
    # Start with S_1
    word = "1"
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we run out of time, we must return a valid superpermutation.
            # The current 'word' is a valid superpermutation for m-1 symbols.
            # It is NOT a valid superpermutation for m symbols.
            # However, for n <= 7, the construction is very fast.
            # S_7 construction:
            # m=2: 1 step
            # m=3: 2 steps
            # m=4: 6 steps
            # m=5: 24 steps
            # m=6: 120 steps
            # m=7: 720 steps
            # Total steps: 873.
            # Each step involves string manipulation on strings of length < 6000.
            # This should take < 1 second.
            # So we should never hit this deadline for n <= 7.
            # But to be safe, if we do, we can't return an invalid word.
            # We can try to complete the current step quickly? No, that's risky.
            # Better to assume it finishes.
            pass

        # Extract unique (m-1) permutations in order of appearance
        perms = []
        seen = set()
        # word is a string of length L
        # We look for windows of length m-1
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            # Check if all chars are unique
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            # Block is P + str(m) + P
            block = p + str(m) + p
            
            # Find max overlap
            # We want to append block to new_word
            # Check suffix of new_word vs prefix of block
            max_overlap = 0
            # Theoretical max overlap is len(block)-1 or len(new_word)-1
            # But since block starts with p and new_word ends with previous p,
            # overlap is usually small.
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word += block[max_overlap:]
        
        word = new_word
        
    return word
