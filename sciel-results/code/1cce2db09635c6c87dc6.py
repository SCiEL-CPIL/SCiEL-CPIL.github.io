import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Standard recursive construction of superpermutation
    # Length is sum_{k=1}^n k!
    # For n=7, length is 5913.
    
    if n == 1:
        return "1"
    
    word = [1]
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            # Return best effort, though construction is fast
            break
        
        perms, seen = [], set()
        # Extract distinct permutations of length m-1 in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        word = new_word
    
    return "".join(map(str, word))
