import time

def construct(n):
    """
    Constructs a superpermutation for n symbols.
    Uses the classic recursive construction which yields length sum_{k=1}^n k!.
    For n=7, length is 5913.
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 120s. We aim to finish in < 110s.
    deadline = start_time + 110.0

    # Base case
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # 123412314213241214312134121432132413214321
        # Actually, let's just run the algorithm for n=4, it's fast.
        pass

    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we run out of time, return what we have if it's valid?
            # The partial word is not a valid superpermutation for n.
            # So we must complete it. For n=7, it takes < 0.1s.
            # So this branch is effectively never taken for n<=7.
            pass
        
        perms = []
        seen = set()
        
        # Extract unique (m-1)-permutations in order of appearance
        # The window size is m-1
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            # Check if it contains all distinct elements 1..m-1
            # Since the previous word was a superpermutation of 1..m-1,
            # any window of size m-1 with distinct elements is a permutation.
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            # Block is p + [m] + p
            # e.g., if p=(1,2), m=3, block is 1,2,3,1,2
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap
            # The suffix of new_word must match the prefix of block.
            # In the classic construction, the overlap is typically len(p)+1?
            # Let's just do the general check.
            max_overlap = 0
            # Optimization: The overlap is at most len(block)
            # And at most len(new_word)
            limit = min(len(new_word), len(block))
            
            # We can optimize this check.
            # The first element of the block is p[0].
            # The last element of new_word is the last element of the previous block's suffix.
            # Previous block was p_prev + [m] + p_prev.
            # So it ends with p_prev.
            # The new block starts with p.
            # The overlap is the longest suffix of p_prev that is a prefix of p.
            
            # However, doing a full comparison is safer and still fast for n=7.
            found = False
            for o in range(limit, -1, -1):
                if o == 0:
                    new_word.extend(block)
                    found = True
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    new_word.extend(block[o:])
                    found = True
                    break
            
            if not found:
                new_word.extend(block)
                
        word = new_word
        
    return "".join(map(str, word))
