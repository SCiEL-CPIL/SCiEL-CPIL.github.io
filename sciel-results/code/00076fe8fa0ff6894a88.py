import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate standard superpermutation for n-1
    # For n=7, we need W_6.
    # W_6 standard is 873.
    # We can generate W_6 using the same recursive method.
    
    def generate_standard(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            # Extract unique permutations of length m-1 from word
            # Optimization: use a set to track seen
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap
                # new_word ends with some suffix, block starts with some prefix
                # Max overlap k such that new_word[-k:] == block[:k]
                # k can be at most min(len(new_word), len(block))
                max_k = min(len(new_word), len(block))
                overlap = 0
                # Check from largest possible overlap downwards
                # Since n is small, this is fast
                for k in range(max_k, -1, -1):
                    if k == 0:
                        overlap = 0
                        break
                    if new_word[len(new_word) - k:] == block[:k]:
                        overlap = k
                        break
                new_word += block[overlap:]
            word = new_word
        return word

    # Get W_{n-1}
    w_prev = generate_standard(n - 1)
    
    # Now extend to n
    # We extract permutations of 1..n-1 from w_prev
    perms, seen = [], set()
    for i in range(len(w_prev) - (n - 1) + 1):
        window = tuple(w_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    # Standard order is the order of appearance in w_prev
    # Let's try a few orders if n is large enough to allow search
    # For n=7, len(perms) is 720.
    
    def build_word(perms, n):
        new_word = []
        for p in perms:
            block = list(p) + [n] + list(p)
            max_k = min(len(new_word), len(block))
            overlap = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - k:] == block[:k]:
                    overlap = k
                    break
            new_word += block[overlap:]
        return new_word

    # 1. Try standard order
    word_std = build_word(perms, n)
    best_word = word_std
    best_len = len(word_std)
    
    # Check time
    if time.time() > deadline:
        return "".join(map(str, best_word))
        
    # 2. Try lexicographic order
    perms_lex = sorted(perms)
    if perms_lex != perms:
        word_lex = build_word(perms_lex, n)
        if len(word_lex) < best_len:
            best_word = word_lex
            best_len = len(word_lex)
            
    # Check time
    if time.time() > deadline:
        return "".join(map(str, best_word))
        
    # 3. Try reverse lexicographic
    perms_rev = sorted(perms, reverse=True)
    if perms_rev != perms:
        word_rev = build_word(perms_rev, n)
        if len(word_rev) < best_len:
            best_word = word_rev
            best_len = len(word_rev)
            
    # Check time
    if time.time() > deadline:
        return "".join(map(str, best_word))
        
    # 4. Randomized search / Hill Climbing?
    # If we have time, we can try to swap adjacent permutations in the list and see if length decreases.
    # This is expensive. 720 items. Swaps are local.
    # Let's do a few random restarts with greedy construction?
    # Or just a few random shuffles?
    
    import random
    random.seed(42)
    
    # Try a few random shuffles
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        perms_rand = perms.copy()
        random.shuffle(perms_rand)
        word_rand = build_word(perms_rand, n)
        if len(word_rand) < best_len:
            best_word = word_rand
            best_len = len(word_rand)
            
    return "".join(map(str, best_word))
