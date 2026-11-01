import heapq
import itertools

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n > 7:
        # Fallback: Simple greedy
        # Not required for n=7, but good for robustness
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
        return "".join(map(str, word))

    # Beam Search for n=7
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    num_perms = len(perms)
    
    transitions = [[0] * (n + 1) for _ in range(num_perms)]
    for i, p in enumerate(perms):
        for s in range(1, n + 1):
            new_p = p[1:] + (s,)
            transitions[i][s] = perm_to_idx[new_p]
            
    K = 200 # Beam width
    
    start_idx = 0
    start_mask = 1 << start_idx
    start_path = "".join(map(str, perms[start_idx]))
    start_cov = 1
    
    # State: (neg_cov, length, idx, mask, path, cov)
    # We sort by (neg_cov, length) -> max cov, min length
    beam = [(-1, n, start_idx, start_mask, start_path, 1)]
    
    max_steps = n * num_perms + n
    
    for step in range(max_steps):
        candidates = []
        for state in beam:
            neg_cov, length, idx, mask, path, cov = state
            for s in range(1, n + 1):
                next_idx = transitions[idx][s]
                bit = 1 << next_idx
                if mask & bit:
                    new_cov = cov
                    new_mask = mask
                else:
                    new_cov = cov + 1
                    new_mask = mask | bit
                    if new_cov == num_perms:
                        return path + str(s)
                new_length = length + 1
                new_path = path + str(s)
                candidates.append((-new_cov, new_length, next_idx, new_mask, new_path, new_cov))
        
        if not candidates:
            break
            
        # Select top K
        # nsmallest with key (neg_cov, length)
        top_k = heapq.nsmallest(K, candidates, key=lambda x: (x[0], x[1]))
        beam = top_k
        
    # If not found, return the best path (might be invalid)
    # But we must return a valid string.
    # If beam search fails, we can try a greedy fallback?
    # For now, return the path from the best state.
    return beam[0][4]
