import itertools
import random

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    all_perms = list(itertools.permutations(range(1, n + 1)))
    best_word = None
    best_len = float('inf')
    
    # Use a few iterations of randomized greedy to find a shorter superpermutation
    # 10 iterations is a good balance for n=7
    num_iterations = 10
    
    for i in range(num_iterations):
        random.seed(i)
        
        current_word = list(all_perms[0])
        seen = set([all_perms[0]])
        unseen = all_perms[1:]
        random.shuffle(unseen)
        
        while unseen:
            tail_len = n - 1
            if len(current_word) >= tail_len:
                tail = tuple(current_word[-tail_len:])
            else:
                tail = tuple(current_word)
            
            best_overlap = -1
            best_perm = None
            best_idx = -1
            
            for idx, p in enumerate(unseen):
                found_overlap = 0
                for k in range(n - 1, 0, -1):
                    if p[0:k] == tail[-k:]:
                        found_overlap = k
                        break
                
                if found_overlap > best_overlap:
                    best_overlap = found_overlap
                    best_perm = p
                    best_idx = idx
                    if best_overlap == n - 1:
                        break
            
            if best_perm is None:
                best_perm = unseen[0]
                best_overlap = 0
                best_idx = 0
            
            unseen[best_idx] = unseen[-1]
            unseen.pop()
            
            seen.add(best_perm)
            
            current_word.extend(best_perm[best_overlap:])
        
        if len(current_word) < best_len:
            best_len = len(current_word)
            best_word = "".join(map(str, current_word))
            
    return best_word
