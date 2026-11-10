import time
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit

    # Step 1: Generate classic superpermutation
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 5:
        return "".join(map(str, word))
        
    # Step 2: Greedy Deletion with Rank Tracking
    target_perms = math.factorial(n)
    
    # Precompute factorials for Lehmer code
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i

    def perm_rank(p):
        rank = 0
        used = [False] * (n + 1)
        for i in range(n):
            count = 0
            for j in range(1, p[i]):
                if not used[j]:
                    count += 1
            rank += count * fact[n - 1 - i]
            used[p[i]] = True
        return rank

    L = len(word)
    nr = L - n + 1
    
    # Initialize ranks and coverage counts
    ranks = [0] * nr
    cov = [0] * target_perms
    
    for i in range(nr):
        r = perm_rank(word[i:i+n])
        ranks[i] = r
        cov[r] += 1
        
    # Deletion Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(word):
            if time.time() > deadline:
                break
            
            L = len(word)
            nr = L - n + 1
            
            # Affected windows: start k in [max(0, i-n+1), min(i, nr-1)]
            lo = max(0, i - n + 1)
            hi = min(i, nr - 1)
            
            if lo > hi:
                i += 1
                continue
                
            # Save old ranks of affected windows
            old_ranks = [ranks[j] for j in range(lo, hi + 1)]
            
            # Decrement coverage for old ranks
            for r in old_ranks:
                cov[r] -= 1
            
            # Compute new ranks
            new_ranks = [0] * (hi - lo + 1)
            new_valid_ranks = []
            
            for k in range(lo, hi + 1):
                w1 = word[k:i]
                w2 = word[i+1:k+n]
                new_p = w1 + w2
                if len(set(new_p)) == n:
                    new_r = perm_rank(new_p)
                    new_ranks[k - lo] = new_r
                    new_valid_ranks.append(new_r)
                    cov[new_r] += 1
                else:
                    new_ranks[k - lo] = -1
            
            # Check validity
            is_valid = True
            for r in old_ranks:
                if cov[r] == 0:
                    is_valid = False
                    break
            
            if is_valid:
                # Perform deletion
                word = word[:i] + word[i+1:]
                
                # Update ranks array
                new_nr = nr - 1
                new_all_ranks = [0] * new_nr
                for j in range(lo):
                    new_all_ranks[j] = ranks[j]
                for j in range(lo, hi + 1):
                    new_all_ranks[j] = new_ranks[j - lo]
                for j in range(hi + 1, new_nr):
                    new_all_ranks[j] = ranks[j + 1]
                
                ranks = new_all_ranks
                improved = True
            else:
                # Revert changes
                for r in old_ranks:
                    cov[r] += 1
                for r in new_valid_ranks:
                    cov[r] -= 1
                i += 1
                
            if time.time() > deadline:
                break
                
    return "".join(map(str, word))
