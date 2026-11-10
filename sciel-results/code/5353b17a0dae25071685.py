import time
import math
import random

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Generate Classic Superpermutation
    word = generate_classic(n)
    target_count = math.factorial(n)
    
    # Precompute factorials for Lehmer code ranking
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

    # 2. Build Initial State
    L = len(word)
    nr = L - n + 1
    ranks = [0] * nr
    cov = [0] * target_count
    
    for i in range(nr):
        r = perm_rank(word[i:i + n])
        ranks[i] = r
        cov[r] += 1
        
    best_len = L
    best_word = word[:]

    # 3. Local Search
    # We will try to delete characters.
    # To avoid getting stuck, we randomize the order of checks.
    
    def try_deletion(w, i, rk, cov):
        Lw = len(w)
        nr_w = Lw - n + 1
        if i < 0 or i >= Lw:
            return None
        
        lo = max(0, i - (n - 1))
        hi = min(i, nr_w - 1)
        
        # Save old ranks
        old_ranks = [rk[j] for j in range(lo, hi + 1)]
        
        # Decrement cov
        for r in old_ranks:
            cov[r] -= 1
            if cov[r] < 0:
                return None # Should not happen if state is consistent
        
        # Check if any became 0
        for r in old_ranks:
            if cov[r] == 0:
                # Revert
                for r2 in old_ranks:
                    cov[r2] += 1
                return None

        # Compute new ranks for affected windows
        # New word: w[:i] + w[i+1:]
        new_w = w[:i] + w[i+1:]
        new_nr = nr_w - 1
        new_rk = [0] * new_nr
        
        # Copy unaffected
        for j in range(lo):
            new_rk[j] = rk[j]
        for j in range(hi + 1, new_nr):
            new_rk[j] = rk[j + 1]
            
        # Recompute affected
        for j in range(lo, hi + 1):
            new_rk[j] = perm_rank(new_w[j:j + n])
            
        # Increment cov for new ranks
        for j in range(lo, hi + 1):
            cov[new_rk[j]] += 1
            
        return new_w, new_rk, cov

    # Main Search Loop
    # Strategy: 
    # 1. Try random deletions.
    # 2. If stuck, try random swaps of adjacent elements (to change structure).
    # 3. If stuck, restart with a perturbed best_word.
    
    attempts = 0
    stuck_counter = 0
    
    while time.time() < deadline:
        L = len(word)
        
        # Try to find a valid deletion
        found = False
        # Random order of indices
        indices = list(range(L))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            res = try_deletion(word, i, ranks, cov)
            if res is not None:
                word, ranks, cov = res
                found = True
                stuck_counter = 0
                if len(word) < best_len:
                    best_len = len(word)
                    best_word = word[:]
                break
        
        if not found:
            stuck_counter += 1
            
            # If stuck for a while, try a swap to escape local minimum
            if stuck_counter > 50:
                # Try a random swap of two elements
                # This is risky and expensive to check, but necessary for progress
                # We will try to swap w[a] and w[b]
                # For simplicity, try adjacent swaps first
                a = random.randint(0, L - 2)
                b = a + 1
                
                # Check if swap is valid
                # This requires checking windows in [a-n+1, b]
                # Let's implement a quick check for swap
                # If it's too slow, we skip to restart
                
                # For now, just restart with a slight perturbation
                # Perturb: Insert a random char? No, must be valid.
                # Let's just keep trying deletions with different random seeds
                # Or restart from best_word
                if stuck_counter > 200:
                    word = best_word[:]
                    # Rebuild state
                    L = len(word)
                    nr = L - n + 1
                    ranks = [0] * nr
                    cov = [0] * target_count
                    for i in range(nr):
                        r = perm_rank(word[i:i + n])
                        ranks[i] = r
                        cov[r] += 1
                    stuck_counter = 0

    return "".join(map(str, best_word))

def generate_classic(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word
