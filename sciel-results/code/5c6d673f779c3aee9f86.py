import time
from itertools import permutations
import sys
import random

def construct(n):
    # Safety margin: 150s limit, aim to finish by 145s
    deadline = time.time() + 145

    if n <= 6:
        # Standard construction is optimal for n<=6? 
        # n=1: 1
        # n=2: 121 (len 3) vs 121 (len 3)
        # n=3: 121321 (len 6) vs 1213121 (len 7)? 
        # Standard recursive: 
        # 1
        # 121
        # 121321
        # This is optimal for n<=6.
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
                best_overlap = 0
                max_check = min(len(new), len(block))
                for o in range(max_check, 0, -1):
                    if new[-o:] == block[:o]:
                        best_overlap = o
                        break
                new += block[best_overlap:]
            word = new
        return "".join(map(str, word))

    # For n=7, we use randomized initialization and local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    def generate_superperm(seed):
        random.seed(seed)
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Shuffle perms to get different structures
            random.shuffle(perms)
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                best_overlap = 0
                max_check = min(len(new), len(block))
                for o in range(max_check, 0, -1):
                    if new[-o:] == block[:o]:
                        best_overlap = o
                        break
                new += block[best_overlap:]
            word = new
        return word

    def optimize(word):
        counts = [0] * num_perms
        L = len(word)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
        
        if any(c == 0 for c in counts):
            return word, len(word)

        def try_delete(i):
            L_curr = len(word)
            if L_curr <= n:
                return False
            
            old_pids = []
            start_min = max(0, i - n + 1)
            start_max = min(i, L_curr - n)
            
            for k in range(start_min, start_max + 1):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    counts[pid] -= 1
                    old_pids.append(pid)
            
            new_starts_min = max(0, i - n + 1)
            new_starts_max = min(i - 1, L_curr - 1 - n)
            
            new_pids = []
            for k in range(new_starts_min, new_starts_max + 1):
                w1 = word[k:i]
                w2 = word[i+1:k+n]
                p = tuple(w1 + w2)
                pid = perm_to_id.get(p)
                if pid is not None:
                    counts[pid] += 1
                    new_pids.append(pid)

            is_valid = True
            for pid in old_pids:
                if counts[pid] == 0:
                    is_valid = False
                    break

            if is_valid:
                word.pop(i)
                return True
            else:
                for pid in old_pids:
                    counts[pid] += 1
                for pid in new_pids:
                    counts[pid] -= 1
                return False

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            L_curr = len(word)
            indices = list(range(L_curr))
            random.shuffle(indices)
            for i in indices:
                if try_delete(i):
                    improved = True
                    L_curr -= 1
        return word, len(word)

    best_word = None
    best_len = float('inf')
    
    # Try a few random seeds
    # Time budget: 145s. Let's say we spend 20s per seed, 7 seeds.
    # Or just run until time is up, updating best.
    seed = 0
    while time.time() < deadline - 10:
        start_time = time.time()
        w = generate_superperm(seed)
        opt_w, opt_len = optimize(w)
        
        if opt_len < best_len:
            best_len = opt_len
            best_word = opt_w
            
        seed += 1
        # If we ran out of time, break
        if time.time() > deadline - 10:
            break

    if best_word is None:
        # Fallback
        w = generate_superperm(0)
        best_word, best_len = optimize(w)

    return "".join(map(str, best_word))

if __name__ == "__main__":
    pass
