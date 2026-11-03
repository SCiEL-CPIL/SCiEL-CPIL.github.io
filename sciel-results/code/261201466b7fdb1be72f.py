import time
from itertools import permutations
import sys

def construct(n):
    # Safety margin: 150s limit, aim to finish by 145s
    deadline = time.time() + 145

    # 1. Generate standard superpermutation
    # Classic recursive construction: length 1! + 2! + ... + n!
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

    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for optimized local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts array
    counts = [0] * num_perms
    L = len(word)
    
    # Count initial permutations
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
            
    # Check if valid initially
    if any(c == 0 for c in counts):
        return "".join(map(str, word))

    def try_delete(i):
        """
        Attempts to delete word[i]. 
        Returns True if successful (word and counts updated), False otherwise.
        """
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

    # Local Search Loop with alternating directions and batched time checks
    improved = True
    direction = 1  # 1 = left-to-right, -1 = right-to-left
    pass_count = 0
    while improved and time.time() < deadline:
        improved = False
        L_curr = len(word)
        
        if direction == 1:
            i = 0
            while i < L_curr:
                if (i % 50 == 0) and time.time() > deadline - 5:
                    break
                if try_delete(i):
                    improved = True
                    L_curr -= 1
                else:
                    i += 1
        else:
            i = L_curr - 1
            while i >= 0:
                if (i % 50 == 0) and time.time() > deadline - 5:
                    break
                if try_delete(i):
                    improved = True
                    L_curr -= 1
                else:
                    i -= 1
        
        direction = -direction
        pass_count += 1

    return "".join(map(str, word))

if __name__ == "__main__":
    pass
