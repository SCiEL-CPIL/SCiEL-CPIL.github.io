import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Generate the naive superpermutation
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Precompute all permutations
    all_perms = set(itertools.permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Track permutation counts in current word
    perm_count = defaultdict(int)
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_count[p] += 1
    
    # Set of "lost" permutations (count == 0)
    lost = set()
    for p in all_perms:
        if perm_count.get(p, 0) == 0:
            lost.add(p)
    
    def compute_old_windows(i):
        """Compute windows in old word that cover position i."""
        L = len(word)
        if L <= n:
            return []
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        return [tuple(word[j:j+n]) for j in range(start_j, end_j + 1)]
    
    def compute_new_windows(i):
        """Compute windows in new word (after deleting i) that cover the junction."""
        L = len(word)
        if L <= n:
            return []
        start_j_new = max(0, i - n + 1)
        end_j_new = min(i, (L - 1) - n)
        new_windows = []
        for j in range(start_j_new, end_j_new + 1):
            new_w = []
            for k in range(n):
                idx = j + k
                if idx < i:
                    new_w.append(word[idx])
                else:
                    new_w.append(word[idx + 1])
            new_windows.append(tuple(new_w))
        return new_windows
    
    def try_delete(i):
        """Attempt to delete word[i]. Returns True if successful."""
        L = len(word)
        if L <= n:
            return False
        
        old_windows = compute_old_windows(i)
        new_windows = compute_new_windows(i)
        
        # Tentatively apply
        for p in old_windows:
            perm_count[p] -= 1
            if perm_count[p] == 0:
                lost.add(p)
        
        for p in new_windows:
            if p in lost:
                lost.remove(p)
            perm_count[p] += 1
        
        if len(lost) > 0:
            # Revert
            for p in new_windows:
                perm_count[p] -= 1
                if perm_count[p] == 0:
                    lost.add(p)
            for p in old_windows:
                if p in lost:
                    lost.remove(p)
                perm_count[p] += 1
            return False
        else:
            word.pop(i)
            return True
    
    def try_insert(pos, val):
        """Insert val at position pos. Returns True if it doesn't create new lost perms."""
        L = len(word)
        
        # Compute windows in old word that are affected
        # After insertion, positions >= pos shift right by 1
        # Windows that include the new position or are shifted
        old_affected = []
        # Windows that will be shifted: those starting from max(0, pos-n+1) to min(pos-1, L-n)
        for j in range(max(0, pos - n + 1), min(pos, L - n) + 1):
            old_affected.append(tuple(word[j:j+n]))
        
        # New windows after insertion
        new_affected = []
        new_L = L + 1
        for j in range(max(0, pos - n + 1), min(pos, new_L - n) + 1):
            new_w = []
            for k in range(n):
                idx = j + k
                if idx < pos:
                    new_w.append(word[idx])
                elif idx == pos:
                    new_w.append(val)
                else:
                    new_w.append(word[idx - 1])
            new_affected.append(tuple(new_w))
        
        # Check if insertion creates any new lost permutations
        would_create_lost = False
        for p in new_affected:
            if p not in perm_count or perm_count[p] == 0:
                if p not in lost:
                    # This would be a new window that wasn't there before
                    # Check if it's a valid permutation
                    if len(set(p)) == n and all(1 <= x <= n for x in p):
                        # It's a new permutation window
                        # This is fine, we're adding coverage
                        pass
                    else:
                        # Invalid window, but we don't care about those
                        pass
                else:
                    # p is in lost, meaning it had count 0
                    # After insertion, if p appears in new_affected, it might get count > 0
                    # But we need to check if it actually appears
                    pass
        
        # Simpler approach: just tentatively insert and check
        word.insert(pos, val)
        
        # Recompute affected windows
        for p in old_affected:
            perm_count[p] -= 1
            if perm_count[p] == 0:
                lost.add(p)
        
        for p in new_affected:
            if p in lost:
                lost.remove(p)
            perm_count[p] += 1
        
        if len(lost) > 0:
            # Revert
            for p in new_affected:
                perm_count[p] -= 1
                if perm_count[p] == 0:
                    lost.add(p)
            for p in old_affected:
                if p in lost:
                    lost.remove(p)
                perm_count[p] += 1
            word.pop(pos)
            return False
        else:
            return True
    
    # Main deletion loop with multiple strategies
    best_len = len(word)
    best_word = list(word)
    
    def update_best():
        nonlocal best_len, best_word
        if len(word) < best_len:
            best_len = len(word)
            best_word = list(word)
    
    # Phase 1: Aggressive backward deletion
    while time.time() < deadline - 30:
        progress = False
        L = len(word)
        
        for i in range(L - 1, -1, -1):
            if time.time() > deadline - 30:
                break
            if try_delete(i):
                progress = True
                update_best()
        
        if not progress:
            break
    
    # Phase 2: Forward deletion passes
    while time.time() < deadline - 20:
        progress = False
        L = len(word)
        
        for i in range(L):
            if time.time() > deadline - 20:
                break
            if try_delete(i):
                progress = True
                update_best()
        
        if not progress:
            break
    
    # Phase 3: Randomized deletion with occasional insertion-based repair
    while time.time() < deadline - 5:
        L = len(word)
        if L <= n:
            break
        
        # Try random deletions
        deleted = False
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            i = random.randint(0, L - 1)
            if try_delete(i):
                deleted = True
                L -= 1
                update_best()
        
        if deleted:
            continue
        
        # Try insertion-based repair: insert a random char, then try to delete more
        if L > n + 5:
            # Find a position where insertion might help
            # Try inserting a symbol that appears frequently nearby
            pos = random.randint(1, L - 2)
            # Choose a value from nearby context
            nearby = word[max(0, pos-3):min(L, pos+3)]
            if nearby:
                val = random.choice(nearby)
                if try_insert(pos, val):
                    # Now try to delete some positions
                    L_new = len(word)
                    for _ in range(20):
                        if time.time() > deadline - 5:
                            break
                        j = random.randint(0, L_new - 1)
                        if try_delete(j):
                            L_new -= 1
                            update_best()
                    if len(word) < best_len:
                        update_best()
                    # If we ended up longer, revert to best
                    if len(word) >= best_len:
                        word = best_word
                        perm_count = defaultdict(int)
                        for i in range(len(word) - n + 1):
                            p = tuple(word[i:i+n])
                            perm_count[p] += 1
                        lost = set()
                        for p in all_perms:
                            if perm_count.get(p, 0) == 0:
                                lost.add(p)
    
    # Final phase: try all positions again in random order
    if time.time() < deadline:
        L = len(word)
        indices = list(range(L))
        random.shuffle(indices)
        for i in indices:
            if time.time() > deadline:
                break
            if try_delete(i):
                update_best()
    
    # Ensure we return the best word found
    if len(word) < best_len:
        best_word = list(word)
    
    # Final validity check
    seen = set()
    for i in range(len(best_word) - n + 1):
        seen.add(tuple(best_word[i:i+n]))
    if len(seen) < total_perms:
        # Should not happen, but return what we have
        pass
    
    return "".join(map(str, best_word))
