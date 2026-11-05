import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Generate standard recursive superpermutation
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
        
    # Convert to list of ints for easier manipulation
    S = list(word)
    
    # 2. Initialize counts
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
    
    best_S = S[:]
    best_len = len(S)
    target_perms = math.factorial(n)
    
    # 3. Local Search with multiple strategies
    def is_valid(S, n):
        """Check if S contains all permutations of 1..n"""
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i+n]))
        return len(seen) == target_perms
    
    def can_remove(S, counts, i, n):
        """Check if position i can be removed without losing any permutation"""
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)
        
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            if counts.get(p, 0) <= 1:
                return False
        return True
    
    def do_remove(S, counts, i, n):
        """Perform removal at position i and update counts"""
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)
        
        # Decrement old windows
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
        
        # Add new straddling windows
        new_len = len(S) - 1
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, new_len - n)
        
        for k in range(start_new, end_new + 1):
            window_chars = []
            for idx in range(k, k+n):
                s_idx = idx if idx < i else idx + 1
                window_chars.append(S[s_idx])
            p_new = tuple(window_chars)
            counts[p_new] = counts.get(p_new, 0) + 1
        
        S.pop(i)
    
    def try_remove_pass(S, counts, n, deadline):
        """Try to remove characters, return new S and counts if any removal was made"""
        removed = False
        i = len(S) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            
            if can_remove(S, counts, i, n):
                do_remove(S, counts, i, n)
                removed = True
                # Don't decrement i, since the next position is now at i
            else:
                i -= 1
        
        return S, counts, removed
    
    def rebuild_counts(S, n):
        """Rebuild counts from scratch"""
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts
    
    def smart_perturb(S, counts, n, deadline):
        """Apply a targeted perturbation to escape local minima"""
        if time.time() > deadline:
            return S, counts, False
        
        if len(S) <= n:
            return S, counts, False
        
        # Strategy 1: Find a position where removal is "almost" valid (count == 1 for some window)
        # and try swapping with a nearby position
        for _ in range(20):
            if time.time() > deadline:
                break
            
            i = random.randint(0, len(S) - 1)
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            # Check how "close" we are to being able to remove
            min_count = float('inf')
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                c = counts.get(p, 0)
                if c < min_count:
                    min_count = c
            
            if min_count <= 2:  # Close to removable
                # Try swapping with a nearby position
                j = i + random.choice([-2, -1, 1, 2])
                if 0 <= j < len(S) and j != i:
                    S[i], S[j] = S[j], S[i]
                    counts = rebuild_counts(S, n)
                    return S, counts, True
        
        # Strategy 2: Insert a character that might enable future removals
        if random.random() < 0.5 and len(S) < best_len * 1.15:
            pos = random.randint(0, len(S))
            # Choose a character that appears less frequently in nearby windows
            nearby_chars = S[max(0, pos-3):min(len(S), pos+3)]
            char_counts = {}
            for c in nearby_chars:
                char_counts[c] = char_counts.get(c, 0) + 1
            # Pick a character that's underrepresented locally
            candidates = [c for c in range(1, n+1) if char_counts.get(c, 0) <= 1]
            if candidates:
                char = random.choice(candidates)
            else:
                char = random.randint(1, n)
            S.insert(pos, char)
            counts = rebuild_counts(S, n)
            return S, counts, True
        
        # Strategy 3: Random swap as fallback
        i, j = random.sample(range(len(S)), 2)
        S[i], S[j] = S[j], S[i]
        counts = rebuild_counts(S, n)
        return S, counts, True
    
    # Multiple passes of removal with periodic perturbation
    max_passes = 100
    stuck_count = 0
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        S, counts, removed = try_remove_pass(S, counts, n, deadline)
        
        if removed:
            best_S = S[:]
            best_len = len(S)
            stuck_count = 0
        else:
            stuck_count += 1
            # Perturb every few stuck iterations
            if stuck_count >= 3:
                S, counts, _ = smart_perturb(S, counts, n, deadline)
                stuck_count = 0
    
    # Final validity check
    if not is_valid(best_S, n):
        # Fall back to the original construction if invalid
        return "".join(map(str, word))
    
    return "".join(map(str, best_S))
