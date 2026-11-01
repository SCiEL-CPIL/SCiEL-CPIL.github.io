import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 115s hard limit for internal logic to allow for return overhead

    # 1. Generate the naive superpermutation
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
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    best_word = word[:]
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
        
    # Helper to check validity using a set of tuples
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
        return len(seen) == total_perms

    # Fast validity check: returns True if valid, False otherwise
    # Uses early termination if we detect a problem
    def is_valid_fast(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            t = tuple(w[i:i+n])
            if t not in seen:
                seen.add(t)
                if len(seen) == total_perms:
                    # We've seen all permutations, but need to verify no duplicates
                    # Actually, if we see total_perms unique ones, it's valid
                    # But we need to make sure we don't have more than total_perms
                    pass
        return len(seen) == total_perms

    current_len = len(best_word)
    
    # Initial check
    if not is_valid(best_word):
        return "".join(map(str, best_word))

    # Count occurrences of each permutation to find redundant positions
    # A position i is "redundant" if removing it doesn't lose any permutation
    # More precisely: if for every permutation that contains position i,
    # there's another window containing the same permutation
    
    # Strategy: try to remove characters that are "redundant"
    # For each position, check if removing it keeps all permutations
    
    def try_remove(w, idx):
        """Try removing position idx, return new word if valid, else None"""
        if len(w) <= n:
            return None
        candidate = w[:idx] + w[idx+1:]
        return is_valid(candidate) and candidate or None

    # More efficient: check if a position is removable by examining affected windows
    def is_position_removable(w, idx):
        """Check if removing position idx preserves all permutations"""
        if idx < 0 or idx >= len(w):
            return False
        if len(w) <= n:
            return False
        
        # Windows affected by removal of position idx:
        # Windows starting from max(0, idx-n+1) to min(idx, len(w)-n)
        # After removal, these windows shift
        
        # Collect all permutations that cover position idx
        affected_start = max(0, idx - n + 1)
        affected_end = min(idx, len(w) - n)
        
        # Permutations that will be lost (only in windows containing idx)
        lost = set()
        for i in range(affected_start, affected_end + 1):
            t = tuple(w[i:i+n])
            lost.add(t)
        
        # After removal, check if each lost permutation still appears
        candidate = w[:idx] + w[idx+1:]
        for t in lost:
            # Check if t appears in candidate
            found = False
            for i in range(len(candidate) - n + 1):
                if tuple(candidate[i:i+n]) == t:
                    found = True
                    break
            if not found:
                return False
        return True

    # Local search loop
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Try to find a removable position
        # Start from random positions, but also try systematically
        if iteration % 10 == 0:
            # Systematic pass: try all positions
            for idx in range(len(best_word)):
                if is_position_removable(best_word, idx):
                    best_word = best_word[:idx] + best_word[idx+1:]
                    current_len = len(best_word)
                    break
        else:
            # Random sampling of positions
            for _ in range(50):
                idx = random.randint(0, len(best_word) - 1)
                if is_position_removable(best_word, idx):
                    best_word = best_word[:idx] + best_word[idx+1:]
                    current_len = len(best_word)
                    break
        
        # Occasionally try a swap or move to escape local optima
        if random.random() < 0.1 and len(best_word) > n + 1:
            candidate = best_word[:]
            idx = random.randint(0, len(candidate) - 2)
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if is_valid(candidate) and len(candidate) < current_len:
                best_word = candidate
                current_len = len(best_word)

    return "".join(map(str, best_word))
