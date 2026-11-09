import time
from collections import defaultdict

def construct(n):
    """
    Constructs a superpermutation for n symbols using the classic recursive
    construction, then attempts local shortening to reduce length.
    """
    if n <= 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Classic recursive construction
    word = _build_recursive(n)
    
    # Local shortening pass using rolling hash for fast validity checks
    if time.time() < deadline - 5:
        word = _local_shorten_fast(word, n, deadline)
    
    return "".join(map(str, word))


def _build_recursive(n):
    """Classic recursive superpermutation of length 1!+2!+...+n!"""
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
            for o in range(min(len(new), len(block)), 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _local_shorten_fast(word, n, deadline):
    """
    Attempt to shorten the superpermutation by removing redundant symbols.
    Uses rolling hash for fast validity checks.
    """
    word = list(word)
    
    # Precompute all valid permutation hashes
    base = n + 1
    powb = base ** (n - 1)
    
    def compute_hash(window):
        h = 0
        for x in window:
            h = h * base + x
        return h
    
    # Build set of all valid permutation hashes
    from itertools import permutations
    target_hashes = set()
    for p in permutations(range(1, n + 1)):
        target_hashes.add(compute_hash(p))
    
    def roll_counts(w):
        """Count occurrences of each permutation hash in word w."""
        if len(w) < n:
            return {}
        h = 0
        for x in w[:n]:
            h = h * base + x
        counts = {h: 1}
        for i in range(n, len(w)):
            h = (h - w[i - n] * powb) * base + w[i]
            counts[h] = counts.get(h, 0) + 1
        return counts
    
    def is_valid(counts):
        """Check if all target permutation hashes are present."""
        for h in target_hashes:
            if h not in counts or counts[h] <= 0:
                return False
        return True
    
    # Initial validity check
    initial_counts = roll_counts(word)
    if not is_valid(initial_counts):
        return word  # Should not happen, but safety
    
    best = word
    best_len = len(word)
    
    # Multi-pass greedy deletion
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        counts = roll_counts(best)
        
        # Try deletions from left to right
        for i in range(len(best) - n + 1):
            if time.time() > deadline - 3:
                break
            
            # Check if removing position i would invalidate any permutation
            # We need to check all windows that include position i
            # A window at position j includes position i if j <= i <= j+n-1
            # So j ranges from max(0, i-n+1) to min(i, len(best)-n)
            
            j_start = max(0, i - n + 1)
            j_end = min(i, len(best) - n)
            
            can_remove = True
            for j in range(j_start, j_end + 1):
                window = best[j:j + n]
                h = compute_hash(window)
                if h in target_hashes:
                    if counts.get(h, 0) <= 1:
                        can_remove = False
                        break
            
            if can_remove:
                # Try the deletion
                cand = best[:i] + best[i + 1:]
                cand_counts = roll_counts(cand)
                if is_valid(cand_counts):
                    best = cand
                    best_len = len(best)
                    improved = True
                    break  # Restart scan with the new (shorter) word
    
    return best
