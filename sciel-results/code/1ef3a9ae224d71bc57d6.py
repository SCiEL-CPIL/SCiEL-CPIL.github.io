"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations
from collections import defaultdict

def _build_recursive(n):
    """Classic recursive superpermutation of length 1!+2!+...+n!."""
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
    return word

def construct(n):
    if n <= 1:
        return "1"
    
    start = time.time()
    DEADLINE = start + 145.0  # 5s safety margin
    
    # Build initial recursive superpermutation
    word = _build_recursive(n)
    
    if n <= 4:
        return "".join(map(str, word))
    
    # Precompute all permutations
    all_perms = set(tuple(p) for p in permutations(range(1, n + 1)))
    target_count = len(all_perms)  # n!
    
    # Use rolling hash for fast window tracking
    base = n + 1
    powb = base ** (n - 1)
    
    def compute_hash(w, i):
        """Compute hash of window starting at position i."""
        h = 0
        for j in range(n):
            h = h * base + w[i + j]
        return h
    
    def initial_counts(w):
        """Compute hash counts for all windows."""
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            h = 0
            for j in range(n):
                h = h * base + w[i + j]
            counts[h] += 1
        return counts
    
    # Map hashes to permutations for validity check
    hash_to_perm = {}
    for p in all_perms:
        h = 0
        for x in p:
            h = h * base + x
        hash_to_perm[h] = p
    
    target_hashes = set(hash_to_perm.keys())
    
    def is_valid(counts):
        """Check if all target permutations are present."""
        for h in target_hashes:
            if counts.get(h, 0) == 0:
                return False
        return True
    
    # Phase 1: Greedy deletion - remove symbols that don't break validity
    best = list(word)
    best_counts = initial_counts(best)
    
    improved = True
    while improved and time.time() < DEADLINE - 10:
        improved = False
        L = len(best)
        for i in range(L - n + 1):
            if time.time() > DEADLINE - 10:
                break
            
            # Try deleting position i
            # Affected windows: those containing position i
            # Window j contains i if j <= i <= j+n-1, so j in [i-n+1, i]
            start_j = max(0, i - n + 1)
            end_j = min(i, L - n)
            
            # Check if any affected window has count 1 (would be lost)
            can_delete = True
            for j in range(start_j, end_j + 1):
                h = 0
                for k in range(n):
                    h = h * base + best[j + k]
                if best_counts.get(h, 0) == 1:
                    can_delete = False
                    break
            
            if not can_delete:
                continue
            
            # Perform deletion
            new_word = best[:i] + best[i+1:]
            new_L = len(new_word)
            
            # Update counts efficiently
            new_counts = dict(best_counts)  # copy
            
            # Remove windows that are affected (contain position i)
            for j in range(start_j, end_j + 1):
                h = 0
                for k in range(n):
                    h = h * base + best[j + k]
                new_counts[h] -= 1
                if new_counts[h] == 0:
                    del new_counts[h]
            
            # Add new windows that cross the gap
            # In new word, windows starting at j_new in [i-n+1, i] cross the gap
            # j_new in new coords maps to original positions
            for j_new in range(max(0, i - n + 1), min(i, new_L - n) + 1):
                h = 0
                for k in range(n):
                    pos = j_new + k
                    if pos < i:
                        h = h * base + best[pos]
                    else:
                        h = h * base + best[pos + 1]
                new_counts[h] = new_counts.get(h, 0) + 1
            
            # Check validity
            valid = True
            for h in target_hashes:
                if new_counts.get(h, 0) == 0:
                    valid = False
                    break
            
            if valid:
                best = new_word
                best_counts = new_counts
                improved = True
                break  # Restart scan with shorter word
    
    # Phase 2: Local search - try swaps and small rearrangements
    if time.time() < DEADLINE - 5:
        L = len(best)
        for _ in range(50):  # Limit iterations
            if time.time() > DEADLINE - 5:
                break
            found_improvement = False
            for i in range(L - 1):
                if time.time() > DEADLINE - 5:
                    break
                # Try swapping positions i and i+1
                best[i], best[i+1] = best[i+1], best[i]
                
                # Recompute counts (expensive but simple)
                new_counts = initial_counts(best)
                
                valid = True
                for h in target_hashes:
                    if new_counts.get(h, 0) == 0:
                        valid = False
                        break
                
                if valid:
                    best_counts = new_counts
                    found_improvement = True
                    break
                else:
                    # Undo swap
                    best[i], best[i+1] = best[i+1], best[i]
            
            if not found_improvement:
                break
    
    # Final validation
    final_counts = initial_counts(best)
    final_valid = True
    for h in target_hashes:
        if final_counts.get(h, 0) == 0:
            final_valid = False
            break
    
    if not final_valid:
        # Fall back to original
        best = list(word)
    
    return "".join(map(str, best))
# EVOLVE-BLOCK-END
