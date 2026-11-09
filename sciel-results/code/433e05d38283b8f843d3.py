"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import permutations as _perms
from collections import Counter

# EVOLVE-BLOCK-START
def construct(n):
    deadline = time.time() + 110.0  # safety margin before 120s limit
    word = _classic_construct(n)
    if len(word) <= 1:
        return "".join(map(str, word))
    
    # Precompute: for each position, the list of window-start indices that include it
    # A window starting at i covers positions i..i+n-1
    # Position pos is included in windows starting at i where i <= pos <= i+n-1
    # i.e., i in [pos-n+1, pos], clamped to [0, len(word)-n]
    
    L = len(word)
    # windows_for_pos[pos] = list of window start indices that include position pos
    windows_for_pos = [[] for _ in range(L)]
    for i in range(L - n + 1):
        # Window starting at i covers positions i to i+n-1
        for pos in range(i, min(i + n, L)):
            windows_for_pos[pos].append(i)
    
    # Count occurrences of each permutation in the current word
    perm_count = Counter()
    for i in range(L - n + 1):
        perm_count[tuple(word[i:i+n])] += 1
    
    # Track which permutations are "critical" (count == 1)
    # Removing a position that is the only occurrence of a permutation is invalid
    
    removed = [False] * L
    # We'll process positions in a random-ish order to find more removable positions
    # Try multiple passes
    
    for pass_num in range(3):
        if time.time() > deadline:
            break
        # Try positions in reverse order (end to start)
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
            if removed[pos]:
                continue
            # Check if removing pos is valid
            # All windows that include pos must have count > 1
            valid = True
            for wi in windows_for_pos[pos]:
                if removed[wi] or removed[wi + n - 1]:
                    # Window already partially removed? Shouldn't happen
                    # since we only remove positions, not windows
                    continue
                w_tuple = tuple(word[wi:wi+n])
                if perm_count[w_tuple] <= 1:
                    valid = False
                    break
            if valid:
                # Remove this position
                removed[pos] = True
                # Update perm_count for all windows that include pos
                for wi in windows_for_pos[pos]:
                    w_tuple = tuple(word[wi:wi+n])
                    perm_count[w_tuple] -= 1
                    if perm_count[w_tuple] == 0:
                        # This shouldn't happen if valid was True,
                        # but be safe
                        pass
    
    # Build final word
    final_word = [word[i] for i in range(L) if not removed[i]]
    return "".join(map(str, final_word))


def _classic_construct(n):
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
    return word
# EVOLVE-BLOCK-END
