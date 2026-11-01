import time
import itertools

def construct(n):
    # 1. Generate standard superpermutation
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
            max_possible = min(len(new), len(block))
            for o in range(max_possible, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Convert to list of ints for manipulation
    w = word[:]
    
    # 2. Get target permutations
    target_perms = set(itertools.permutations(range(1, n + 1)))
    num_perms = len(target_perms)
    
    # 3. Greedy deletion
    deadline = time.time() + 100 # 20s safety margin
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        # Early exit
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in target_perms:
                seen.add(p)
                if len(seen) == num_perms:
                    return True
        return len(seen) == num_perms

    # Initial check (should be valid)
    if not is_valid(w):
        return "".join(map(str, w))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to delete from the end to the beginning
        # Deleting from the end is often safer for structure?
        # Or random?
        # Let's try end to start.
        for i in range(len(w) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Tentative delete
            new_w = w[:i] + w[i+1:]
            if is_valid(new_w):
                w = new_w
                improved = True
                break # Restart scan
    
    return "".join(map(str, w))
