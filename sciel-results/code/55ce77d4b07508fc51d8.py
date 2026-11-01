import time
import itertools
import random

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
    deadline = time.time() + 110 # 40s safety margin (150s total limit)
    
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

    # First pass: simple greedy deletion from end to start
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(w) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Tentative delete
            new_w = w[:i] + w[i+1:]
            if is_valid(new_w):
                w = new_w
                improved = True
                break # Restart scan

    # Second pass: randomized local search with swaps and insertions
    if time.time() < deadline:
        best_w = w[:]
        best_len = len(w)
        
        # Use a fixed seed for reproducibility in evaluation
        random.seed(42)
        
        while time.time() < deadline:
            # Try a random modification
            if len(w) > n:
                if random.random() < 0.7:
                    # Try deleting a random position
                    idx = random.randint(0, len(w) - 1)
                    new_w = w[:idx] + w[idx+1:]
                else:
                    # Try swapping two random positions
                    if len(w) > 2:
                        i, j = random.sample(range(len(w)), 2)
                        new_w = w[:]
                        new_w[i], new_w[j] = new_w[j], new_w[i]
                    else:
                        continue
            
            if is_valid(new_w):
                w = new_w
                if len(w) < best_len:
                    best_w = w[:]
                    best_len = len(w)
                    # If we made a significant improvement, we can continue
                    # Otherwise, maybe we should try to get stuck out of local minima
            else:
                # If invalid, we can occasionally accept a slightly worse solution
                # to escape local minima, but for now let's just stick to improvements
                pass
            
            # Occasionally, if we're stuck, we might want to try a more aggressive move
            # But for simplicity, let's just keep trying random moves until time runs out

    return "".join(map(str, w))
