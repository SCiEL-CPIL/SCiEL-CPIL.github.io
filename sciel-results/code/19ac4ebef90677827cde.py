import time
import random
from itertools import combinations

def is_sidon(S):
    """Check if a list of integers is a Sidon set (unique pair sums including diagonals)."""
    sums = set()
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_max_sidon_subset(S):
    """Find the size of the largest Sidon subset of S and return the subset."""
    n = len(S)
    # Check from largest possible down to smallest
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            if is_sidon(subset):
                return k, list(subset)
    return 2, [] # Any 2 distinct elements form a Sidon set

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Initialize with a dense set in a small range to force collisions
    S = random.sample(range(0, 21), 12)
    S.sort()
    best_set = S[:]
    best_score, _ = find_max_sidon_subset(S)
    
    # Local search with conflict resolution
    while time.time() < deadline:
        improved = False
        current_score, conflict_subset = find_max_sidon_subset(S)
        
        if current_score < best_score:
            best_score = current_score
            best_set = S[:]
            if best_score == 3: # Theoretical minimum likely
                break

        # Try to break the current maximum Sidon subset
        if conflict_subset:
            # Pick a random element from the conflict subset and move it
            for _ in range(50):
                if time.time() > deadline:
                    break
                idx_in_S = S.index(random.choice(conflict_subset))
                # Try moving this element to a new value
                # Bias towards values that cause collisions with others
                # Just random in small range for now
                new_val = random.randint(0, 25)
                if new_val in S:
                    continue
                
                new_S = S[:]
                new_S[idx_in_S] = new_val
                new_S.sort()
                
                new_score, _ = find_max_sidon_subset(new_S)
                
                if new_score < best_score:
                    best_score = new_score
                    S = new_S
                    best_set = S[:]
                    improved = True
                    break # Restart loop with new best
                elif new_score < current_score:
                    # Accept improvement even if not global best yet
                    S = new_S
                    improved = True
                    break
                elif new_score == current_score and random.random() < 0.2:
                    S = new_S
                    improved = True
        
        if not improved:
            # Stuck, restart with random dense set
            S = random.sample(range(0, 21), 12)
            S.sort()
            score, _ = find_max_sidon_subset(S)
            if score < best_score:
                best_score = score
                best_set = S[:]

    if best_set is None:
        return list(range(12))
        
    return best_set
