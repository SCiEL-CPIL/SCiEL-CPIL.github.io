"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Precompute conflict matrix: conflict[i][j] = True if (i+1, j+1) is INCOMPATIBLE
    N = 256
    conflict = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            a, b = i + 1, j + 1
            if (a * b) % (a + b) == 0:
                conflict[i][j] = True
                conflict[j][i] = True
    
    def is_compatible_set(s):
        """Check if set s (0-indexed) has no conflicts."""
        for i, a in enumerate(s):
            for b in s[i+1:]:
                if conflict[a][b]:
                    return False
        return True
    
    # Start with all odd numbers (0-indexed: 0, 2, 4, ..., 254)
    odds = set(i for i in range(N) if (i + 1) % 2 == 1)
    best_set = set(odds)
    best_size = len(best_set)
    
    evens = [i for i in range(N) if (i + 1) % 2 == 0]
    powers_of_2_idx = [i for i in range(N) if (i + 1) in [2, 4, 8, 16, 32, 64, 128]]
    
    def can_add(num, s):
        """Check if num (0-indexed) can be added to set s."""
        for x in s:
            if x == num:
                continue
            if conflict[num][x]:
                return False
        return True
    
    # Multiple greedy attempts with different random orderings
    num_attempts = 500
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        current = set(odds)
        
        # Create a shuffled list of evens
        shuffled_evens = list(evens)
        random.shuffle(shuffled_evens)
        
        # In some attempts, prioritize powers of 2
        if attempt % 3 == 0:
            other_evens = [e for e in shuffled_evens if e not in powers_of_2_idx]
            for p in powers_of_2_idx:
                if can_add(p, current):
                    current.add(p)
            shuffled_evens = other_evens
        
        for e in shuffled_evens:
            if time.time() > deadline:
                break
            if can_add(e, current):
                current.add(e)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Local search: try to add more elements by removing conflicting ones
    for ls_iter in range(200):
        if time.time() > deadline:
            break
        
        current = set(best_set)
        remaining = [x for x in range(N) if x not in current]
        random.shuffle(remaining)
        
        improved = False
        for r in remaining:
            if time.time() > deadline:
                break
            conflicts = [x for x in current if conflict[r][x]]
            if not conflicts:
                current.add(r)
                improved = True
            elif len(conflicts) == 1:
                c = conflicts[0]
                new_set = current - {c}
                new_set.add(r)
                if len(new_set) > len(current):
                    current = new_set
                    improved = True
                elif len(new_set) == len(current):
                    if random.random() < 0.3:
                        current = new_set
                        improved = True
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            improved = True
        
        if not improved:
            if ls_iter < 30:
                continue
            break
    
    # Final polish: try all possible single swaps to improve
    for polish_iter in range(50):
        if time.time() > deadline:
            break
        
        current = set(best_set)
        remaining = [x for x in range(N) if x not in current]
        random.shuffle(remaining)
        
        improved = False
        for r in remaining:
            if time.time() > deadline:
                break
            conflicts = [x for x in current if conflict[r][x]]
            if not conflicts:
                current.add(r)
                improved = True
            elif len(conflicts) <= 2:
                # Try removing up to 2 conflicts and adding r
                for c in conflicts:
                    new_set = current - {c}
                    new_set.add(r)
                    if len(new_set) > len(current):
                        current = new_set
                        improved = True
                        break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            improved = True
        
        if not improved:
            break
    
    return list(x + 1 for x in best_set)
# EVOLVE-BLOCK-END
