import time
import random

def construct():
    n = 256
    deadline = time.time() + 85  # 5s safety margin
    
    def is_valid_addition(a, T):
        """Check if adding a to T (set of elements > a) violates constraint.
        Returns True if a can be added (no pair in T sums to 0 mod a)."""
        if len(T) < 2:
            return True
        # Check all pairs in T
        T_list = list(T)
        m = len(T_list)
        for i in range(m):
            x = T_list[i]
            rx = x % a
            for j in range(i+1, m):
                y = T_list[j]
                if (rx + y) % a == 0:
                    return False
        return True
    
    def greedy_descending(start_from=256, exclude=set()):
        """Greedy: process from start_from down to 1, add if valid."""
        T = set()
        for a in range(start_from, 0, -1):
            if a in exclude:
                continue
            if is_valid_addition(a, T):
                T.add(a)
        return T
    
    def verify(S):
        """Verify S is valid."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Initial greedy
    best = greedy_descending()
    best_size = len(best)
    
    # Local search: try removing one element and re-greedy
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_list = sorted(best)
        for a in best_list:
            if time.time() >= deadline:
                break
            # Remove a, re-greedy
            exclude = {a}
            candidate = greedy_descending(exclude=exclude)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
                improved = True
    
    # Randomized restarts
    attempts = 0
    while time.time() < deadline and attempts < 50:
        attempts += 1
        # Random perturbation: exclude a random subset
        exclude = set()
        for a in range(1, n+1):
            if random.random() < 0.05:
                exclude.add(a)
        candidate = greedy_descending(exclude=exclude)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Final verification
    if not verify(best):
        # Fallback to known valid set
        best = set(range(171, 257))
    
    return list(best)
