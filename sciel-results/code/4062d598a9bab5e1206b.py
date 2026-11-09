import time
import random
import math

def construct():
    max_val = 1000
    deadline = time.time() + 108  # 42s margin for safety
    
    # Simulated Annealing approach
    # State: in_S array, counts array
    in_S = [False] * (max_val + 1)
    counts = [0] * (2 * max_val + 1)
    S = []  # list of elements in S for quick iteration
    
    def add_element(x):
        """Add x to S, updating counts. Assumes validity is checked externally."""
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
    
    def remove_element(x):
        """Remove x from S, updating counts."""
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1
    
    def can_add(x):
        """Check if x can be added without violating B2[2] constraint."""
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if y != x and counts[x + y] >= 2:
                return False
        return True
    
    def can_swap(a, c):
        """Check if replacing a with c maintains validity."""
        if a == c:
            return False
        
        # Check 2*c
        new_2c = counts[2 * c]
        # Does a contribute to 2*c?
        y_candidate = 2 * c - a
        if y_candidate != a and y_candidate != c and 1 <= y_candidate <= max_val and in_S[y_candidate]:
            new_2c -= 1
        if new_2c > 2:
            return False
        
        # Check c+y for all y in S\{a}
        for y in S:
            if y == a:
                continue
            s = c + y
            if s > 2 * max_val:
                continue
            new_count = counts[s] + 1  # +1 for new c+y
            # Does a contribute to s?
            z = s - a
            if z != a and z != c and z != y and 1 <= z <= max_val and in_S[z]:
                new_count -= 1
            if new_count > 2:
                return False
        return True
    
    def greedy_init(gap):
        """Initialize with elements at regular gaps."""
        S.clear()
        for i in range(max_val + 1):
            in_S[i] = False
        for i in range(2 * max_val + 1):
            counts[i] = 0
        
        start = random.randint(1, gap)
        for x in range(start, max_val + 1, gap):
            if can_add(x):
                add_element(x)
    
    def simulated_annealing(initial_temp, cooling_rate, min_temp, max_iter):
        """Run simulated annealing to improve the current set."""
        temp = initial_temp
        best_size = len(S)
        best_S = list(S)
        
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            if temp < min_temp:
                break
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            unused = [x for x in range(1, max_val + 1) if not in_S[x]]
            
            if move_type < 0.3 and unused:
                # Try to add a random element
                c = random.choice(unused)
                if can_add(c):
                    add_element(c)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                else:
                    # Accept with probability based on "cost" (how many constraints violated)
                    # For simplicity, reject
                    pass
            elif move_type < 0.5 and S:
                # Try to remove a random element
                a = random.choice(S)
                remove_element(a)
                # Remove is always "safe" in terms of validity, but might reduce size
                # We'll let the add moves try to recover
            else:
                # Try a swap
                if S and unused:
                    a = random.choice(S)
                    c = random.choice(unused)
                    if can_swap(a, c):
                        remove_element(a)
                        add_element(c)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = list(S)
            
            # Cool down
            temp *= cooling_rate
        
        # Restore best
        S.clear()
        for i in range(max_val + 1):
            in_S[i] = False
        for i in range(2 * max_val + 1):
            counts[i] = 0
        for x in best_S:
            add_element(x)
        
        return best_size
    
    def polish():
        """Try to add any remaining elements in multiple passes."""
        for _ in range(5):
            unused = [x for x in range(1, max_val + 1) if not in_S[x]]
            random.shuffle(unused)
            added_any = False
            for c in unused:
                if can_add(c):
                    add_element(c)
                    added_any = True
            if not added_any:
                break
    
    # Main: Multiple restarts with different initializations and SA parameters
    best_size = 0
    best_S = []
    
    gap_strategies = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30, 35, 40, 50]
    
    while time.time() < deadline:
        # Choose initialization strategy
        gap = random.choice(gap_strategies)
        greedy_init(gap)
        
        # Run simulated annealing
        initial_temp = random.uniform(1.0, 5.0)
        cooling_rate = random.uniform(0.97, 0.995)
        min_temp = 0.01
        max_iter = random.randint(5000, 20000)
        
        sa_size = simulated_annealing(initial_temp, cooling_rate, min_temp, max_iter)
        
        # Polish
        polish()
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
        
        # Check if we should break early
        if best_size >= 750:  # Good enough threshold
            break
    
    # Final verification
    if best_S:
        verify_counts = [0] * (2 * max_val + 1)
        for i, a in enumerate(best_S):
            for j, b in enumerate(best_S):
                if i <= j:
                    verify_counts[a + b] += 1
        for c in verify_counts:
            if c > 2:
                return []
    
    return best_S
