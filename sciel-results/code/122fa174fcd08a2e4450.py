import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 105  # 15s margin for safety
    
    # State
    S = []
    in_S = [False] * (max_val + 1)
    counts = [0] * (2 * max_val + 1)
    
    def try_add(x):
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if y != x and counts[x + y] >= 2:
                return False
        # Add
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    def try_remove(x):
        # Use list.remove which is O(n), but we also need to handle the counts
        # Find index for O(1) removal if possible, but S is a list
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    def is_valid_swap(a, c):
        # Check if S - {a} + {c} is valid
        # Only need to check sums that could exceed 2 after the swap
        # Affected sums: those involving a or c
        
        # Check 2*c: count of 2*c after removing a
        # a contributes to 2*c if there exists y in S\{a} with a+y=2*c, i.e., y=2*c-a
        # and y in S, y!=a, y!=c
        new_2c = counts[2*c]
        # Does a contribute to 2*c?
        y_candidate = 2*c - a
        if y_candidate != a and y_candidate != c and 1 <= y_candidate <= max_val and in_S[y_candidate]:
            new_2c -= 1
        # a itself doesn't contribute to 2*c unless a==c (false)
        if new_2c > 2:
            return False
        
        # For each y in S\{a}, check c+y
        # Current count of c+y: counts[c+y]
        # a contributes to c+y if a+z = c+y for some z in S\{a}, i.e., z = c+y-a
        # So for each y in S\{a}:
        #   new_count(c+y) = counts[c+y] - (1 if a contributes to c+y else 0) + 1 (from c+y itself)
        # a contributes to c+y if z = c+y-a is in S, z!=a, z!=c, z!=y
        for y in S:
            if y == a:
                continue
            s = c + y
            if s > 2*max_val:
                continue
            new_count = counts[s] + 1  # +1 for the new c+y
            # Check if a contributed to s = c+y
            # a + z = s => z = s - a = c + y - a
            z = c + y - a
            if z != a and z != c and z != y and 1 <= z <= max_val and in_S[z]:
                new_count -= 1
            if new_count > 2:
                return False
        return True

    def local_search():
        while time.time() < deadline:
            # 1. Try to add any unused element
            added = False
            unused = [x for x in range(1, max_val + 1) if not in_S[x]]
            random.shuffle(unused)
            
            for c in unused:
                if try_add(c):
                    added = True
                    break
            
            if added:
                continue
            
            # 2. Try systematic swaps: for each a in S, try all unused c
            # But that's O(|S| * |unused|) which is too slow. 
            # Instead, try a limited number of random swaps but with better strategy
            swapped = False
            if not S:
                break
            
            # Try up to 200 random swaps
            for _ in range(200):
                a = random.choice(S)
                c = random.choice(unused)
                if is_valid_swap(a, c):
                    try_remove(a)
                    try_add(c)
                    swapped = True
                    break
            
            if swapped:
                continue
            
            # 3. Perturbation: remove elements that are "constraining" others
            # An element a is constraining if many sums involving a are at count 2
            if len(S) > 10:
                # Find elements with high constraint
                constraint_scores = []
                for a in S:
                    score = 0
                    for y in S:
                        if y != a:
                            if counts[a + y] == 2:
                                score += 1
                    if counts[2*a] == 2:
                        score += 1
                    constraint_scores.append((score, a))
                
                # Remove 2-4 elements with highest constraint
                constraint_scores.sort(reverse=True)
                k = min(random.randint(2, 4), len(S) - 5)
                for _, a in constraint_scores[:k]:
                    try_remove(a)
                
    # Main Loop: Multiple Restarts
    best_size = 0
    best_S = []
    
    while time.time() < deadline:
        # Reset State
        S.clear()
        for i in range(max_val + 1):
            in_S[i] = False
        for i in range(2 * max_val + 1):
            counts[i] = 0
            
        # 1. Greedy Initialization with multiple strategies
        # Strategy 1: Random order
        order = list(range(1, max_val + 1))
        random.shuffle(order)
        for x in order:
            try_add(x)
        
        # 2. Local Search
        local_search()
        
        # 3. Final polish: try to add any remaining elements
        unused = [x for x in range(1, max_val + 1) if not in_S[x]]
        random.shuffle(unused)
        for c in unused:
            if try_add(c):
                pass
        
        # 4. Update Best
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
        
        # If we've found a large set, we can be more aggressive with restarts
        # but keep trying for improvement
    
    # Verify the best set is actually valid
    if best_S:
        # Re-verify counts
        verify_counts = [0] * (2 * max_val + 1)
        for i, a in enumerate(best_S):
            for j, b in enumerate(best_S):
                if i <= j:
                    verify_counts[a + b] += 1
        for c in verify_counts:
            if c > 2:
                # Invalid - should not happen, but return empty if so
                return []
    
    return best_S
