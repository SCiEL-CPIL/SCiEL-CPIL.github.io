"""Stronger B2[2] set construction with greedy init + local search."""
import time
import random
import sys

N = 1000

def construct():
    t0 = time.time()
    deadline = t0 + 130  # safety margin
    
    def validate(S):
        """Check if set S is valid B2[2]."""
        counts = {}
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for b in S_list[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
                if counts[s] > 2:
                    return False
        return True
    
    def get_counts(S_list):
        """Get sum counts for a sorted list."""
        counts = {}
        for i, a in enumerate(S_list):
            for b in S_list[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts
    
    def can_add(S_list, counts, x, in_set):
        """Check if x can be added to S_list."""
        if in_set[x]:
            return False
        # Check new sums
        for y in S_list:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2*x, 0) >= 2:
            return False
        return True
    
    def add_to(S_list, counts, in_set, x):
        """Add x to the set."""
        S_list.append(x)
        in_set[x] = True
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in S_list[:-1]:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
    
    def remove_from(S_list, counts, in_set, x, idx):
        """Remove x at index idx from the set."""
        # Remove sums
        for i, y in enumerate(S_list):
            if i == idx:
                continue
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        in_set[x] = False
        S_list.pop(idx)
    
    # Phase 1: Greedy initialization (multiple orders)
    best_S = []
    best_size = 0
    
    # Standard greedy
    S_list = []
    counts = {}
    in_set = [False] * (N + 1)
    for x in range(1, N + 1):
        if can_add(S_list, counts, x, in_set):
            add_to(S_list, counts, in_set, x)
    if len(S_list) > best_size:
        best_size = len(S_list)
        best_S = S_list[:]
    
    # Greedy with different starting points
    for start in [1, 2, 3, 5, 10]:
        S_list = []
        counts = {}
        in_set = [False] * (N + 1)
        for x in range(start, N + 1):
            if can_add(S_list, counts, x, in_set):
                add_to(S_list, counts, in_set, x)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_S = S_list[:]
    
    # Random greedy attempts
    random.seed(42)
    for attempt in range(5):
        S_list = []
        counts = {}
        in_set = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if time.time() > deadline - 10:
                break
            if can_add(S_list, counts, x, in_set):
                add_to(S_list, counts, in_set, x)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_S = S_list[:]
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Local search with swaps
    S_list = best_S[:]
    S_list.sort()
    counts = get_counts(S_list)
    in_set = [False] * (N + 1)
    for x in S_list:
        in_set[x] = True
    
    # Try to add elements first
    def try_additions(S_list, counts, in_set):
        added = 0
        # Try adding random elements
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            x = random.randint(1, N)
            if can_add(S_list, counts, x, in_set):
                add_to(S_list, counts, in_set, x)
                added += 1
        # Try adding from low to high
        for x in range(1, N + 1):
            if time.time() > deadline - 5:
                break
            if not in_set[x] and can_add(S_list, counts, x, in_set):
                add_to(S_list, counts, in_set, x)
                added += 1
        return added
    
    # Try swaps
    def try_swap(S_list, counts, in_set, x_out, x_in):
        """Try to swap x_out with x_in. Returns True if successful."""
        if in_set[x_in]:
            return False
        if x_out == x_in:
            return False
        
        # Find index of x_out
        idx = -1
        for i, v in enumerate(S_list):
            if v == x_out:
                idx = i
                break
        if idx == -1:
            return False
        
        # Temporarily remove x_out
        for i, y in enumerate(S_list):
            if i == idx:
                continue
            s = x_out + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        counts[2*x_out] -= 1
        if counts[2*x_out] == 0:
            del counts[2*x_out]
        
        # Check if x_in can be added
        valid = True
        for y in S_list:
            if y == x_out:
                continue
            s = x_in + y
            if counts.get(s, 0) >= 2:
                valid = False
                break
        if valid and counts.get(2*x_in, 0) >= 2:
            valid = False
        
        if valid:
            # Complete the swap
            in_set[x_out] = False
            S_list[idx] = x_in
            in_set[x_in] = True
            counts[2*x_in] = counts.get(2*x_in, 0) + 1
            for i, y in enumerate(S_list):
                if i == idx:
                    continue
                s = x_in + y
                counts[s] = counts.get(s, 0) + 1
            return True
        else:
            # Restore
            in_set[x_out] = True
            S_list[idx] = x_out
            counts[2*x_out] = counts.get(2*x_out, 0) + 1
            for i, y in enumerate(S_list):
                if i == idx:
                    continue
                s = x_out + y
                counts[s] = counts.get(s, 0) + 1
            return False
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Try additions
        if try_additions(S_list, counts, in_set) > 0:
            improved = True
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_S = S_list[:]
                best_S.sort()
            continue
        
        # Try swaps
        S_list.sort()
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            if not S_list:
                break
            idx_out = random.randint(0, len(S_list) - 1)
            x_out = S_list[idx_out]
            x_in = random.randint(1, N)
            
            if try_swap(S_list, counts, in_set, x_out, x_in):
                improved = True
                S_list.sort()
                # Recount after swap
                counts = get_counts(S_list)
                in_set = [False] * (N + 1)
                for x in S_list:
                    in_set[x] = True
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S = S_list[:]
                    best_S.sort()
                break
        
        # Also try to replace an element with a better one
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            if not S_list:
                break
            idx_out = random.randint(0, len(S_list) - 1)
            x_out = S_list[idx_out]
            # Try nearby values
            for offset in [-3, -2, -1, 1, 2, 3]:
                x_in = x_out + offset
                if 1 <= x_in <= N and x_in != x_out:
                    if try_swap(S_list, counts, in_set, x_out, x_in):
                        improved = True
                        S_list.sort()
                        counts = get_counts(S_list)
                        in_set = [False] * (N + 1)
                        for x in S_list:
                            in_set[x] = True
                        if len(S_list) > best_size:
                            best_size = len(S_list)
                            best_S = S_list[:]
                            best_S.sort()
                        break
                if improved:
                    break
    
    # Final validation
    if not validate(best_S):
        # Fallback to simple greedy
        S_list = []
        counts = {}
        in_set = [False] * (N + 1)
        for x in range(1, N + 1):
            if can_add(S_list, counts, x, in_set):
                add_to(S_list, counts, in_set, x)
        return S_list
    
    return best_S
