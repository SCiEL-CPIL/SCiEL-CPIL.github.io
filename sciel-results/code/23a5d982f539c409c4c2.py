import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    N = 256
    # Helper to check if adding x to set A creates a 4-AP
    def is_add_4ap_free(A, x):
        # A is a set
        # Check if x is part of any 4-AP with elements in A
        # 4-AP: y, y+d, y+2d, y+3d
        # x is one of these.
        
        # We iterate d.
        # Max d is such that 3d < N
        for d in range(1, (N - 1) // 3 + 1):
            # Case 1: x is start (y)
            if x + 3*d <= N:
                if (x + d in A) and (x + 2*d in A) and (x + 3*d in A):
                    return False
            # Case 2: x is 2nd (y+d)
            if x - d >= 1 and x + 2*d <= N:
                if (x - d in A) and (x + d in A) and (x + 2*d in A):
                    return False
            # Case 3: x is 3rd (y+2d)
            if x - 2*d >= 1 and x + d <= N:
                if (x - 2*d in A) and (x - d in A) and (x + d in A):
                    return False
            # Case 4: x is 4th (y+3d)
            if x - 3*d >= 1:
                if (x - 3*d in A) and (x - 2*d in A) and (x - d in A):
                    return False
        return True

    # Helper to count 3-APs involving x in set A
    def count_3aps_involving(A, x):
        count = 0
        # x is middle: a + c = 2x
        for a in A:
            if a == x: continue
            c = 2*x - a
            if c in A and c != x and c != a:
                # Ensure we don't double count? 
                # The pair (a,c) is unique if we just count pairs.
                # But we need distinct progressions.
                # A 3-AP is defined by its set of 3 elements.
                # If we count pairs (a,c) such that a+c=2x, each pair defines one AP.
                # So we just count such pairs.
                if a < c: # to count each AP once? No, the AP is {a,x,c}.
                    # Actually, if we just count all a in A such that c in A, we count each AP once 
                    # if we fix x as middle.
                    # Wait, if a and c are both in A, the AP is {a,x,c}.
                    # We just need to count how many such pairs exist.
                    pass
        # Let's do it simply:
        # Middle:
        for a in A:
            if a == x: continue
            c = 2*x - a
            if c in A and c != x:
                count += 1
        # First: x, b, c => c = 2b - x
        for b in A:
            if b == x: continue
            c = 2*b - x
            if c in A and c != x and c != b:
                count += 1
        # Last: a, b, x => a = 2b - x
        for b in A:
            if b == x: continue
            a = 2*b - x
            if a in A and a != x and a != b:
                count += 1
        return count

    # Initialize with greedy 4-AP free set
    A = set()
    for x in range(1, N + 1):
        if is_add_4ap_free(A, x):
            A.add(x)
    
    current_score = 0
    # Calculate initial score
    # We can just count all 3-APs in A
    # O(N^2)
    for a in A:
        for b in A:
            if a == b: continue
            c = 2*b - a
            if c in A and c > b:
                current_score += 1

    best_A = set(A)
    best_score = current_score

    while time.time() < deadline:
        improved = False
        
        # 1. Try Best Add
        best_gain = 0
        best_add_x = -1
        for x in range(1, N + 1):
            if x in A: continue
            if is_add_4ap_free(A, x):
                gain = count_3aps_involving(A, x)
                if gain > best_gain:
                    best_gain = gain
                    best_add_x = x
        
        if best_add_x != -1:
            A.add(best_add_x)
            current_score += best_gain
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_A = set(A)
            continue # Restart local search after improvement
        
        # 2. Try Best Swap
        best_swap_gain = 0
        best_swap_x = -1
        best_swap_y = -1
        
        # Optimize: Only check swaps if we have time
        # N^2 checks, each O(N).
        # We can sample or limit if too slow.
        # For N=256, N^2 = 65k. O(N) = 256. Total ~16M ops.
        # Python might take ~1-2s per full pass.
        # So we can do a few passes.
        
        for x in A:
            loss_x = count_3aps_involving(A, x)
            A.remove(x)
            for y in range(1, N + 1):
                if y in A: continue # y not in A (since x removed, y could be x? No, y not in original A)
                # y is not in A (which is now A\{x})
                # But y might have been x? No, x is removed.
                # We need y not in A_original.
                # If y was in A_original and y != x, it is in A (current).
                # So we just check if y is in current A.
                
                if is_add_4ap_free(A, y):
                    gain_y = count_3aps_involving(A, y)
                    net_gain = gain_y - loss_x
                    if net_gain > best_swap_gain:
                        best_swap_gain = net_gain
                        best_swap_x = x
                        best_swap_y = y
            A.add(x) # Restore x
        
        if best_swap_x != -1:
            A.remove(best_swap_x)
            A.add(best_swap_y)
            current_score += best_swap_gain
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_A = set(A)
            continue # Restart
        
        # 3. If no improvement, Perturb
        # Remove a few random elements and try to fill with best adds?
        # Or just random restart?
        # Let's do a random perturbation: swap k random elements if possible
        k = 5
        for _ in range(k):
            if len(A) < 2: break
            x = random.choice(list(A))
            # Find a valid y
            found = False
            for _ in range(100): # Try random y
                y = random.randint(1, N)
                if y in A: continue
                A.remove(x)
                if is_add_4ap_free(A, y):
                    A.add(y)
                    found = True
                    # Recalculate score? 
                    # This is getting messy to track score incrementally with random perturbations.
                    # Let's just recalculate score after perturbation.
                    break
                A.add(x)
            if not found:
                A.add(x)
        
        # Recalculate score
        current_score = 0
        for a in A:
            for b in A:
                if a == b: continue
                c = 2*b - a
                if c in A and c > b:
                    current_score += 1

        if current_score > best_score:
            best_score = current_score
            best_A = set(A)

    return sorted(best_A)
