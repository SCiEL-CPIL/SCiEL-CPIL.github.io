import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 256
    
    def is_valid_addition(S, y):
        """Check if adding y to S creates a 3-AP."""
        # 1. y is the middle: a, y, c with a + c = 2y
        # a < y < c
        # a = 2y - c
        # Iterate possible a in S
        # Or iterate a in range(1, y) and check if 2y-a in S
        # Faster: iterate a in S such that a < y
        for a in S:
            if a < y:
                c = 2 * y - a
                if c > y and c in S:
                    return False
            elif a > y:
                # a acts as c? No, we covered a < y.
                # If a > y, then a could be c.
                # c = a. b = y. a2 = 2y - a.
                # We need a2 in S and a2 < y.
                a2 = 2 * y - a
                if 0 < a2 < y and a2 in S:
                    return False
        # 2. y is the right end: a, b, y with a + 2b = y
        # a = y - 2b
        # b must be in S, b < y/2
        # Iterate b in S such that b < y
        for b in S:
            if b < y:
                a = y - 2 * b
                if a > 0 and a < b and a in S:
                    return False
            elif b > y:
                # b could be the middle? No, y is end.
                # If b > y, b cannot be the middle of (a, b, y) because b > y.
                # Can b be the left end? y, b, c. Covered in case 3.
                pass
        # 3. y is the left end: y, b, c with 2b - y = c
        # c = 2b - y
        # b must be in S, b > y
        # Iterate b in S such that b > y
        for b in S:
            if b > y:
                c = 2 * b - y
                if c < b and c in S:
                    return False
        return True

    def greedy_construct():
        S = set()
        for x in range(1, N + 1):
            if is_valid_addition(S, x):
                S.add(x)
        return S

    def score(S):
        return sum(1.0 / x for x in S)

    def local_search(S):
        S = set(S)
        current_score = score(S)
        improved = True
        last_improvement_time = time.time()
        
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Phase 1: Try to add the smallest possible y
            # Sort candidates by value (1/y), so smallest y first
            candidates = [x for x in range(1, N + 1) if x not in S]
            # candidates are already sorted by x, which is 1/x descending
            for y in candidates:
                if is_valid_addition(S, y):
                    S.add(y)
                    current_score += 1.0 / y
                    improved = True
                    last_improvement_time = time.time()
                    break # Restart loop to catch new additions enabled by this one? 
                         # Actually, adding one might block others, but might enable? 
                         # No, adding never enables new additions in terms of validity 
                         # (it only adds constraints). 
                         # However, we might have skipped a smaller y because it was invalid 
                         # before, but is it still invalid? Yes. 
                         # So we can just continue scanning? 
                         # But we broke. Let's restart the candidate list scan 
                         # to ensure we don't miss a smaller y that became valid? 
                         # Impossible. Adding elements only makes things MORE constrained. 
                         # So if y was invalid, it remains invalid. 
                         # So we can just remove y from candidates and continue? 
                         # But we used a list. Let's just break and restart the 
                         # "find smallest valid" process. 
                         # Since N is small, restarting is fine.
                    continue # This continues the while loop
            
            # Phase 2: Try swaps
            # If no addition was made, try to swap x in S with y not in S
            # We want to remove x and add y such that score increases.
            # 1/y > 1/x => y < x
            # Sort S by x descending (remove largest/least valuable first? 
            # Or try all?)
            # Let's try removing each x and finding the best y.
            # To be efficient, try x in descending order of value (ascending x? no, descending x).
            # Actually, removing a large x (small weight) is "cheaper" in terms of weight loss,
            # so it's easier to find a y that compensates.
            # Let's iterate x in S sorted by x descending.
            S_list = sorted(S, reverse=True)
            for x in S_list:
                if time.time() > deadline - 1.0:
                    return S, current_score
                S_temp = S - {x}
                # Find smallest y not in S such that is_valid_addition(S_temp, y)
                # and y < x (to ensure gain)
                # Actually, we just want max gain. 
                # Gain = 1/y - 1/x. Maximized when y is smallest.
                # So find smallest valid y in S_temp.
                found = False
                for y in range(1, x): # y must be < x for gain
                    if y in S:
                        continue
                    if is_valid_addition(S_temp, y):
                        # Perform swap
                        S = S_temp
                        S.add(y)
                        current_score += (1.0 / y) - (1.0 / x)
                        improved = True
                        last_improvement_time = time.time()
                        found = True
                        break
                if found:
                    break # Restart while loop
            
            # If no improvement, we might be stuck.
            # Optional: Random restart or perturbation?
            # For now, just stop if no improvement.
            if not improved:
                break

        return S, current_score

    # 1. Initialize with Greedy
    best_S = greedy_construct()
    best_score = score(best_S)
    
    # 2. Local Search from Greedy
    S_ls, score_ls = local_search(best_S)
    if score_ls > best_score:
        best_S = S_ls
        best_score = score_ls

    # 3. Try a different initialization: Base-3 numbers
    # Numbers in [1, 256] with only 0 and 1 in base 3
    base3_S = set()
    for x in range(1, N + 1):
        b3 = []
        temp = x
        while temp > 0:
            b3.append(temp % 3)
            temp //= 3
        if all(d in [0, 1] for d in b3):
            base3_S.add(x)
    
    if score(base3_S) > best_score:
        best_S = base3_S
        best_score = score(best_S)
    
    # Local search from Base-3
    if time.time() < deadline - 10.0:
        S_ls2, score_ls2 = local_search(best_S)
        if score_ls2 > best_score:
            best_S = S_ls2
            best_score = score_ls2

    # 4. Randomized Local Search (Simulated Annealing lite)
    # If time remains, do a few random restarts with perturbation
    if time.time() < deadline - 5.0:
        for _ in range(10): # Max 10 restarts
            if time.time() > deadline - 5.0:
                break
            # Perturb best_S: remove a random element
            S_perturbed = set(best_S)
            if len(S_perturbed) > 0:
                # Remove a random element
                to_remove = random.choice(list(S_perturbed))
                S_perturbed.remove(to_remove)
                # Run local search on perturbed set
                S_rs, score_rs = local_search(S_perturbed)
                if score_rs > best_score:
                    best_S = S_rs
                    best_score = score_rs

    return list(best_S)
