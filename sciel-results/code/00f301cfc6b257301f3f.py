import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin before 90s kill
    
    N = 256
    # 1-indexed, so valid indices are 1..256
    # We represent the set as a Python set of integers for easy manipulation
    
    def count_3aps(A):
        """Count distinct 3-term arithmetic progressions in A."""
        # A is a set of ints
        count = 0
        # Optimization: iterate over pairs (a, c) with a < c, check if (a+c)//2 is in A
        # Or iterate a, b, check c = 2b - a
        # Let's use the middle element approach: for each b in A, for each d > 0, check if b-d and b+d in A.
        # This is O(|A|^2) roughly.
        A_list = sorted(A)
        A_set = A
        for b in A_list:
            # Max d such that b-d >= 1
            max_d = b - 1
            for d in range(1, max_d + 1):
                if (b - d) in A_set and (b + d) in A_set:
                    count += 1
        return count

    def has_4ap(A):
        """Check if A contains any 4-term arithmetic progression."""
        A_set = A
        # Iterate over pairs (a, c) to define d = (c-a)/2? No, 4-AP is a, a+d, a+2d, a+3d.
        # Check all a, d such that a+3d <= N.
        for a in A_set:
            max_d = (N - a) // 3
            for d in range(1, max_d + 1):
                if (a + d) in A_set and (a + 2*d) in A_set and (a + 3*d) in A_set:
                    return True
        return False

    def get_4ap_violations(A):
        """Return list of (a, d) for 4-APs in A."""
        A_set = A
        violations = []
        for a in A_set:
            max_d = (N - a) // 3
            for d in range(1, max_d + 1):
                if (a + d) in A_set and (a + 2*d) in A_set and (a + 3*d) in A_set:
                    violations.append((a, d))
        return violations

    # Greedy Initialization
    def greedy_init():
        A = set()
        # Try to add elements in a way that maximizes 3-APs
        # Simple greedy: iterate x from 1 to N, add if no 4-AP formed.
        # To get a denser set, we might want to prioritize elements that form 3-APs with existing ones.
        # Let's start with a random permutation and greedily add if valid.
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        A = set()
        for x in candidates:
            # Check if adding x creates a 4-AP
            # x can be part of 4-AP in 4 positions:
            # 1. x is start: x, x+d, x+2d, x+3d
            # 2. x is 2nd: x-d, x, x+d, x+2d
            # 3. x is 3rd: x-2d, x-d, x, x+d
            # 4. x is 4th: x-3d, x-2d, x-d, x
            
            valid = True
            # Check case 1: x is start
            for d in range(1, (N - x) // 3 + 1):
                if (x+d) in A and (x+2*d) in A and (x+3*d) in A:
                    valid = False
                    break
            if not valid: continue
            
            # Check case 2: x is 2nd
            for d in range(1, x):
                if (x-d) in A and (x+d) in A and (x+2*d) in A:
                    valid = False
                    break
            if not valid: continue
            
            # Check case 3: x is 3rd
            for d in range(1, (x-1)//2 + 1):
                if (x-2*d) in A and (x-d) in A and (x+d) in A:
                    valid = False
                    break
            if not valid: continue
            
            # Check case 4: x is 4th
            for d in range(1, (x-1)//3 + 1):
                if (x-3*d) in A and (x-2*d) in A and (x-d) in A:
                    valid = False
                    break
            if not valid: continue
            
            A.add(x)
        return A

    # Local Search Polish
    def local_search(A, max_iters=1000):
        current_A = set(A)
        current_score = count_3aps(current_A)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try removing an element
            for x in list(current_A):
                new_A = set(current_A)
                new_A.remove(x)
                if has_4ap(new_A):
                    continue
                new_score = count_3aps(new_A)
                if new_score > current_score:
                    current_A = new_A
                    current_score = new_score
                    improved = True
                    break
            if improved:
                continue
                
            # Try adding an element
            for x in range(1, N + 1):
                if x in current_A:
                    continue
                new_A = set(current_A)
                new_A.add(x)
                if has_4ap(new_A):
                    continue
                new_score = count_3aps(new_A)
                if new_score > current_score:
                    current_A = new_A
                    current_score = new_score
                    improved = True
                    break
            if improved:
                continue
                
            # Try swapping
            # To save time, only try a few random swaps if stuck
            if iters % 10 == 0:
                if len(current_A) > 1:
                    x_out = random.choice(list(current_A))
                    x_in = random.randint(1, N)
                    if x_in not in current_A:
                        new_A = set(current_A)
                        new_A.remove(x_out)
                        new_A.add(x_in)
                        if not has_4ap(new_A):
                            new_score = count_3aps(new_A)
                            if new_score > current_score:
                                current_A = new_A
                                current_score = new_score
                                improved = True
        return current_A, current_score

    # Simulated Annealing
    def simulated_annealing(A, time_limit):
        current_A = set(A)
        current_score = count_3aps(current_A)
        best_A = set(current_A)
        best_score = current_score
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        while time.time() < time_limit:
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.7:
                # Swap
                if len(current_A) == 0:
                    continue
                x_out = random.choice(list(current_A))
                x_in = random.randint(1, N)
                if x_in == x_out:
                    continue
                new_A = set(current_A)
                new_A.remove(x_out)
                if x_in not in current_A:
                    new_A.add(x_in)
                else:
                    # x_in is already in, so we just removed x_out. This is a removal.
                    # Let's stick to strict swaps for simplicity in this branch
                    # Actually, if x_in is in current_A, we are just removing x_out.
                    # Let's ensure x_in is not in current_A for a swap.
                    # If x_in is in current_A, we skip or treat as removal.
                    # Let's just retry if x_in is in current_A.
                    continue
            else:
                # Add or Remove
                if random.random() < 0.5 and len(current_A) < N:
                    x_in = random.randint(1, N)
                    if x_in in current_A:
                        continue
                    new_A = set(current_A)
                    new_A.add(x_in)
                else:
                    if len(current_A) == 0:
                        continue
                    x_out = random.choice(list(current_A))
                    new_A = set(current_A)
                    new_A.remove(x_out)
            
            # Check validity
            if has_4ap(new_A):
                continue
            
            new_score = count_3aps(new_A)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_A = new_A
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(current_A)
                
                # Occasionally polish
                if random.random() < 0.01:
                    polished_A, polished_score = local_search(current_A, max_iters=50)
                    if polished_score > best_score:
                        best_score = polished_score
                        best_A = set(polished_A)
                        current_A = set(polished_A)
                        current_score = polished_score
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_A, best_score

    # Main Loop
    # 1. Get a good initial solution
    A_init = greedy_init()
    # Polish it
    A_curr, score_curr = local_search(A_init, max_iters=200)
    
    best_A = set(A_curr)
    best_score = score_curr
    
    # 2. Run SA for the remaining time
    # We split the time: some for SA, some for final polish
    sa_deadline = start_time + 75.0
    
    # Run multiple short SA runs or one long one?
    # One long SA with restarts if stuck?
    # Let's run SA until close to deadline.
    A_sa, score_sa = simulated_annealing(A_curr, sa_deadline)
    
    if score_sa > best_score:
        best_A = set(A_sa)
        best_score = score_sa
        
    # 3. Final Polish with remaining time
    final_deadline = deadline - 5.0
    if time.time() < final_deadline:
        A_final, score_final = local_search(best_A, max_iters=10000)
        if score_final > best_score:
            best_A = set(A_final)
            best_score = score_final
            
    return sorted(best_A)

# EVOLVE-BLOCK-END
