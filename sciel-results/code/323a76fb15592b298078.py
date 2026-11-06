import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # in_S is a list of 0s and 1s, size N+1
    def is_4ap_free(in_S, x):
        # Check 1: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if in_S[x + d] and in_S[x + 2 * d] and in_S[x + 3 * d]:
                return False
        
        # Check 2: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if in_S[x - 3 * d] and in_S[x - 2 * d] and in_S[x - d]:
                return False
        
        # Check 3: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if in_S[x - 2 * d] and in_S[x - d] and in_S[x + d]:
                return False
        
        # Check 4: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if in_S[x - d] and in_S[x + d] and in_S[x + 2 * d]:
                return False
        
        return True

    def greedy_random(in_S):
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_4ap_free(in_S, x):
                in_S[x] = 1

    def local_search(in_S, time_limit):
        # in_S is modified in place
        # We track size
        current_size = sum(in_S)
        best_size = current_size
        best_in_S = in_S.copy()
        
        while time.time() < time_limit:
            # Try Add
            if current_size < N:
                # Sample a few candidates
                for _ in range(20):
                    x = random.randint(1, N)
                    if not in_S[x]:
                        if is_4ap_free(in_S, x):
                            in_S[x] = 1
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_in_S = in_S.copy()
                            break
            
            # Try Swap
            if current_size > 0:
                # Pick random y in S
                # To pick random y in S efficiently, we can maintain a list of indices
                # But maintaining the list is overhead.
                # For N=512, sum(in_S) is ~200.
                # We can just pick random x and check if in_S[x] is 1, retry if not.
                # Or just pick a random index and check.
                for _ in range(20):
                    y = random.randint(1, N)
                    if in_S[y]:
                        x = random.randint(1, N)
                        if not in_S[x]:
                            in_S[y] = 0
                            current_size -= 1
                            if is_4ap_free(in_S, x):
                                in_S[x] = 1
                                current_size += 1
                                if current_size > best_size:
                                    best_size = current_size
                                    best_in_S = in_S.copy()
                            else:
                                in_S[y] = 1
                                current_size += 1
                            break
        
        return best_in_S, best_size

    # Initialize
    in_S = [0] * (N + 1)
    greedy_random(in_S)
    best_in_S, best_size = local_search(in_S, start_time + 5)
    
    # Main loop: Restart or Refine
    # Let's try refining the best found so far
    current_in_S = best_in_S.copy()
    
    while time.time() < deadline - 5:
        # Run local search on current best
        t_start = time.time()
        # Give it a few seconds
        new_best_in_S, new_best_size = local_search(current_in_S, t_start + 5)
        
        if new_best_size > best_size:
            best_size = new_best_size
            best_in_S = new_best_in_S
            current_in_S = best_in_S.copy()
        else:
            # If no improvement, maybe try a fresh random start?
            # Or just keep trying to refine?
            # Let's try a fresh random start every 10 seconds if stuck
            if time.time() - start_time > 10 and random.random() < 0.1:
                in_S = [0] * (N + 1)
                greedy_random(in_S)
                fresh_in_S, fresh_size = local_search(in_S, time.time() + 2)
                if fresh_size > best_size:
                    best_size = fresh_size
                    best_in_S = fresh_in_S
                    current_in_S = best_in_S.copy()
    
    # Convert best_in_S to sorted list
    result = [i for i in range(1, N + 1) if best_in_S[i]]
    return result
