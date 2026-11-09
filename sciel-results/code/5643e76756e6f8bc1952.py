import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for evaluator overhead

    def can_add(x, S):
        # Check if adding x to S creates a 4-AP
        # S is a set of ints
        # x is not in S
        
        # Case 1: x is 1st term (a)
        # x, x+d, x+2d, x+3d
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in S) and (x + 2 * d in S) and (x + 3 * d in S):
                return False

        # Case 2: x is 2nd term (a+d)
        # x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2 * d in S):
                return False

        # Case 3: x is 3rd term (a+2d)
        # x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2 * d in S) and (x - d in S) and (x + d in S):
                return False

        # Case 4: x is 4th term (a+3d)
        # x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3 * d in S) and (x - 2 * d in S) and (x - d in S):
                return False

        return True

    best_set = set()
    best_size = 0

    while time.time() - start_time < time_limit:
        # Random Restart
        S = set()
        elements = list(range(1, N + 1))
        random.shuffle(elements)

        # Greedy fill
        for x in elements:
            if can_add(x, S):
                S.add(x)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

        # Local Search / Improvement
        # We try to add elements. If stuck, we try swaps or removals.
        improved = True
        stuck_count = 0
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add any element
            # Iterate through all elements not in S
            # To make it faster, we can just loop 1..N
            added = False
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        added = True
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = S.copy()
                        break # Restart the search for next add

            if not added:
                # Stuck. Try to improve via swap or removal.
                # Strategy: Try a random swap. If it allows an addition later, it's good.
                # Or just remove a random element to unblock.
                
                if S:
                    # Try a swap: remove y, add x
                    # Pick random x not in S
                    candidates = [i for i in range(1, N + 1) if i not in S]
                    if candidates:
                        x_new = random.choice(candidates)
                        # Pick random y in S
                        y_old = random.choice(list(S))
                        
                        # Check if S - {y_old} + {x_new} is valid
                        # We need to check if adding x_new to (S - {y_old}) is valid
                        # And removing y_old doesn't break anything (it can't, as S was valid)
                        S.remove(y_old)
                        if can_add(x_new, S):
                            S.add(x_new)
                            improved = True
                            # If we swapped, we didn't increase size, but we changed structure.
                            # We set improved=True to try adding again.
                            # Note: This might oscillate. We rely on the 'add' phase to find gains.
                        else:
                            # Swap not valid, revert
                            S.add(y_old)
                            # If swap failed, maybe just remove y_old?
                            # Removing reduces size, which is bad unless it allows multiple adds.
                            # Let's try removing y_old and see if we can add something else.
                            S.remove(y_old)
                            improved = True # Allow loop to try adding
                    else:
                        # No candidates outside S, S is full? Unlikely.
                        break
                else:
                    break
            
            # If we are stuck for a while, break to restart?
            # The 'improved' flag is set to True on removal/swap, so it continues.
            # To avoid infinite loops of swaps/removals without gain, we can track time or iterations.
            # But the outer loop has a time limit.
            
    return sorted(best_set)
