import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    # Precompute scaled values A_i = L / d_i
    A = [L // d for d in denominators]
    
    # Greedy initialization
    selected = [False] * n
    S = 0
    for i in range(n):
        if S + A[i] < L:
            selected[i] = True
            S += A[i]
            
    best_S = S
    best_selected = selected[:]
    
    # Helper to get indices from boolean list
    def get_indices(sel):
        return [i for i in range(n) if sel[i]]
    
    # Local Search
    def try_move(sel, S, i, j, add_i, add_j):
        """
        Try a move. 
        If add_i is not None, remove i.
        If add_j is not None, add j.
        """
        new_S = S
        if i is not None:
            new_S -= A[i]
        if j is not None:
            new_S += A[j]
        return new_S

    # Main Loop
    while time.time() < deadline:
        improved = False
        
        # 1. Try single additions
        for j in range(n):
            if not selected[j]:
                if S + A[j] < L:
                    if S + A[j] > best_S:
                        selected[j] = True
                        S += A[j]
                        best_S = S
                        best_selected = selected[:]
                        improved = True
                        break # Restart loop with new state
        
        if improved:
            continue
            
        # 2. Try single removals (only if it allows a better state later? 
        #    Actually, removing alone never improves S. 
        #    But removing might allow adding a larger value. 
        #    So we combine removals with additions in swaps.)
        
        # 3. Try swaps (remove i, add j)
        swap_improved = False
        for i in range(n):
            if selected[i]:
                for j in range(n):
                    if not selected[j]:
                        new_S = S - A[i] + A[j]
                        if new_S < L and new_S > S:
                            # Perform swap
                            selected[i] = False
                            selected[j] = True
                            S = new_S
                            best_S = S
                            best_selected = selected[:]
                            swap_improved = True
                            break
                if swap_improved:
                    break
        
        if swap_improved:
            continue
            
        # 4. If no simple improvement, try "2-swap" (remove i1, i2, add j1, j2)
        #    This is O(N^4) which is 47^4 ~ 5M, might be too slow per iteration.
        #    Let's stick to 1-swap and maybe random restarts.
        
        # 5. Random Restart / Simulated Annealing step
        #    Pick a random valid move that might decrease S temporarily to escape local optimum
        #    Or just restart with a different greedy strategy?
        #    Given the structure, 1-swap local search is often sufficient for "closest under 1".
        #    But to be safe, let's try a few random perturbations.
        
        # Try a random valid swap that doesn't necessarily improve, 
        # but we only accept if it improves after a few steps? 
        # Simpler: Just break if no improvement found in a full pass.
        # Since N is small, 1-swap local search is very fast. 
        # If we are stuck, we are likely at a local optimum.
        
        # To escape, we can try a "kick": remove a random element and greedily refill.
        # Or just break and return best. 
        # Given the problem size, the global optimum is likely reachable by 1-swaps from greedy.
        
        # Let's add a check: if we haven't improved in last K iterations, do a random kick.
        # For now, if no improvement, break.
        break
        
    return get_indices(best_selected)
