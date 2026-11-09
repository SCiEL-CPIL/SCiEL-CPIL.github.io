import time
import random
import math

def construct():
    max_val = 1000
    deadline = time.time() + 112  # Safety margin for 120s limit
    
    best_S = []
    best_size = 0
    
    # We will run multiple independent trials.
    # Each trial:
    # 1. Greedy initialization (random order)
    # 2. Local search with "add" and "swap" moves.
    #    - Add: try to add a random unused element. If success, size increases.
    #    - Swap: pick random used 'a' and random unused 'c'. 
    #      Remove 'a'. Try to add 'c'. 
    #      If success, size maintained (but structure changed, might enable future adds).
    #      If fail, restore 'a'.
    #    This is effectively a random walk on the state space, biased towards growth when possible.
    
    def run_trial():
        S = []
        in_S = [False] * (max_val + 1)
        counts = [0] * (2 * max_val + 1)
        
        def add(x):
            S.append(x)
            in_S[x] = True
            counts[2 * x] += 1
            for y in S:
                if y != x:
                    counts[x + y] += 1
        
        def remove(x):
            S.remove(x)
            in_S[x] = False
            counts[2 * x] -= 1
            for y in S:
                if y != x:
                    counts[x + y] -= 1
        
        def can_add(x):
            if counts[2 * x] >= 2:
                return False
            for y in S:
                if y != x:
                    if counts[x + y] >= 2:
                        return False
            return True
        
        # 1. Greedy Init
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        for c in candidates:
            if can_add(c):
                add(c)
        
        current_size = len(S)
        
        # 2. Local Search
        # We maintain a list of unused elements to speed up selection?
        # Or just random pick and check in_S.
        # Since max_val is 1000, checking in_S is O(1).
        # Picking random unused:
        # While loop until we pick one not in S.
        
        # To avoid getting stuck, we use a "temperature" or just random moves.
        # The "swap" move is key for escaping local optima where no single add is possible.
        
        # We track best size in this trial.
        trial_best_size = current_size
        trial_best_S = list(S)
        
        # Limit trial time or iterations?
        # Let's use a global deadline check inside the loop.
        
        # Strategy:
        # While time < deadline - 5:
        #   If random() < 0.5:
        #       Try Add
        #   Else:
        #       Try Swap
        
        # Optimization:
        # If we add, size increases. We should record new best.
        # If we swap, size is same. We don't update best size, but we change state.
        
        # To make it more effective, we can bias towards adding if possible.
        # But random selection of 'c' for add might often fail.
        
        # Let's try a simple random walk.
        
        while time.time() < deadline - 2:
            # Choose move type
            # If we are far from optimal (e.g. < 700), maybe bias towards swaps to rearrange?
            # Or just 50/50.
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Try Add
                # Pick random c
                # To speed up, we can pick from a precomputed list of unused?
                # Maintaining a list of unused is O(N) to update, O(1) to pick.
                # But S changes, so unused changes.
                # Let's just pick random 1..1000 and check in_S.
                c = random.randint(1, max_val)
                if not in_S[c]:
                    if can_add(c):
                        add(c)
                        current_size += 1
                        if current_size > trial_best_size:
                            trial_best_size = current_size
                            trial_best_S = list(S)
            else:
                # Try Swap
                # Pick random a in S
                # Pick random c not in S
                if not S:
                    continue
                
                a = random.choice(S)
                
                # Pick c not in S
                # Random loop
                c = random.randint(1, max_val)
                while in_S[c]:
                    c = random.randint(1, max_val)
                
                if a == c:
                    continue
                
                # Attempt swap: remove a, try add c
                remove(a)
                if can_add(c):
                    add(c)
                    # Size unchanged, but state changed
                else:
                    # Restore a
                    add(a)
        
        return trial_best_S, trial_best_size

    # Run trials
    # How many trials?
    # Greedy init takes ~ O(N^2) = 10^6 ops?
    # can_add is O(|S|). |S| grows to ~500.
    # 1000 * 500 = 500k ops. Very fast.
    
    # Local search step:
    # can_add is O(|S|) ~ 500.
    # remove/add is O(|S|) ~ 500.
    # So one step is ~ 1000-2000 ops.
    # In 100s, we can do ~ 50,000 - 100,000 steps per trial?
    # Actually, Python overhead is higher. Maybe 10k-50k steps per trial.
    # If each trial takes 1-2 seconds, we can run 50-100 trials.
    
    # Let's run trials until time runs out.
    
    while time.time() < deadline - 5:
        S_trial, size_trial = run_trial()
        if size_trial > best_size:
            best_size = size_trial
            best_S = S_trial
        
        # Optional: Break if we hit a very high score?
        # Theoretical max is around 700-800?
        # If we get > 750, it's good.
        if best_size >= 780:
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
                # Should not happen, but if it does, return empty or fix?
                # Returning [] is safe but scores 0.
                # Let's try to repair? No, just return empty to be safe against invalidity.
                return []
    
    return best_S
