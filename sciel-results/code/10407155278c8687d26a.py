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
        
        # Improved Strategy:
        # 1. Calculate "slack" for all unused elements.
        #    Slack(x) = min(2 - counts[2x], min_{y in S, y!=x} (2 - counts[x+y]))
        #    If slack > 0, x can be added.
        # 2. If any slack > 0, pick the one with max slack (or random among top k) and add.
        # 3. If no slack > 0, perform a swap to create slack.
        #    Swap: remove a, add c.
        #    We want to remove 'a' that "blocks" many elements and add 'c' that is "less blocking".
        #    Approximation: 
        #       - Pick random c not in S.
        #       - Find 'a' in S such that removing a increases slack for c?
        #       - Simpler: Random swap as before, but maybe bias towards swapping out 'a' 
        #         that has high contribution to counts (e.g. high degree in sum graph).
        
        # Let's implement a hybrid:
        # - 70% chance: Try to add element with highest slack (if any).
        # - 30% chance: Try a random swap.
        
        # To speed up slack calculation, we can just check can_add for a few candidates?
        # Or maintain a list of "blocked" elements?
        # Since N=1000, calculating slack for all 1000 unused elements takes 1000 * |S| ~ 500k ops.
        # This is too slow for every step.
        
        # Alternative:
        # Keep a list of "candidates" that were recently blocked?
        # Or just pick random unused elements and check can_add.
        # The original 50/50 random walk is actually not too bad if we increase the number of trials
        # or make the swaps more effective.
        
        # Let's try a "Greedy Add" phase first.
        # Loop:
        #   Find any x not in S such that can_add(x).
        #   If found, add it.
        #   If not found, break.
        # This is O(N * |S|) per pass. |S|~500, N~1000. 500k ops.
        # We can do this a few times at the start.
        
        # Actually, the initial greedy already does this.
        # The local search is needed to escape local optima.
        
        # Let's stick to the random walk but improve the "Add" move.
        # Instead of random c, pick from a small set of "promising" candidates.
        # Or just keep it simple: random c.
        
        # The key improvement from inspiration 2 (score 1.0208) likely involves better swapping or initialization.
        # Let's try to bias swaps towards elements that are "central" or "constrained".
        
        # New Swap Strategy:
        # 1. Pick random c not in S.
        # 2. Calculate "cost" of adding c if we could remove one element.
        #    This is complex.
        
        # Let's try a different angle: Simulated Annealing?
        # Accept swaps that make the set "worse" (more conflicts) with some probability?
        # No, we must maintain validity.
        
        # Let's try to make the "Add" move smarter.
        # Maintain a list `unused` of elements not in S.
        # When we try to add, pick randomly from `unused`.
        # If it fails, do nothing.
        
        # To escape local optima, we need swaps that change the structure significantly.
        # Random swaps are good for this.
        
        # Let's just optimize the existing code for speed and run more trials.
        # The current code uses `S.remove(x)` which is O(N).
        # We can use a set for S? No, we need order for some things? No, S is just a set of ints.
        # But `remove` iterates.
        # Let's use a list for S and a set for in_S (already have boolean array).
        # `S.remove(x)` is O(N). N=500. 500 ops.
        # `add` is O(1) append + O(N) loop.
        
        # The bottleneck is the Python loops in add/remove/can_add.
        
        # Let's try to reduce the number of swaps needed by using a better initialization.
        # "Greedy with look-ahead"?
        
        # For now, I will implement a "Polish" step at the end of the trial.
        # After the random walk, do a full scan for any addable elements.
        
        # And I will increase the probability of Add moves to 0.7, as growth is the primary goal.
        
        # Also, I will track the "best" S found in this trial more aggressively.
        
        # Revised Loop:
        # 1. Polish (greedy add all possible).
        # 2. Random Walk (70% Add, 30% Swap).
        # 3. Polish again.
        
        def polish():
            nonlocal current_size
            changed = True
            while changed:
                changed = False
                for x in range(1, max_val + 1):
                    if not in_S[x] and can_add(x):
                        add(x)
                        current_size += 1
                        changed = True
                        if current_size > trial_best_size:
                            trial_best_size = current_size
                            trial_best_S = list(S)
        
        # Initial Polish
        polish()
        
        while time.time() < deadline - 2:
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.7:
                # Try Add
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
                if not S:
                    continue
                
                a = random.choice(S)
                
                # Pick c not in S
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
        
        # Final Polish
        polish()
        
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
