import time
import random

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin for 90s limit
    
    # Initial Greedy Solution
    def greedy_init():
        selected = set()
        reachable = {0}
        # Try to add elements in descending order
        for x in range(100, 0, -1):
            new_reachable = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_reachable:
                selected.add(x)
                reachable.update(new_reachable)
        return selected, reachable

    S, R = greedy_init()
    
    # Local Search
    # We maintain S as a set and R as the set of reachable sums using elements in S
    # Note: R calculation is O(|S| * |R|). |R| <= 201. |S| <= 100.
    # Recalculating R from scratch is ~20,000 ops, which is fast.
    
    def recalc_reachable(S):
        reachable = {0}
        for x in S:
            new_reachable = {s + x for s in reachable if s + x <= 200}
            reachable.update(new_reachable)
            if 200 in reachable:
                return None # Invalid
        return reachable

    def is_valid(S):
        # Quick check: if sum(S) < 200, valid.
        # Else check reachable
        if sum(S) < 200:
            return True
        R_new = recalc_reachable(S)
        return R_new is not None

    # Optimization: Keep R updated incrementally where possible, 
    # but for swaps/adds, full recalc is safer and fast enough for |S|~50-70.
    
    missing = set(range(1, 101)) - S
    
    while time.time() < deadline:
        # Try to add elements
        added = False
        # Shuffle missing to try different orders
        # But we want to prioritize adding. 
        # Let's iterate through missing.
        for x in list(missing):
            if time.time() > deadline:
                break
            # Check if adding x is valid
            # New reachable sums would be R U {s+x for s in R}
            # We need 200 not in (R U {s+x for s in R if s+x<=200})
            # Since 200 not in R (S is valid), we just need 200 not in {s+x for s in R if s+x<=200}
            # i.e., (200-x) not in R
            if (200 - x) not in R:
                S.add(x)
                missing.remove(x)
                # Update R
                new_sums = {s + x for s in R if s + x <= 200}
                R.update(new_sums)
                added = True
                # Restart loop to check if more can be added with new R
                break
        
        if not added:
            # No element can be added directly.
            # Try swaps to improve structure
            # Pick a random x from missing, random y from S
            if missing and S:
                x = random.choice(list(missing))
                y = random.choice(list(S))
                # Try swap y -> x
                S_test = S.copy()
                S_test.remove(y)
                S_test.add(x)
                
                # Check validity of S_test
                # Efficient check: 
                # Remove y's contribution from R? Hard.
                # Just recalc for S_test.
                R_test = recalc_reachable(S_test)
                if R_test is not None:
                    # Swap is valid
                    S = S_test
                    missing = set(range(1, 101)) - S
                    R = R_test
                    # After swap, maybe we can add something?
                    # The loop will continue and check additions again.

    # Ensure return is a list or set? "Return a nonempty subset"
    # Seed returns a list. Let's return a list.
    return list(S)
