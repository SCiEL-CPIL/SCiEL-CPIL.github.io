"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Find the largest Sidon set in [1, 284] containing {1, 2, 5, 11}.
    
    Strategy:
    1. Start with the greedy solution to get a baseline.
    2. Use local search (hill climbing with random restarts) to try to improve.
       - Move: remove one element, add one different element.
       - Or: remove one element (to allow adding two later, but simple swap is faster).
       - Actually, standard local search for maximal independent set:
         - State: set of elements S.
         - Validity: all pairwise sums distinct.
         - Objective: maximize |S|.
         - Moves:
           - Add a random candidate x not in S if valid.
           - If no add is possible (stuck), remove a random element and try to add others.
           - Or: swap: remove x, add y.
    
    Given the small size (n=284), we can afford a decent amount of local search.
    The seed solution is likely size 16-18. The max is 20.
    """
    
    def is_valid_sidon(S):
        """Check if S is a valid Sidon set."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_conflicts(x, S):
        """
        Return set of elements in S that conflict with x.
        Conflict if x + y == x' + y' for some x, y, x', y' in S U {x}.
        Specifically, adding x creates new sums x+y for y in S, and 2x.
        These must not clash with existing sums or each other.
        """
        # This function is useful for quick checks during search.
        # But for simplicity in the swap logic, we might just re-check validity.
        # However, for speed, let's compute the "forbidden" values.
        pass

    def try_add(S, x):
        """Try to add x to S. Return new S if valid, else None."""
        if x in S:
            return None
        new_S = S | {x}
        # Check validity efficiently
        # Existing sums
        sums = set()
        for i in range(len(new_S)):
            for j in range(i, len(new_S)):
                s = new_S[i] + new_S[j]
                if s in sums:
                    return None
                sums.add(s)
        return new_S

    def greedy_construct():
        a = [1, 2, 5, 11]
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        
        for x in range(12, 285):
            # Check if x can be added
            ok = True
            # Check 2x
            if 2*x in sums:
                ok = False
            # Check x + y
            if ok:
                for y in a:
                    if x + y in sums:
                        ok = False
                        break
            if ok:
                # Check that x+y1 != x+y2 (impossible since y1!=y2)
                # Check that x+y != 2x (implies y=x, impossible)
                # So just update
                a.append(x)
                sums.add(2*x)
                for y in a[:-1]:
                    sums.add(x + y)
        return set(a)

    # 1. Get baseline
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Deadline guard
    deadline = time.time() + 110  # Leave 10s margin for 120s limit
    
    # 2. Local Search
    # We will use a simple hill-climbing / simulated annealing style approach.
    # Since we want to MAXIMIZE size, we prioritize moves that increase size.
    # If stuck (cannot increase size), we allow moves that keep size same (shuffle)
    # or even decrease size (to escape local minima) but with penalty.
    
    # Precompute all possible elements
    all_elements = set(range(1, 285))
    prescribed = {1, 2, 5, 11}
    
    # Helper to check if a set is valid and return its size
    # We assume inputs are mostly valid or close to valid.
    
    def valid_size(S):
        """Returns size of largest valid Sidon subset? No, just check if S is valid."""
        if not is_valid_sidon(S):
            return -1
        return len(S)

    # Better: Incremental validity check is hard to maintain with swaps easily.
    # Given n=284, O(n^2) check is ~80k ops. 10000 restarts * 100 steps = 1M checks?
    # 1M * 80k = 80 billion ops. Too slow in Python.
    
    # We need a faster validity check or a smarter search.
    # Let's use the conflict graph idea implicitly.
    # When considering adding x to S, we only need to check sums involving x.
    
    def can_add(S, sums_cache, x):
        """
        Check if x can be added to S.
        sums_cache: set of all sums a+b for a,b in S (a<=b).
        Returns (bool, new_sums)
        """
        # New sums: 2x, and x+y for y in S
        # Check 2x
        if 2*x in sums_cache:
            return False, None
        
        new_sums_to_check = [2*x]
        for y in S:
            s = x + y
            if s in sums_cache:
                return False, None
            new_sums_to_check.append(s)
        
        # Check if new sums conflict with each other
        # 2x == x+y => x=y (impossible since x not in S)
        # x+y1 == x+y2 => y1=y2 (impossible)
        # So only need to check against sums_cache.
        
        # If valid, return the new sums to update cache
        return True, set(new_sums_to_check)

    def remove_and_update(S, sums_cache, x):
        """
        Remove x from S and update sums_cache.
        Returns (new_S, new_sums_cache)
        """
        new_S = S - {x}
        # Sums to remove: 2x, and x+y for y in new_S
        # Note: sums like y1+y2 where y1,y2 != x are unchanged.
        # We need to rebuild sums_cache or update it.
        # Rebuilding is O(n^2). Updating is O(n).
        # Let's update.
        
        # Identify sums to remove
        sums_to_remove = {2*x}
        for y in new_S:
            sums_to_remove.add(x + y)
        
        # Subtract from cache
        new_sums_cache = sums_cache - sums_to_remove
        
        return new_S, new_sums_cache

    # Initialize cache for best_set
    def build_cache(S):
        sums = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        return sums

    current_S = set(best_set)
    current_sums = build_cache(current_S)
    
    # Random restarts
    random.seed(42)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Start from current best or a perturbed version?
        # Let's start from the greedy solution each time, but perturb it slightly.
        # Or start from best found so far.
        
        S = set(best_set)
        sums = build_cache(S)
        
        # Perturbation: remove k random non-prescribed elements
        k = random.randint(1, 3)
        removable = [x for x in S if x not in prescribed]
        if removable:
            to_remove = random.sample(removable, min(k, len(removable)))
            for x in to_remove:
                S, sums = remove_and_update(S, sums, x)
        
        # Hill climbing phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any element
            # To avoid O(N) scan every time, we can keep a list of candidates.
            # But N=284 is small.
            candidates = [x for x in range(1, 285) if x not in S]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                ok, new_sums = can_add(S, sums, x)
                if ok:
                    S.add(x)
                    sums |= new_sums
                    improved = True
                    added = True
                    # Restart candidate loop? Or continue?
                    # Continuing might allow adding more in one pass.
                    # But sums changed, so previous 'ok' checks are invalid.
                    # Break and restart loop.
                    break
            
            if not added:
                # Try swaps: remove one, add one
                # This is more expensive.
                # Only do this if we haven't improved in a while?
                # Or just do a few random swaps.
                
                # Pick a random element to remove
                removable = [x for x in S if x not in prescribed]
                if not removable:
                    break
                
                x_remove = random.choice(removable)
                
                # Try to find a replacement
                candidates = [x for x in range(1, 285) if x not in S]
                random.shuffle(candidates)
                
                swap_done = False
                for x_add in candidates:
                    # Temporarily remove x_remove
                    S_temp, sums_temp = remove_and_update(S, sums, x_remove)
                    ok, new_sums = can_add(S_temp, sums_temp, x_add)
                    if ok:
                        # Perform swap
                        S = S_temp
                        sums = sums_temp
                        S.add(x_add)
                        sums |= new_sums
                        improved = True
                        swap_done = True
                        break
                
                if not swap_done:
                    # Try removing an element without adding (to allow 2 adds later?)
                    # Or just accept stuck.
                    # Let's try a "kick": remove a random element and don't add back immediately,
                    # then try to fill.
                    S, sums = remove_and_update(S, sums, x_remove)
                    improved = True # Force loop to continue to try adding
                
                # If we did a swap or kick, loop continues to try adding.
                # If we didn't do anything (no removable?), break.
                
                if not swap_done and not removable:
                    break

        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            # print(f"New best size: {best_size}")

    # Final validation
    if not is_valid_sidon(best_set):
        # Fallback to greedy if something went wrong
        best_set = greedy_construct()
        
    # Ensure prescribed elements are present (they should be, but just in case)
    # The search never removes prescribed elements.
    
    return list(best_set)
# EVOLVE-BLOCK-END
