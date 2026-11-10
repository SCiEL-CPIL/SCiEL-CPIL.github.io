"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    MAX_SIZE = 64
    
    def check_sidon(A):
        """Check if A is a Sidon set (unique pair sums including a+a)."""
        sums = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(A):
        """Get the set of non-zero differences of A."""
        diffs = set()
        n = len(A)
        for i in range(n):
            for j in range(n):
                if i != j:
                    diffs.add(A[i] - A[j])
        return diffs

    def greedy_construct(order, max_time_check):
        """
        Greedily construct two Sidon sets A and B from the given order of candidates.
        Returns (A, B).
        """
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        # We need to check Sidon property efficiently.
        # For small sets (<= 64), O(k^2) check is fast enough if we do it incrementally.
        # But checking full Sidon property from scratch each time is O(k^2).
        # Total O(N * K^2) which is 256 * 64^2 ~ 1M ops. Very fast.
        
        for x in order:
            if time.time() > max_time_check:
                break
            
            # Try adding to A
            can_add_A = True
            # Check Sidon for A + [x]
            # Optimization: Only check sums involving x
            # Sums of existing A are already unique.
            # Need to ensure x+x and x+a (for a in A) are not in existing sums,
            # and x+x is not equal to any a+b in A.
            
            # Let's just do the full check for simplicity first, it's fast enough for k<=64.
            # Actually, let's do a slightly optimized check.
            # Existing sums of A: S_A
            # New sums: {x+x} U {x+a for a in A}
            # We need:
            # 1. x+x not in S_A
            # 2. x+a not in S_A for any a in A
            # 3. x+a1 != x+a2 for a1 != a2 (always true if a1!=a2)
            # 4. x+x != x+a (always true since x != 0)
            
            # So we just need to check if any of the new sums collide with old sums.
            # We can maintain a set of sums for A and B.
            
            # Let's refactor to maintain sum sets for speed.
            pass
        
        # Refactored greedy with maintained state
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if time.time() > max_time_check:
                break
            
            # Try adding to A
            new_sums_A = {x + x}
            for a in A:
                new_sums_A.add(x + a)
            
            if not (new_sums_A & sums_A):
                # Sidon condition satisfied for A
                # Check difference disjointness
                new_diffs_A = set()
                for a in A:
                    new_diffs_A.add(x - a)
                    new_diffs_A.add(a - x)
                
                if not (new_diffs_A & diffs_B):
                    # Valid addition to A
                    A.append(x)
                    sums_A.update(new_sums_A)
                    diffs_A.update(new_diffs_A)
                    continue
            
            # Try adding to B
            new_sums_B = {x + x}
            for b in B:
                new_sums_B.add(x + b)
            
            if not (new_sums_B & sums_B):
                # Sidon condition satisfied for B
                # Check difference disjointness
                new_diffs_B = set()
                for b in B:
                    new_diffs_B.add(x - b)
                    new_diffs_B.add(b - x)
                
                if not (new_diffs_B & diffs_A):
                    # Valid addition to B
                    B.append(x)
                    sums_B.update(new_sums_B)
                    diffs_B.update(new_diffs_B)
                    continue
        
        return A, B

    def local_improve(A, B):
        """
        Try to improve A and B by swapping elements or moving elements.
        Returns improved (A, B).
        """
        # This is a simple local search.
        # Try to find an element x in A that can be moved to B, or vice versa,
        # or find an element x not in A or B that can be added to A or B.
        # Since we want to maximize |A|+|B|, adding is better than swapping.
        # But the greedy might have missed some addable elements.
        
        # Let's re-scan all elements 1..256 and try to add them if possible.
        # This is essentially re-running greedy but with the current A, B as starting point?
        # No, the greedy already scanned in order. If it didn't add x, it's because
        # it couldn't be added to A or B at that time.
        # However, adding other elements later might have changed the constraints?
        # No, adding elements only adds constraints (more sums, more diffs).
        # So if x couldn't be added to A when A was smaller, it still can't be added
        # now that A is larger?
        # Wait. The Sidon condition depends on sums. If A grows, sums_A grows.
        # If x+a collided with a sum in A, adding more to A doesn't remove that collision.
        # So the set of "addable" elements only shrinks as A and B grow.
        # Therefore, if the greedy scan processed x and rejected it, it will never
        # become addable later in the same run.
        # So local improvement by adding is impossible if the greedy was complete.
        
        # However, the greedy choice of *which* set to add to might be suboptimal.
        # E.g., x could be added to A, but adding it to B would allow y to be added to A later,
        # whereas adding x to A prevents y from being added to A or B.
        # So we need to explore different orderings or choices.
        
        # Since the greedy is deterministic given the order, and we are doing random restarts,
        # we rely on different orders to find better packings.
        
        # One specific improvement: If |A| < |B|, maybe we should have put more in A?
        # The heuristic "try A first" biases towards A.
        # Let's try a "balanced" heuristic: try the side with fewer elements first?
        # Or try the side with fewer differences?
        
        # For now, let's just return A, B. The random restarts will handle diversity.
        return A, B

    best_size = 0
    best_A = []
    best_B = []
    
    # Strategy: Randomized Greedy with different orderings
    # Ordering 1: Ascending
    # Ordering 2: Descending
    # Ordering 3: Random
    # Ordering 4: Random with bias towards small numbers (Smaller numbers are "safer" for Sidon?)
    
    # Let's run multiple random restarts with different random seeds and orderings.
    
    while time.time() < deadline:
        # Choose a strategy
        strategy = random.randint(0, 2)
        
        if strategy == 0:
            # Ascending order
            order = list(range(1, N + 1))
        elif strategy == 1:
            # Descending order
            order = list(range(N, 0, -1))
        else:
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
            
        # Run greedy
        A, B = greedy_construct(order, deadline)
        
        # Check size
        current_size = len(A) + len(B)
        
        if current_size > best_size:
            best_size = current_size
            best_A = A
            best_B = B
            
            # If we hit the max possible size (unlikely but possible), stop
            if best_size >= 2 * MAX_SIZE:
                break
        
        # Optional: Local improvement pass
        # Since we established that adding is monotonic, we can't add more.
        # But we could try to swap elements to free up space?
        # E.g., remove x from A, try to add y to A, if y > x in "value" or something?
        # This is complex. Let's stick to random restarts first.
        
        # To diversify, we can also try "side preference" variations
        # e.g., always try B first, or random side preference.
        # The current greedy tries A first, then B.
        # Let's modify greedy_construct to accept a preference.
        
    # Final verification (just in case)
    if not check_sidon(best_A) or not check_sidon(best_B):
        # Fallback to a very simple valid construction if something went wrong
        # e.g., A = [1], B = [2]
        return [[1], [2]]
    
    if get_diffs(best_A) & get_diffs(best_B):
        # Fallback
        return [[1], [2]]
        
    return [best_A, best_B]

# EVOLVE-BLOCK-END
