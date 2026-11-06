import random
import time

def construct():
    n = 128
    target_lo = 128
    target_hi = 256
    
    # State:
    # A: set of elements in A
    # counts: array of size 257, counts[s] = number of pairs in A summing to s
    
    def get_initial_A():
        # Heuristic start: include 0, 128, and some spread
        A = set()
        A.add(0)
        A.add(128)
        # Add random elements to start
        for i in range(10):
            A.add(random.randint(0, 128))
        return A

    def init_counts(A):
        counts = [0] * 257
        elems = list(A)
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                counts[s] += 1
        return counts

    def calc_cost(counts, A):
        # Penalty for uncovered targets
        uncovered = 0
        peak = 0
        for s in range(target_lo, target_hi + 1):
            c = counts[s]
            if c == 0:
                uncovered += 1
            if c > peak:
                peak = c
        # Objective: Minimize (uncovered * 1000 + peak)
        # We want covered first, then low peak.
        return uncovered * 1000 + peak

    def try_move(A, counts, x, add):
        """
        Attempt to add or remove x from A.
        Returns (new_A, new_counts, cost_change)
        Modifies A and counts in place for speed, then reverts if needed? 
        Better: Calculate delta, apply, check.
        """
        if add:
            if x in A:
                return False
            # Apply addition
            old_len = len(A)
            A.add(x)
            # Update counts
            counts[2 * x] += 1
            for y in A:
                if y != x:
                    counts[x + y] += 1
            
            # Calculate new cost
            # We need to check if this improves things.
            # To avoid O(129) cost calc every time, we can track uncovered and peak.
            # But peak change is tricky. 
            # Let's just recalc cost. It's O(129).
            cost = calc_cost(counts, A)
            
            # Revert if we decide not to take it (handled by caller)
            # For now, just return the state.
            return cost
        else:
            if x not in A:
                return False
            # Apply removal
            A.remove(x)
            counts[2 * x] -= 1
            for y in A:
                counts[x + y] -= 1
            
            cost = calc_cost(counts, A)
            return cost

    # Simulated Annealing
    start_time = time.time()
    time_limit = 55.0 # Leave buffer
    
    best_A = None
    best_cost = float('inf')
    
    # Multiple restarts
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        A = get_initial_A()
        counts = init_counts(A)
        current_cost = calc_cost(counts, A)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = set(A)
            
        T = 50.0
        T_min = 0.1
        cooling = 0.999
        
        while time.time() - start_time < time_limit and T > T_min:
            # Pick a random move: Add or Remove
            # Bias towards keeping size reasonable? 
            # Let's pick a random element from 0..128
            x = random.randint(0, 128)
            
            if x in A:
                # Try remove
                old_A = set(A)
                old_counts = counts.copy()
                
                A.remove(x)
                counts[2 * x] -= 1
                for y in A:
                    counts[x + y] -= 1
                
                new_cost = calc_cost(counts, A)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                else:
                    # Revert
                    A = old_A
                    counts = old_counts
            else:
                # Try add
                old_A = set(A)
                old_counts = counts.copy()
                
                A.add(x)
                counts[2 * x] += 1
                for y in A:
                    if y != x:
                        counts[x + y] += 1
                
                new_cost = calc_cost(counts, A)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                else:
                    # Revert
                    A = old_A
                    counts = old_counts
            
            T *= cooling
            
            # If we found a perfect solution (cost == peak, uncovered == 0), 
            # we can try to lower peak further or stop if peak is 1?
            # Peak 1 is likely impossible for this density.
            # Let's just let SA run.
            
            # Early termination if cost is very low?
            if current_cost < 10:
                 # Try to optimize further with local search?
                 # For now, just keep cooling slowly
                 T *= 0.95

    return sorted(best_A)
