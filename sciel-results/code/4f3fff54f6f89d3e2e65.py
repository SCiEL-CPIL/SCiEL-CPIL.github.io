import time
import random

def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    
    Strategy:
    1. Greedy initialization.
    2. Efficient local search using bitsets and block counts.
    3. Randomized restarts/perturbations.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    # Precompute for each x, the list of y such that x and y are in a 3-AP with some z.
    # Actually, we need to update block counts dynamically.
    
    class Solver:
        def __init__(self):
            self.n = n
            self.in_set = [False] * (n + 1)
            self.block_count = [0] * (n + 1)
            self.current_sum = 0.0
            self.set_list = []
            
        def add(self, x):
            if self.in_set[x]:
                return
            # Check if x is blocked
            if self.block_count[x] > 0:
                return
            
            self.in_set[x] = True
            self.set_list.append(x)
            self.current_sum += recip[x]
            
            # Update block counts for all z in set_list
            # For each z in S, x and z form pairs that block new elements.
            for z in self.set_list[:-1]: # Exclude x itself
                # Pair (x, z)
                # 1. x, z are endpoints -> midpoint m = (x+z)/2
                if (x + z) % 2 == 0:
                    m = (x + z) // 2
                    if 1 <= m <= self.n and m != x and m != z:
                        self.block_count[m] += 1
                        if self.block_count[m] == 1:
                            # m became blocked
                            pass # No need to do anything special yet
                
                # 2. x is endpoint, z is middle -> other endpoint y = 2z - x
                y = 2 * z - x
                if 1 <= y <= self.n and y != x and y != z:
                    self.block_count[y] += 1
                    if self.block_count[y] == 1:
                        pass
                
                # 3. z is endpoint, x is middle -> other endpoint y = 2x - z
                y = 2 * x - z
                if 1 <= y <= self.n and y != x and y != z:
                    self.block_count[y] += 1
                    if self.block_count[y] == 1:
                        pass

        def remove(self, x):
            if not self.in_set[x]:
                return
            
            self.in_set[x] = False
            self.set_list.remove(x)
            self.current_sum -= recip[x]
            
            # Update block counts for all z in set_list (which no longer contains x)
            for z in self.set_list:
                # Pair (x, z) was contributing to block counts. Remove them.
                # 1. Midpoint m
                if (x + z) % 2 == 0:
                    m = (x + z) // 2
                    if 1 <= m <= self.n and m != x and m != z:
                        self.block_count[m] -= 1
                        if self.block_count[m] == 0:
                            pass # m became unblocked
                
                # 2. y = 2z - x
                y = 2 * z - x
                if 1 <= y <= self.n and y != x and y != z:
                    self.block_count[y] -= 1
                    if self.block_count[y] == 0:
                        pass
                
                # 3. y = 2x - z
                y = 2 * x - z
                if 1 <= y <= self.n and y != x and y != z:
                    self.block_count[y] -= 1
                    if self.block_count[y] == 0:
                        pass

        def can_add(self, x):
            return not self.in_set[x] and self.block_count[x] == 0

        def greedy_fill(self):
            # Add all possible smallest elements
            changed = True
            while changed:
                changed = False
                for x in range(1, self.n + 1):
                    if self.can_add(x):
                        self.add(x)
                        changed = True

    # Initialize
    solver = Solver()
    for x in range(1, n + 1):
        if solver.can_add(x):
            solver.add(x)
    solver.greedy_fill()
    
    best_sum = solver.current_sum
    best_state = (solver.in_set[:], solver.block_count[:], solver.set_list[:], solver.current_sum)
    
    def local_search(solver):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_sum = solver.current_sum
            S = solver.set_list[:]
            
            # Try all swaps: remove x, add y
            # y must be unblocked after removing x
            # To find candidate y's, we can just check all y not in S?
            # That's O(N).
            # Better: Only y that were blocked by x might become unblocked.
            # But y could also be unblocked already? No, if it was unblocked, greedy_fill would have added it.
            # So all unblocked y are in S.
            # Thus, candidate y's are those that are currently blocked, but become unblocked when x is removed.
            
            for x in S:
                if time.time() > deadline - 5.0:
                    break
                
                # Temporarily remove x
                solver.remove(x)
                
                # Find candidate y's
                # y is a candidate if solver.can_add(y) is True.
                # Since we removed x, some y's might now be addable.
                # We want the one with max weight (smallest index).
                best_y = -1
                best_gain = 0.0
                
                # Optimization: Only check y that were blocked by x?
                # Actually, just iterate 1..N? N=256 is small.
                for y in range(1, n + 1):
                    if solver.can_add(y):
                        gain = recip[y]
                        # We are removing x, so net change is gain - recip[x]
                        # We want to maximize net change.
                        # Since we are iterating y, we can just pick the best y.
                        # But we must ensure y != x (x is removed, so y=x is not in set, but can_add(x) might be true?
                        # If we remove x, x is not in set. Is x blocked?
                        # x was in set, so it wasn't blocked. After removal, x is not in set.
                        # Is x blocked by remaining S?
                        # If we add x back, it must be valid.
                        # So can_add(x) should be True.
                        # But we want to add a *different* y to improve.
                        # If we add x back, sum is same. No improvement.
                        # So we should ignore y=x.
                        if y == x:
                            continue
                        
                        # If gain - recip[x] > 0, it's an improvement.
                        if gain > recip[x]:
                            if best_y == -1 or gain > recip[best_y]:
                                best_y = y
                                best_gain = gain
                
                if best_y != -1:
                    solver.add(best_y)
                    if solver.current_sum > current_sum + 1e-10:
                        improved = True
                        current_sum = solver.current_sum
                        S = solver.set_list[:]
                        # Restart search from new state? Or continue?
                        # Restarting is safer to avoid cycles.
                        break
                else:
                    # No improvement found by removing x and adding best y
                    # Revert removal of x
                    solver.add(x)
            
            # Also try adding elements if any became unblocked?
            # Greedy fill handles that, but we should call it if we didn't swap.
            if not improved:
                solver.greedy_fill()
                if solver.current_sum > current_sum + 1e-10:
                    improved = True
        
        return solver

    solver = local_search(solver)
    best_sum = solver.current_sum
    best_state = (solver.in_set[:], solver.block_count[:], solver.set_list[:], solver.current_sum)
    
    random.seed(42)
    
    while time.time() < deadline - 10.0:
        # Perturb
        solver = Solver()
        # Restore best state
        solver.in_set = best_state[0][:]
        solver.block_count = best_state[1][:]
        solver.set_list = best_state[2][:]
        solver.current_sum = best_state[3]
        
        # Remove 1-3 random elements
        num_removals = random.randint(1, 3)
        candidates = solver.set_list[:]
        random.shuffle(candidates)
        for _ in range(min(num_removals, len(candidates))):
            rem_val = candidates.pop()
            solver.remove(rem_val)
        
        solver.greedy_fill()
        solver = local_search(solver)
        
        if solver.current_sum > best_sum + 1e-10:
            best_sum = solver.current_sum
            best_state = (solver.in_set[:], solver.block_count[:], solver.set_list[:], solver.current_sum)

    result = best_state[2][:]
    result.sort()
    
    # Final validation
    assert len(result) > 0
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
    
    return result
