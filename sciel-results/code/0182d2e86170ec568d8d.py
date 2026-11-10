import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    num_targets = target_max - target_min + 1
    
    # Pre-compute target indices for faster access? 
    # counts is a list of 257 ints.
    
    class State:
        __slots__ = ['A', 'counts', 'uncovered', 'peak', 'sq']
        
        def __init__(self, A):
            self.A = set(A)
            self.counts = [0] * 257
            self.uncovered = 0
            self._update_counts()
            self._eval()
            
        def _update_counts(self):
            self.counts = [0] * 257
            A_list = sorted(self.A)
            for i in range(len(A_list)):
                ai = A_list[i]
                for j in range(i, len(A_list)):
                    s = ai + A_list[j]
                    if s <= 256:
                        self.counts[s] += 1
            # Count uncovered
            self.uncovered = 0
            for t in range(target_min, target_max + 1):
                if self.counts[t] == 0:
                    self.uncovered += 1

        def _eval(self):
            if self.uncovered > 0:
                self.peak = 10**9
                self.sq = 10**9
                return
            # Compute peak and sq for targets
            c_slice = self.counts[target_min:target_max+1]
            self.peak = max(c_slice)
            self.sq = sum(c*c for c in c_slice)

        def add(self, x):
            if x in self.A:
                return
            self.A.add(x)
            # Update counts
            for z in self.A:
                if z == x:
                    continue
                s = x + z
                if s <= 256:
                    if s >= target_min and s <= target_max:
                        if self.counts[s] == 0:
                            self.uncovered -= 1
                    self.counts[s] += 1
            s = 2 * x
            if s <= 256:
                if s >= target_min and s <= target_max:
                    if self.counts[s] == 0:
                        self.uncovered -= 1
                self.counts[s] += 1
            self._eval()

        def remove(self, x):
            if x not in self.A:
                return
            self.A.remove(x)
            # Update counts
            for z in self.A:
                s = x + z
                if s <= 256:
                    self.counts[s] -= 1
                    if s >= target_min and s <= target_max:
                        if self.counts[s] == 0:
                            self.uncovered += 1
            s = 2 * x
            if s <= 256:
                self.counts[s] -= 1
                if s >= target_min and s <= target_max:
                    if self.counts[s] == 0:
                        self.uncovered += 1
            self._eval()

    def get_cost(state):
        if state.uncovered > 0:
            return 10**9 + state.uncovered
        return state.peak + 0.01 * state.sq

    def simulated_annealing(initial_A):
        state = State(initial_A)
        
        # Greedy fix if not covered
        if state.uncovered > 0:
            while state.uncovered > 0 and time.time() - start_time < time_limit - 2.0:
                # Find best element to add
                best_x = -1
                best_cov = -1
                # Optimization: only check candidates that might cover uncovered
                # For simplicity, check all not in A
                for x in range(n + 1):
                    if x in state.A:
                        continue
                    # Simulate add to count coverage
                    cov = 0
                    for z in state.A:
                        s = x + z
                        if target_min <= s <= target_max and state.counts[s] == 0:
                            cov += 1
                    s = 2 * x
                    if target_min <= s <= target_max and state.counts[s] == 0:
                        cov += 1
                    if cov > best_cov:
                        best_cov = cov
                        best_x = x
                if best_x == -1 or best_cov == 0:
                    break
                state.add(best_x)
                
        if state.uncovered > 0:
            return None, None, None

        best_local_peak = state.peak
        best_local_sq = state.sq
        best_local_A = set(state.A)
        
        temp = 5.0
        cooling = 0.9995
        min_temp = 0.05
        
        while time.time() - start_time < time_limit - 2.0:
            if temp < min_temp:
                break
            
            move_type = random.random()
            accepted = False
            x_out = -1
            x_in = -1
            
            if move_type < 0.5:
                # Swap
                if len(state.A) == 0:
                    continue
                x_out = random.choice(list(state.A))
                cands = [x for x in range(n + 1) if x not in state.A]
                if not cands:
                    continue
                x_in = random.choice(cands)
                
                state.remove(x_out)
                state.add(x_in)
                
                if state.uncovered == 0:
                    cost = get_cost(state)
                    best_cost = get_cost(State(best_local_A)) # This is expensive!
                    # Optimization: Store best_cost
                    # Let's store best_cost in outer scope or calculate incrementally?
                    # For now, calculate best_cost once per loop iteration? No, best_local changes.
                    # Let's just compare peak/sq directly.
                    if state.peak < best_local_peak or (state.peak == best_local_peak and state.sq < best_local_sq):
                        best_local_peak = state.peak
                        best_local_sq = state.sq
                        best_local_A = set(state.A)
                        accepted = True
                    else:
                        # Calculate delta for SA
                        # We need cost of current and best
                        curr_cost = state.peak + 0.01 * state.sq
                        best_cost = best_local_peak + 0.01 * best_local_sq
                        delta = curr_cost - best_cost
                        if delta < 0:
                            accepted = True
                        else:
                            prob = math.exp(-delta / temp)
                            if random.random() < prob:
                                accepted = True
                # If uncovered > 0, reject
                
                if not accepted:
                    # Undo
                    state.remove(x_in)
                    state.add(x_out)
                    
            elif move_type < 0.8:
                # Remove
                if len(state.A) <= 1:
                    continue
                x_out = random.choice(list(state.A))
                state.remove(x_out)
                
                if state.uncovered == 0:
                    if state.peak < best_local_peak or (state.peak == best_local_peak and state.sq < best_local_sq):
                        best_local_peak = state.peak
                        best_local_sq = state.sq
                        best_local_A = set(state.A)
                        accepted = True
                    else:
                        curr_cost = state.peak + 0.01 * state.sq
                        best_cost = best_local_peak + 0.01 * best_local_sq
                        delta = curr_cost - best_cost
                        if delta < 0:
                            accepted = True
                        else:
                            prob = math.exp(-delta / temp)
                            if random.random() < prob:
                                accepted = True
                if not accepted:
                    state.add(x_out)
                    
            else:
                # Add
                cands = [x for x in range(n + 1) if x not in state.A]
                if not cands:
                    continue
                x_in = random.choice(cands)
                state.add(x_in)
                
                if state.uncovered == 0:
                    if state.peak < best_local_peak or (state.peak == best_local_peak and state.sq < best_local_sq):
                        best_local_peak = state.peak
                        best_local_sq = state.sq
                        best_local_A = set(state.A)
                        accepted = True
                    else:
                        curr_cost = state.peak + 0.01 * state.sq
                        best_cost = best_local_peak + 0.01 * best_local_sq
                        delta = curr_cost - best_cost
                        if delta < 0:
                            accepted = True
                        else:
                            prob = math.exp(-delta / temp)
                            if random.random() < prob:
                                accepted = True
                if not accepted:
                    state.remove(x_in)
                    
            temp *= cooling
            
        return best_local_peak, best_local_sq, best_local_A

    best_global_peak = 10**9
    best_global_sq = 10**9
    best_global_A = None
    
    while time.time() - start_time < time_limit - 5.0:
        # Random Init
        size = random.randint(30, 60)
        init_A = set(random.sample(range(n + 1), size))
        
        res = simulated_annealing(init_A)
        if res and res[0] is not None:
            peak, sq, A_set = res
            if peak < best_global_peak or (peak == best_global_peak and sq < best_global_sq):
                best_global_peak = peak
                best_global_sq = sq
                best_global_A = A_set
                
    if best_global_A is None:
        # Fallback: Full set
        return list(range(n + 1))
        
    return sorted(list(best_global_A))
