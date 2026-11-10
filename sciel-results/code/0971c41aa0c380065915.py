import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    n = 128
    target_min = 128
    target_max = 256
    total_targets = target_max - target_min + 1  # 129
    
    # Precompute for fast coverage check
    # We'll maintain:
    # - counts[t] for t in [0, 256]
    # - uncovered: number of t in [target_min, target_max] with counts[t] == 0
    # - peak: max counts[t] for t in [target_min, target_max] (or large if uncovered)
    # - sum_sq: sum of counts[t]^2 for t in [target_min, target_max]
    
    # For incremental updates, when we add x to A:
    #   For each z in A (including z=x for the pair (x,x)):
    #     s = x + z
    #     if s <= 256:
    #       old_count = counts[s]
    #       if target_min <= s <= target_max:
    #         if old_count == 0: uncovered -= 1
    #         sum_sq += (old_count+1)^2 - old_count^2 = 2*old_count + 1
    #         new_count = old_count + 1
    #         if new_count > peak: peak = new_count
    #       counts[s] = old_count + 1
    # When we remove x from A:
    #   For each z in A_remaining (after removal, i.e., z in A\{x}):
    #     s = x + z
    #     if s <= 256:
    #       old_count = counts[s]
    #       new_count = old_count - 1
    #       if target_min <= s <= target_max:
    #         sum_sq += new_count^2 - old_count^2 = -2*old_count + 1
    #         if old_count == 1: uncovered += 1
    #         if old_count == peak and new_count < peak:
    #           # need to recompute peak (expensive) - handle lazily or track
    #       counts[s] = new_count
    #   Also handle s = 2*x
    #   For the pair (x,x): s = 2x
    #     same logic as above
    
    # To avoid expensive peak recomputation on removal, I'll track a multiset of counts
    # or just recompute peak when it might have changed. Given the small size, let me
    # use a different approach: maintain counts array and compute peak/sum_sq from
    # the slice when needed, but cache them and only recompute when the max element
    # is modified.
    
    # Actually, for speed, let me just recompute peak and sum_sq from the 129-element
    # slice each time. 129 elements is very fast in Python.
    
    # Even better: I'll maintain the counts array and compute peak and sum_sq
    # incrementally for additions (easy), and for removals I'll only recompute
    # peak if the removed element's count was equal to the current peak.
    
    # Let me use a simpler but correct approach: maintain counts, uncovered,
    # and recompute peak and sum_sq from the 129-element slice. This is O(129) per
    # evaluation, which is fine.
    
    class State:
        __slots__ = ['A', 'counts', 'uncovered', 'peak', 'sum_sq']
        
        def __init__(self, A_set):
            self.A = set(A_set)
            self.counts = [0] * 257
            self._compute_all()
        
        def _compute_all(self):
            self.counts = [0] * 257
            A_list = sorted(self.A)
            for i in range(len(A_list)):
                ai = A_list[i]
                for j in range(i, len(A_list)):
                    s = ai + A_list[j]
                    if s <= 256:
                        self.counts[s] += 1
            self._update_metrics()
        
        def _update_metrics(self):
            self.uncovered = 0
            self.peak = 0
            self.sum_sq = 0
            for t in range(target_min, target_max + 1):
                c = self.counts[t]
                if c == 0:
                    self.uncovered += 1
                else:
                    if c > self.peak:
                        self.peak = c
                    self.sum_sq += c * c
        
        def add(self, x):
            if x in self.A:
                return
            self.A.add(x)
            # Update counts for pairs involving x
            # Pairs: (x, z) for z in A, z != x, plus (x, x)
            for z in self.A:
                if z == x:
                    continue
                s = x + z
                if s <= 256:
                    old = self.counts[s]
                    new = old + 1
                    self.counts[s] = new
                    if target_min <= s <= target_max:
                        if old == 0:
                            self.uncovered -= 1
                        self.sum_sq += 2 * old + 1  # (old+1)^2 - old^2
                        if new > self.peak:
                            self.peak = new
            s = 2 * x
            if s <= 256:
                old = self.counts[s]
                new = old + 1
                self.counts[s] = new
                if target_min <= s <= target_max:
                    if old == 0:
                        self.uncovered -= 1
                    self.sum_sq += 2 * old + 1
                    if new > self.peak:
                        self.peak = new
        
        def remove(self, x):
            if x not in self.A:
                return
            self.A.remove(x)
            # Update counts for pairs involving x
            # After removal, pairs (x, z) for z in A (now A\{x}) are removed
            # Plus (x, x) is removed
            # We need to subtract these contributions
            # But we must be careful: we removed x from A first, so A is now A\{x}
            # The pairs that are no longer counted are: (x, z) for z in A\{x}, and (x, x)
            for z in self.A:  # This is A\{x} now
                s = x + z
                if s <= 256:
                    old = self.counts[s]
                    new = old - 1
                    self.counts[s] = new
                    if target_min <= s <= target_max:
                        self.sum_sq -= 2 * old - 1  # old^2 - (old-1)^2 = 2*old - 1
                        if old == 1:
                            self.uncovered += 1
                        # Peak might have decreased; we'll handle this lazily
            s = 2 * x
            if s <= 256:
                old = self.counts[s]
                new = old - 1
                self.counts[s] = new
                if target_min <= s <= target_max:
                    self.sum_sq -= 2 * old - 1
                    if old == 1:
                        self.uncovered += 1
            
            # After removals, peak might be stale. Recompute if needed.
            # Simple approach: if the peak might have changed, recompute.
            # To keep it simple and correct, recompute peak from the slice
            # only when we suspect it changed (i.e., when we removed from a position
            # that had count == peak). But for safety, let's just recompute always.
            # Actually, let's track whether peak might have decreased.
            # For now, recompute peak from slice - it's only 129 elements.
            # But this defeats the purpose of incremental. Let me use a flag.
            # Actually, the simplest correct approach: maintain peak correctly.
            # On removal, if old count was == peak, we need to find new peak.
            # Let me just recompute peak when necessary.
            
            # To keep things simple and avoid bugs, let me recompute peak
            # only when we removed from a target sum that had count == peak.
            # I'll track this with a flag.
            
            # Actually, for correctness and simplicity, let me just recompute
            # peak from the slice every time we remove. It's 129 iterations.
            self._recompute_peak()
        
        def _recompute_peak(self):
            self.peak = 0
            for t in range(target_min, target_max + 1):
                c = self.counts[t]
                if c > self.peak:
                    self.peak = c
            
            # Also verify uncovered is correct (should be, but just in case)
            # Actually, uncovered is maintained incrementally and should be correct.
        
        def get_cost(self):
            if self.uncovered > 0:
                return 10**9 + self.uncovered
            return self.peak + 0.01 * self.sum_sq
    
    def simulated_annealing(initial_A):
        state = State(initial_A)
        
        # Greedy fix if not covered
        if state.uncovered > 0:
            while state.uncovered > 0 and time.time() - start_time < time_limit - 3.0:
                # Find best element to add
                best_x = -1
                best_cov = -1
                for x in range(n + 1):
                    if x in state.A:
                        continue
                    # Count how many uncovered targets this would cover
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
                return None
        
        best_local = state.get_cost()
        best_local_peak = state.peak
        best_local_sq = state.sum_sq
        best_local_A = set(state.A)
        
        temp = 5.0
        cooling = 0.9997
        min_temp = 0.03
        
        while time.time() - start_time < time_limit - 3.0:
            if temp < min_temp:
                break
            
            move_type = random.random()
            accepted = False
            x_out = -1
            x_in = -1
            
            if move_type < 0.5:
                # Swap
                if len(state.A) == 0:
                    temp *= cooling
                    continue
                x_out = random.choice(list(state.A))
                cands = [x for x in range(n + 1) if x not in state.A]
                if not cands:
                    temp *= cooling
                    continue
                x_in = random.choice(cands)
                
                state.remove(x_out)
                state.add(x_in)
                
                if state.uncovered == 0:
                    cost = state.get_cost()
                    if cost < best_local:
                        best_local = cost
                        best_local_peak = state.peak
                        best_local_sq = state.sum_sq
                        best_local_A = set(state.A)
                        accepted = True
                    else:
                        delta = cost - best_local
                        if delta < 0:
                            accepted = True
                        else:
                            prob = math.exp(-delta / temp)
                            if random.random() < prob:
                                accepted = True
                # If uncovered > 0, reject
                
                if not accepted:
                    # Undo: remove x_in, add x_out
                    state.remove(x_in)
                    state.add(x_out)
                    
            elif move_type < 0.8:
                # Remove
                if len(state.A) <= 1:
                    temp *= cooling
                    continue
                x_out = random.choice(list(state.A))
                state.remove(x_out)
                
                if state.uncovered == 0:
                    cost = state.get_cost()
                    if cost < best_local:
                        best_local = cost
                        best_local_peak = state.peak
                        best_local_sq = state.sum_sq
                        best_local_A = set(state.A)
                        accepted = True
                    else:
                        delta = cost - best_local
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
                    temp *= cooling
                    continue
                x_in = random.choice(cands)
                state.add(x_in)
                
                if state.uncovered == 0:
                    cost = state.get_cost()
                    if cost < best_local:
                        best_local = cost
                        best_local_peak = state.peak
                        best_local_sq = state.sum_sq
                        best_local_A = set(state.A)
                        accepted = True
                    else:
                        delta = cost - best_local
                        if delta < 0:
                            accepted = True
                        else:
                            prob = math.exp(-delta / temp)
                            if random.random() < prob:
                                accepted = True
                if not accepted:
                    state.remove(x_in)
                    
            temp *= cooling
            
        # Return the best found
        if best_local_A is not None:
            return best_local_peak, best_local_sq, best_local_A
        else:
            return None
    
    # Multiple restarts with different initial sets
    best_global_peak = 10**9
    best_global_sq = 10**9
    best_global_A = None
    
    # Try a few different initial sizes and strategies
    while time.time() - start_time < time_limit - 5.0:
        # Random Init with varied sizes
        size = random.randint(35, 65)
        init_A = set(random.sample(range(n + 1), size))
        
        res = simulated_annealing(init_A)
        if res is not None:
            peak, sq, A_set = res
            if peak < best_global_peak or (peak == best_global_peak and sq < best_global_sq):
                best_global_peak = peak
                best_global_sq = sq
                best_global_A = A_set
    
    if best_global_A is None:
        # Fallback: full set
        return list(range(n + 1))
        
    return sorted(list(best_global_A))
