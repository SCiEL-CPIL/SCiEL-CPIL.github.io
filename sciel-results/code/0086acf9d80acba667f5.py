import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    N = 1000
    initial_size = 150
    best_set = set()
    best_size = 0
    
    # Efficient state management
    class B2State:
        def __init__(self, S):
            self.S = set(S)
            self.arr = np.zeros(N + 1, dtype=int)
            for x in self.S:
                self.arr[x] = 1
            self.conv = np.convolve(self.arr, self.arr, mode='full')
            self.violations = self._calc_violations()
            
        def _calc_violations(self):
            viol = 0
            for s in range(2, 2 * N + 1):
                count_ord = self.conv[s]
                d = 1 if (s % 2 == 0 and (s // 2) in self.S) else 0
                max_allowed = 3 if d == 1 else 4
                if count_ord > max_allowed:
                    viol += count_ord - max_allowed
            return viol
            
        def remove(self, x):
            if x not in self.S:
                return
            self.S.remove(x)
            # Update arr
            self.arr[x] = 0
            # Update conv
            # conv_new = conv_old - shift(arr_old, x) - shift(arr_old, -x) + delta(2x)
            # shift(arr, x)[s] = arr[s-x]
            # This is effectively convolving with e_x
            # e_x is 1 at x, 0 elsewhere.
            # conv(e_x, arr) = arr shifted by x.
            # conv(arr, e_x) = arr shifted by -x (which is arr reversed and shifted? No, convolution is commutative).
            # Let's just recompute conv? 
            # np.convolve is fast enough for 1000 size? 
            # FFT conv is O(N log N). 1000 is small.
            # Recalculating is simpler and less error-prone.
            self.conv = np.convolve(self.arr, self.arr, mode='full')
            self.violations = self._calc_violations()
            
        def add(self, x):
            if x in self.S:
                return
            self.S.add(x)
            self.arr[x] = 1
            self.conv = np.convolve(self.arr, self.arr, mode='full')
            self.violations = self._calc_violations()

    # Phase 1: Random Init and Prune
    for attempt in range(30):
        if time.time() > deadline - 30:
            break
            
        S = set(random.sample(range(1, N + 1), initial_size))
        state = B2State(S)
        
        # Prune
        for _ in range(500):
            if state.violations == 0:
                break
            if time.time() > deadline - 30:
                break
                
            # Find a bad element
            # Heuristic: pick random element from S
            # Better: pick element from a violated sum
            # We don't have the list of violated sums in state.
            # Let's just pick random.
            # Or we can store violated sums.
            
            # To be safe and simple, pick random.
            # But random might be slow to converge.
            # Let's try to identify a bad element more smartly.
            # We can check a few random elements and pick the one that reduces violations most.
            
            best_elem = None
            best_viol_after = state.violations
            candidates = random.sample(list(state.S), min(10, len(state.S)))
            
            for cand in candidates:
                # Try removing cand
                # We need to simulate removal
                # To avoid full conv recompute for each candidate, we can just pick one
                # Let's just pick the first one that seems bad?
                # Actually, just removing a random element from a violated sum is good.
                # Let's find a violated sum.
                
                # This requires iterating conv.
                # Let's just do random removal for now, it's robust.
                best_elem = cand
                break # Just take the first candidate to save time
                
            if best_elem:
                state.remove(best_elem)
                
        if state.violations == 0:
            if len(state.S) > best_size:
                best_size = len(state.S)
                best_set = set(state.S)
                
        # Phase 2: Local Search
        current_state = state
        no_improve = 0
        
        while time.time() < deadline - 5:
            if no_improve > 100:
                break
                
            # Try to add
            candidates = [x for x in range(1, N + 1) if x not in current_state.S]
            if not candidates:
                break
            random.shuffle(candidates)
            
            added = False
            for cand in candidates[:10]:
                # Check if adding cand keeps it valid
                # We can simulate
                # To avoid creating a new B2State object every time (overhead),
                # we can just check the change in violations.
                # But B2State.add is simple.
                
                # Let's create a temporary state or just check manually
                # Manual check:
                # New conv = old conv + shifts
                # This is complex.
                # Let's just use the class.
                
                temp_state = B2State(current_state.S)
                temp_state.add(cand)
                if temp_state.violations == 0:
                    current_state = temp_state
                    added = True
                    if len(current_state.S) > best_size:
                        best_size = len(current_state.S)
                        best_set = set(current_state.S)
                    no_improve = 0
                    break
                else:
                    # Try swap?
                    pass
            
            if not added:
                no_improve += 1
                # Try swap
                if len(current_state.S) > 0:
                    rem = random.choice(list(current_state.S))
                    add_cands = [x for x in range(1, N + 1) if x not in current_state.S]
                    if add_cands:
                        add = random.choice(add_cands)
                        temp_state = B2State(current_state.S)
                        temp_state.remove(rem)
                        temp_state.add(add)
                        if temp_state.violations == 0:
                            current_state = temp_state
                            # Size same, but might unlock additions
                            # Don't reset no_improve, but maybe don't increment?
                            # Let's treat as neutral.
                            pass

    return sorted(list(best_set))
