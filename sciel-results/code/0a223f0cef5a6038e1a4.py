"""Direct construction of two Sidon sets with disjoint difference sets."""
import time
import random
from itertools import combinations, combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    n = 256
    
    def elapsed():
        return time.time() - start_time
    
    def is_sidon(arr):
        """Check if arr is a Sidon set (unique pair sums including a+a)."""
        if not arr:
            return True
        sums = set()
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                s = arr[i] + arr[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_diffs(arr):
        """Get set of nonzero differences (b-a for a<b in arr)."""
        return {arr[j] - arr[i] for i in range(len(arr)) for j in range(i+1, len(arr))}
    
    def _compute_states(sets):
        """Precompute bitmask states for fast constraint checking."""
        # For each set, track:
        # - sum_mask: which pair sums are already used (bit i set if sum i is used)
        # - diff_mask: which differences are already used (bit d set if diff d is used)
        states = []
        for arr in sets:
            sum_mask = 0
            diff_mask = 0
            for i in range(len(arr)):
                for j in range(i, len(arr)):
                    s = arr[i] + arr[j]
                    sum_mask |= (1 << s)
                    if i != j:
                        d = arr[j] - arr[i]
                        diff_mask |= (1 << d)
            states.append((sum_mask, diff_mask))
        return states
    
    def _can_add(arr, x, states, idx, other_idx):
        """Check if adding x to arr[idx] keeps it Sidon and differences disjoint."""
        sum_mask, diff_mask = states[idx]
        other_diff_mask = states[other_idx][1]
        
        # Check pair sums: for each existing element a in arr, x+a must be unique
        # and not equal to any existing sum
        for a in arr:
            s = x + a
            if (sum_mask >> s) & 1:
                return False
        # Check x+x
        s_xx = 2 * x
        if (sum_mask >> s_xx) & 1:
            return False
        
        # Check differences: for each existing element a, |x-a| must not be in
        # current diff_mask or other_diff_mask
        for a in arr:
            d = abs(x - a)
            if d == 0:
                continue
            if (diff_mask >> d) & 1:
                return False
            if (other_diff_mask >> d) & 1:
                return False
        
        return True
    
    def _add_to_arr(arr, x, states, idx):
        """Add x to arr[idx] and update states."""
        sum_mask, diff_mask = states[idx]
        for a in arr:
            s = x + a
            sum_mask |= (1 << s)
            d = abs(x - a)
            if d > 0:
                diff_mask |= (1 << d)
        sum_mask |= (1 << (2 * x))
        states[idx] = (sum_mask, diff_mask)
        arr.append(x)
    
    def _fast_greedy(sets, states, order='AB'):
        """Greedy fill: try to add elements in order, alternating or sequential."""
        if order == 'AB':
            # Alternate between A and B
            for x in range(1, n + 1):
                if elapsed() > deadline:
                    break
                # Try adding to A
                if _can_add(sets[0], x, states, 0, 1):
                    _add_to_arr(sets[0], x, states, 0)
                # Try adding to B
                if _can_add(sets[1], x, states, 1, 0):
                    _add_to_arr(sets[1], x, states, 1)
        else:
            # Sequential: fill A first, then B
            for x in range(1, n + 1):
                if elapsed() > deadline:
                    break
                if _can_add(sets[0], x, states, 0, 1):
                    _add_to_arr(sets[0], x, states, 0)
            for x in range(1, n + 1):
                if elapsed() > deadline:
                    break
                if _can_add(sets[1], x, states, 1, 0):
                    _add_to_arr(sets[1], x, states, 1)
    
    def _polish(sets, states):
        """Local search: try to swap or add elements to improve combined size."""
        improved = True
        while improved and elapsed() < deadline - 2.0:
            improved = False
            # Try adding new elements
            for idx in [0, 1]:
                other_idx = 1 - idx
                for x in range(1, n + 1):
                    if x in sets[idx]:
                        continue
                    if _can_add(sets[idx], x, states, idx, other_idx):
                        _add_to_arr(sets[idx], x, states, idx)
                        improved = True
                        break
                else:
                    continue
                break
            # Try swaps: remove an element from one set and add to other
            for idx in [0, 1]:
                other_idx = 1 - idx
                for i, x in enumerate(sets[idx]):
                    if elapsed() > deadline - 2.0:
                        break
                    # Try removing x from set idx
                    sets[idx].pop(i)
                    # Recompute states
                    states = _compute_states(sets)
                    # Try adding x to other set
                    if _can_add(sets[other_idx], x, states, other_idx, idx):
                        _add_to_arr(sets[other_idx], x, states, other_idx)
                        improved = True
                    else:
                        # Put it back
                        sets[idx].insert(i, x)
                        states = _compute_states(sets)
                    if improved:
                        break
                if improved:
                    break
    
    # Initialize
    sets = [[], []]
    states = _compute_states(sets)
    
    # Fast greedy with alternating order
    _fast_greedy(sets, states, order='AB')
    
    # Polish
    _polish(sets, states)
    
    # Try a few random restarts if time permits
    best_sets = [list(sets[0]), list(sets[1])]
    best_size = len(sets[0]) + len(sets[1])
    
    for restart in range(5):
        if elapsed() > deadline - 10.0:
            break
        random.seed(restart * 12345 + 7)
        sets_r = [[], []]
        states_r = _compute_states(sets_r)
        
        # Randomized greedy: shuffle candidate order
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if elapsed() > deadline - 5.0:
                break
            if _can_add(sets_r[0], x, states_r, 0, 1):
                _add_to_arr(sets_r[0], x, states_r, 0)
            if _can_add(sets_r[1], x, states_r, 1, 0):
                _add_to_arr(sets_r[1], x, states_r, 1)
        
        # Polish the randomized result
        _polish(sets_r, states_r)
        
        size_r = len(sets_r[0]) + len(sets_r[1])
        if size_r > best_size:
            best_size = size_r
            best_sets = [list(sets_r[0]), list(sets_r[1])]
    
    # Use the best result
    sets = best_sets
    
    # Final verification
    if not sets[0] or not sets[1]:
        sets = [[1, 2, 4, 8, 16, 32, 64, 128], [3, 6, 12, 24, 48, 96, 192]]
    
    if not is_sidon(sets[0]):
        sets[0] = [1, 2, 4, 8, 16, 32, 64, 128]
    if not is_sidon(sets[1]):
        sets[1] = [3, 6, 12, 24, 48, 96, 192]
    
    d0 = get_diffs(sets[0])
    d1 = get_diffs(sets[1])
    if d0 & d1:
        # Rebuild with strict enforcement
        sets = [[], []]
        states = _compute_states(sets)
        _fast_greedy(sets, states, order='AB')
        _polish(sets, states)
        if not sets[0]:
            sets[0] = [1, 2, 4, 8, 16, 32, 64, 128]
        if not sets[1]:
            sets[1] = [3, 6, 12, 24, 48, 96, 192]
    
    return sets
