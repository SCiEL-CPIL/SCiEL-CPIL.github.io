import time
import math
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 55.0
    
    # 1. Generate Thue-Morse sequence as a strong initial feasible solution
    def thue_morse(n):
        seq = []
        for i in range(n):
            c = bin(i).count('1')
            seq.append(-1 if c % 2 == 0 else 1)
        return seq

    seq = thue_morse(n)
    
    # 2. Fast cost function using NumPy
    # We need to compute max absolute subarray sum for all APs.
    # An AP is defined by start r (0 <= r < d) and step d.
    # The subsequence is s[r], s[r+d], ...
    # Max absolute subarray sum = max( max_subarray_sum, -min_subarray_sum )
    
    def get_cost_np(s_np):
        max_disc = 0
        for d in range(1, n + 1):
            for r in range(d):
                sub = s_np[r::d]
                if len(sub) == 0:
                    continue
                # Prefix sums
                prefix = np.cumsum(sub)
                
                # Max subarray sum
                # max_{i <= j} (prefix[j] - prefix[i-1]) where prefix[-1]=0
                # This is max(prefix[j] - min(prefix[:j]))
                # We can compute min_prefix_up_to_j efficiently?
                # np.minimum.accumulate(prefix) gives min(prefix[0..j])
                # But we need min(prefix[0..j-1])?
                # Actually, standard Kadane on prefix sums:
                # Let P be prefix sums. We want max_{0 <= i < j <= L} (P[j] - P[i]) where P[0] is sum of first element?
                # No, let P[0] = 0, P[k] = sum_{i=0}^{k-1} sub[i].
                # Then sum_{i=l}^{r} sub[i] = P[r+1] - P[l].
                # We want max |P[r+1] - P[l]| for 0 <= l < r+1 <= L.
                # This is equivalent to max(P) - min(P) if we consider all pairs?
                # No, max_{l < r} (P[r] - P[l]) is max_subarray_sum.
                # And min_{l < r} (P[r] - P[l]) is min_subarray_sum.
                
                # To find max_{l < r} (P[r] - P[l]):
                # Iterate r from 1 to L. Maintain min_P_so_far = min(P[0..r-1]).
                # candidate = P[r] - min_P_so_far.
                # max_sub = max(max_sub, candidate).
                
                # Vectorized approach:
                # min_P_so_far for each r is np.minimum.accumulate(P[:-1])?
                # No, np.minimum.accumulate(P) gives min(P[0..i]) for each i.
                # We need min(P[0..i-1]) to subtract from P[i].
                # Let min_pref = np.minimum.accumulate(P)
                # Then min(P[0..i-1]) is min_pref[i-1] for i>=1.
                # For i=0, there is no l < 0.
                
                P = np.concatenate(([0], prefix))
                # P has length L+1
                # We want max_{0 <= l < r <= L} (P[r] - P[l])
                # min_P_before_r = min(P[0..r-1])
                # For r=1, min is P[0]. For r=2, min is min(P[0], P[1]).
                
                min_before = np.empty_like(P)
                min_before[0] = 0 # dummy
                if len(P) > 1:
                    min_before[1:] = np.minimum.accumulate(P[:-1])
                    
                diffs_max = P - min_before
                max_sub = np.max(diffs_max[1:]) if len(P) > 1 else 0
                
                # Similarly for min subarray sum: min_{l < r} (P[r] - P[l])
                # = min_{r} (P[r] - max(P[0..r-1]))
                max_before = np.empty_like(P)
                max_before[0] = 0
                if len(P) > 1:
                    max_before[1:] = np.maximum.accumulate(P[:-1])
                    
                diffs_min = P - max_before
                min_sub = np.min(diffs_min[1:]) if len(P) > 1 else 0
                
                curr_disc = max(max_sub, -min_sub)
                if curr_disc > max_disc:
                    max_disc = curr_disc
        return max_disc

    best_seq_np = np.array(seq)
    best_cost = get_cost_np(best_seq_np)
    
    # 3. CSP Backtracking with Forward Checking
    # Goal: Find a sequence with max_disc <= B.
    # We start with B = best_cost - 1.
    # If found, update best_cost and B.
    # If not found within time, return best_seq.
    
    # Precompute AP indices for forward checking
    # For each index i, we need to know which APs it belongs to.
    # AP is defined by (d, r). i belongs to (d, r) if i % d == r.
    # In that AP, the position is pos = i // d.
    
    # To check if an AP is violated, we need the current sum of any subarray.
    # During backtracking, we assign s[i] one by one.
    # Forward checking: When we assign s[i], we update the state of all APs containing i.
    # If any AP's *current* max absolute subarray sum exceeds B, we prune.
    # Note: This is a necessary condition. It is not sufficient (future assignments might fix it? No, if current sum > B, it's already violated).
    # Wait, if we are assigning signs, a "current" sum is a partial sum.
    # If we have assigned some elements in an AP, and the max absolute sum of the *assigned* elements (treating unassigned as 0? No, that's wrong).
    # Actually, standard CSP forward checking checks if a value assignment leads to a domain wipeout for a neighbor.
    # Here, the "constraint" is on the final sum.
    # A better heuristic: 
    # If the *maximum possible* sum of the remaining unassigned elements in an AP, added to the current sum, cannot bring it back to <= B, prune?
    # Or simpler: If the current sum of the *assigned* part of an AP already exceeds B, prune?
    # No, because we can assign -1 to later elements to reduce the sum.
    # Example: AP sum so far is 5. B=3. We have 2 unassigned. We can make them -1, -1. Sum becomes 3. OK.
    # So we need to check if it is *possible* to satisfy the bound.
    # Condition for AP (d, r) to be satisfiable:
    # Let S_assigned be the sum of assigned elements in this AP.
    # Let k be the number of unassigned elements in this AP.
    # The final sum S_final will be in [S_assigned - k, S_assigned + k].
    # We need |S_final| <= B.
    # This is possible if and only if [S_assigned - k, S_assigned + k] intersects [-B, B].
    # i.e., S_assigned - k <= B AND S_assigned + k >= -B.
    # i.e., S_assigned <= B + k AND S_assigned >= -B - k.
    
    # This is a very strong and cheap forward check!
    
    # State to maintain for each AP (d, r):
    # - sum_assigned: sum of signs assigned so far in this AP.
    # - count_unassigned: number of unassigned positions in this AP.
    
    # Precomputation:
    # ap_info[(d, r)] = {
    #    'indices': list of indices,
    #    'sum': 0,
    #    'unassigned': len(indices)
    # }
    
    ap_info = {}
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for r in range(d):
            indices = list(range(r, n, d))
            ap_info[(d, r)] = {
                'indices': indices,
                'sum': 0,
                'unassigned': len(indices)
            }
            for pos, idx in enumerate(indices):
                index_to_aps[idx].append((d, r, pos))
                
    # Backtracking function
    # s: numpy array of signs, 0 if unassigned.
    # idx: current index to assign.
    # order: list of indices to assign in this order.
    
    def backtrack(s, idx, order, B):
        if time.time() > deadline:
            return False, None
            
        if idx == len(order):
            # All assigned. Check if valid (should be if forward check passed, but verify cost)
            # Actually, forward check ensures |sum| <= B is *possible*, but does it ensure the *max subarray* sum is <= B?
            # NO. The forward check above only checks the *total sum* of the AP.
            # The task requires max *subarray* sum to be <= B.
            # The total sum constraint is necessary but not sufficient.
            # Example: +1, -1, +1. Total sum 1. Max subarray 1. OK.
            # Example: +1, +1, -1. Total sum 1. Max subarray 2.
            # If B=1, the total sum check passes (1 <= 1 + 0). But max subarray is 2 > 1.
            # So forward checking on total sum is weak.
            
            # To make it stronger, we should check the current max subarray sum of the *assigned* elements?
            # No, unassigned elements can be anything.
            
            # Given the weakness, the backtracking might go deep and fail at the end.
            # Let's add a check at the end of backtracking:
            cost = get_cost_np(s)
            if cost <= B:
                return True, s.copy()
            return False, None
        
        i = order[idx]
        
        # Try +1 and -1
        for val in [1, -1]:
            s[i] = val
            
            # Forward Check: Update APs containing i
            valid = True
            affected_aps = index_to_aps[i]
            
            for d, r, pos in affected_aps:
                info = ap_info[(d, r)]
                info['sum'] += val
                info['unassigned'] -= 1
                
                # Check if this AP can still satisfy |Total Sum| <= B
                # Condition: sum - unassigned <= B AND sum + unassigned >= -B
                if info['sum'] - info['unassigned'] > B or info['sum'] + info['unassigned'] < -B:
                    valid = False
                    break
            
            if valid:
                # Recurse
                res, new_s = backtrack(s, idx + 1, order, B)
                if res:
                    return True, new_s
            
            # Undo
            s[i] = 0
            for d, r, pos in affected_aps:
                info = ap_info[(d, r)]
                info['sum'] -= val
                info['unassigned'] += 1
                
        return False, None

    # Strategy:
    # 1. Start with B = best_cost - 1.
    # 2. Try to find a solution with this B using backtracking.
    # 3. If found, update best_cost, B = best_cost - 1, repeat.
    # 4. If not found, maybe relax B? No, we want to minimize.
    # 5. If time runs out, return best_seq.
    
    # To make backtracking efficient, we should order variables wisely.
    # Random order is okay, but maybe ordering by "constraint tightness" is better.
    # For now, use random order with restarts.
    
    current_B = best_cost - 1
    if current_B < 1:
        current_B = 1 # Minimum possible discrepancy is 1 (singleton)
        
    rng = random.Random(42)
    
    # We will run backtracking for a few seconds.
    # If it finds a solution, we tighten B.
    # If it doesn't, we might try a different random order or give up on that B.
    
    # Since full backtracking on 256 variables is huge, we need to be very careful.
    # The forward check is weak (only total sum), so it might not prune enough.
    # Let's add a heuristic: if the current *max subarray sum* of the assigned elements in an AP is already > B, prune?
    # No, unassigned elements could be negative to reduce it?
    # Actually, if we have a prefix sum that is very high, and we can only add -1s, we might not be able to reduce it enough if the length is short.
    
    # Let's stick to the total sum check for now, but limit the depth or time per backtracking run.
    
    # Better approach:
    # Use the backtracking to *repair* the current best sequence?
    # Or start from scratch?
    # Starting from scratch with B=1 is likely to fail quickly.
    # B=2? B=3?
    
    # Let's try to find a solution with B = best_cost - 1.
    # If best_cost is 5, we try B=4.
    
    # To speed up, we can fix a subset of variables to their best_seq values and backtrack on the rest?
    # This is "partial backtracking".
    
    # Let's try a simple loop:
    # While time < deadline:
    #   Pick a random subset of indices to "freeze" (e.g., 50%).
    #   Initialize s with best_seq values for frozen, 0 for others.
    #   Update AP info for frozen variables.
    #   Run backtracking on unfrozen variables.
    #   If success, update best_seq.
    
    # This is complex. Let's try a simpler "Local Search with CSP pruning" hybrid.
    # Actually, let's just run the backtracking on the full sequence but with a very tight time limit per attempt (e.g., 0.1s).
    # If it fails, restart with a new random order.
    
    # However, 256 variables is too many for full backtracking in Python even with pruning, unless the pruning is very strong.
    # The total sum pruning is weak.
    
    # Let's improve the pruning:
    # For each AP, maintain the current max subarray sum of the *assigned* elements?
    # No, that's not sufficient.
    
    # Alternative:
    # Use the backtracking to optimize a *small* window?
    # No.
    
    # Let's fall back to a robust Local Search (SA) but with a better cost function and maybe a different move strategy (e.g., flip 2 bits).
    # The current program already does SA.
    # The directive asks for CSP.
    
    # Let's implement a "Greedy CSP" approach:
    # 1. Start with Thue-Morse.
    # 2. Find the AP with the worst discrepancy.
    # 3. Identify the element in that AP that, if flipped, reduces the discrepancy the most.
    # 4. Flip it.
    # 5. Repeat.
    # This is essentially steepest descent.
    
    # Let's combine:
    # Phase 1: Greedy Descent (fast).
    # Phase 2: SA to escape local minima.
    
    # I will implement the Greedy Descent as the "CSP-like" solver that tries to satisfy the constraint B = current_max - 1.
    
    def greedy_descent(s, B, max_iters=1000):
        s = s.copy()
        current_cost = get_cost_np(s)
        if current_cost <= B:
            return s, True
            
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Find worst AP
            # We need to know which AP is violating B.
            # Instead of computing full cost, we can just find the max discrepancy AP.
            # But get_cost_np returns the max value, not the AP.
            # Let's modify get_cost_np to also return the worst AP?
            # Or just recompute cost.
            
            # To be fast, let's just try flipping random elements and see if cost decreases.
            # This is random hill climbing.
            
            idx = rng.randrange(n)
            s[idx] *= -1
            new_cost = get_cost_np(s)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost <= B:
                    return s, True
            else:
                s[idx] *= -1 # revert
                
        return s, False

    # Run Greedy Descent to tighten B
    current_B = best_cost - 1
    improved = True
    while improved and time.time() < deadline:
        improved = False
        s, found = greedy_descent(best_seq_np, current_B, max_iters=5000)
        if found:
            best_seq_np = s
            best_cost = get_cost_np(best_seq_np)
            current_B = best_cost - 1
            improved = True
            
    # If we still have time, run SA to try to break B
    if time.time() < deadline:
        T = 5.0
        cooling = 0.999
        min_T = 0.5
        current_seq = best_seq_np.copy()
        current_cost = best_cost
        
        while time.time() < deadline:
            idx = rng.randrange(n)
            current_seq[idx] *= -1
            new_cost = get_cost_np(current_seq)
            delta = new_cost - current_cost
            
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq_np = current_seq.copy()
            else:
                if rng.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    current_seq[idx] *= -1
                    
            T *= cooling
            if T < min_T:
                T = min_T
                
    return best_seq_np.tolist()

# Test
if __name__ == "__main__":
    s = construct()
    print(len(s))
    print(s[:10])
