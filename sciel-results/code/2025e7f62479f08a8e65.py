import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 55  # 5s margin
    
    # Precompute APs ending at each index
    # ap_end[i] = list of (start, step, length) for APs ending at i
    # Actually, we need to update sums when we flip a sign.
    # Let's precompute for each index i, the list of APs that *contain* i.
    # But for greedy construction, we only care about APs that *end* at i?
    # No, flipping x_i affects all APs containing i.
    # However, if we build left-to-right, when we choose x_i, all APs containing i 
    # and ending at j >= i will be affected. But we haven't chosen x_j for j > i yet.
    # So we can only evaluate APs that are *fully determined* so far.
    # An AP (start, step, length) is fully determined when we have chosen 
    # x_start, ..., x_{start + (length-1)*step}.
    # The "last" element is start + (length-1)*step.
    # So when we choose x_i, we can evaluate all APs where i is the last element.
    # This is a valid greedy heuristic: minimize the max discrepancy of "completed" APs.
    
    # Precompute: for each i, list of (start, step) such that i = start + (len-1)*step
    # i.e., start = i - (len-1)*step >= 0.
    # Let k = len-1. i = start + k*step.
    # So for a fixed i, we iterate over step s from 1 to i.
    # Then start = i - k*s. We need start >= 0.
    # k can be 0, 1, ..., i//s.
    # So for each s, for each k in 0..i//s, start = i - k*s.
    # The AP is (start, s, k+1).
    
    # We need to maintain the sum of each AP.
    # Total number of APs is O(n^2 log n).
    # For n=256, this is manageable.
    
    # Let's use a dictionary to store current sums of APs.
    # Key: (start, step, length) or just (start, step) since length is determined by end?
    # No, different lengths can have same start, step.
    # But in our greedy step, we only complete APs ending at i.
    # So we can just keep a list of "active" APs?
    # Actually, once an AP is completed (its last element is chosen), its sum is fixed 
    # (until we flip a sign in it, which we won't do in greedy phase).
    # Wait, we *do* flip signs in SA phase.
    # So we need a data structure that allows fast updates for any flip.
    
    # Let's go back to the idea of maintaining sums for ALL APs.
    # Precompute for each index i, the list of APs containing i.
    # ap_contains[i] = list of (start, step, length)
    
    # To speed up, we can group APs by (start, step).
    # For a given (start, step), the APs are prefixes of the sequence 
    # start, start+step, start+2*step, ...
    # Let's call the "base AP" the infinite sequence defined by start, step.
    # The sum of the prefix of length L is S(start, step, L).
    # When we flip x_i, it affects all base APs (start, step) such that 
    # i = start + k*step for some k >= 0.
    # And for each such base AP, it affects all prefixes of length > k.
    
    # This is getting complicated. Let's stick to the simpler O(n^2) per flip 
    # but optimize the inner loop.
    
    # Alternative: Use the "freq-array" insight from inspiration 3.
    # "Correct AP definition (all starts) and efficient freq-array based max tracking"
    
    # Let's try a different structural change:
    # **Greedy Construction with Lookahead**
    # Build sequence left-to-right.
    # At step i, try both +1 and -1.
    # For each choice, calculate the new max discrepancy among all APs 
    # that end at i (i.e., are completed by this choice).
    # Choose the sign that minimizes this new max.
    # This is O(n^3) total if we recompute sums from scratch, 
    # but we can maintain sums incrementally.
    
    # Let's maintain a 2D array sum_ap[start][step] = current sum of the AP 
    # starting at 'start' with 'step', up to the current index i.
    # When we add x_i, we update sum_ap[start][step] for all 
    # (start, step) such that i is in the AP.
    # i = start + k*step => start = i - k*step.
    # For a fixed i, the pairs (start, step) are:
    # step = 1: start = i, i-1, ..., 0 (k=0..i) -> i+1 pairs
    # step = 2: start = i, i-2, ..., 0 or 1 (k=0..i//2) -> i//2 + 1 pairs
    # ...
    # Total pairs for index i is sum_{s=1}^{i} (i//s + 1) ~ i log i.
    # Total updates over all i is sum_{i=0}^{n-1} i log i ~ n^2 log n.
    # For n=256, 256^2 * 8 ~ 500k updates. Very fast.
    
    # But wait, this only tracks the sum of the *prefix* of the base AP.
    # The discrepancy is the max over ALL prefixes of ALL base APs.
    # So we need to track the max prefix sum for each (start, step).
    
    # Let's define:
    # max_prefix_sum[start][step] = max(0, x_start, x_start+x_{start+step}, ...)
    # min_prefix_sum[start][step] = min(0, x_start, x_start+x_{start+step}, ...)
    # The contribution of this base AP to the global max discrepancy is 
    # max(max_prefix_sum, -min_prefix_sum).
    
    # When we add x_i, we update the current sum for each (start, step) 
    # where i is in the AP.
    # current_sum[start][step] += x_i
    # max_prefix_sum[start][step] = max(max_prefix_sum, current_sum)
    # min_prefix_sum[start][step] = min(min_prefix_sum, current_sum)
    
    # This works! And it's efficient.
    
    # Initialize
    # Use 2D arrays indexed by start and step.
    # start in 0..n-1, step in 1..n-1.
    # current_sum = np.zeros((n, n), dtype=int)
    # max_pref = np.zeros((n, n), dtype=int)
    # min_pref = np.zeros((n, n), dtype=int)
    
    # But we only need to consider (start, step) where start + step < n? 
    # No, even if start + step >= n, the AP of length 1 is valid.
    # But for step > start, the AP length is 1.
    # Actually, the condition is just that the AP is contained in [0, n-1].
    
    # Let's precompute the list of (start, step) pairs for each i.
    # pairs_for_i[i] = list of (start, step) such that i = start + k*step for some k>=0
    # and start >= 0.
    
    pairs_for_i = [[] for _ in range(n)]
    for i in range(n):
        for step in range(1, n):
            # i = start + k*step => start = i - k*step
            # k can be 0, 1, ..., i//step
            # start = i, i-step, i-2*step, ...
            # We need start >= 0.
            # So k can be 0 to i//step.
            # For each k, start = i - k*step.
            # But we also need the AP to be valid, i.e., all elements in [0, n-1].
            # Since we are building left-to-right, and i is the current index,
            # all elements start, start+step, ..., i are <= i < n.
            # And start >= 0.
            # So the AP is valid.
            for k in range(i // step + 1):
                start = i - k * step
                pairs_for_i[i].append((start, step))
    
    # Now, greedy construction
    seq = [0] * n
    current_sum = np.zeros((n, n), dtype=int)
    max_pref = np.zeros((n, n), dtype=int)
    min_pref = np.zeros((n, n), dtype=int)
    
    # Global max discrepancy
    global_max = 0
    
    for i in range(n):
        # Try +1
        # We need to calculate the new global max if we choose +1
        # and if we choose -1.
        
        # To avoid modifying the arrays, we can calculate the change.
        # For each (start, step) in pairs_for_i[i]:
        #   new_sum = current_sum[start][step] + sign
        #   new_max_pref = max(max_pref[start][step], new_sum)
        #   new_min_pref = min(min_pref[start][step], new_sum)
        #   new_contrib = max(new_max_pref, -new_min_pref)
        #   If new_contrib > global_max, then new_global_max = max(new_global_max, new_contrib)
        
        # Let's do this for both signs and pick the better one.
        
        best_sign = 1
        best_new_max = 0
        
        # Calculate for +1
        new_max_plus = global_max
        for start, step in pairs_for_i[i]:
            new_sum = current_sum[start][step] + 1
            new_mp = max_pref[start][step]
            if new_sum > new_mp:
                new_mp = new_sum
            new_mn = min_pref[start][step]
            if new_sum < new_mn:
                new_mn = new_sum
            contrib = max(new_mp, -new_mn)
            if contrib > new_max_plus:
                new_max_plus = contrib
        
        # Calculate for -1
        new_max_minus = global_max
        for start, step in pairs_for_i[i]:
            new_sum = current_sum[start][step] - 1
            new_mp = max_pref[start][step]
            if new_sum > new_mp:
                new_mp = new_sum
            new_mn = min_pref[start][step]
            if new_sum < new_mn:
                new_mn = new_sum
            contrib = max(new_mp, -new_mn)
            if contrib > new_max_minus:
                new_max_minus = contrib
        
        # Choose the sign with smaller new max
        if new_max_plus <= new_max_minus:
            seq[i] = 1
            global_max = new_max_plus
            sign = 1
        else:
            seq[i] = -1
            global_max = new_max_minus
            sign = -1
            
        # Update the arrays
        for start, step in pairs_for_i[i]:
            current_sum[start][step] += sign
            if current_sum[start][step] > max_pref[start][step]:
                max_pref[start][step] = current_sum[start][step]
            if current_sum[start][step] < min_pref[start][step]:
                min_pref[start][step] = current_sum[start][step]
    
    # Now we have a good initial sequence.
    # Let's polish it with Simulated Annealing.
    # We need a fast cost function for SA.
    # We can reuse the same logic but in a batch.
    
    # Let's define a function to calculate the cost of a given sequence.
    def get_cost(s):
        current_sum = np.zeros((n, n), dtype=int)
        max_pref = np.zeros((n, n), dtype=int)
        min_pref = np.zeros((n, n), dtype=int)
        global_max = 0
        for i in range(n):
            sign = s[i]
            for start, step in pairs_for_i[i]:
                current_sum[start][step] += sign
                if current_sum[start][step] > max_pref[start][step]:
                    max_pref[start][step] = current_sum[start][step]
                if current_sum[start][step] < min_pref[start][step]:
                    min_pref[start][step] = current_sum[start][step]
                contrib = max(max_pref[start][step], -min_pref[start][step])
                if contrib > global_max:
                    global_max = contrib
        return global_max
    
    best_seq = seq[:]
    best_cost = global_max
    
    # SA
    rng = random.Random(42)
    temp = 10.0
    cooling = 0.99
    
    while time.time() < deadline:
        idx = rng.randrange(n)
        seq[idx] *= -1
        new_cost = get_cost(seq)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_seq = seq[:]
        else:
            # Accept with probability exp(-(new_cost - best_cost) / temp)
            if new_cost > best_cost:
                delta = new_cost - best_cost
                if rng.random() < np.exp(-delta / temp):
                    pass # Accept
                else:
                    seq[idx] *= -1 # Reject
            # If new_cost == best_cost, accept (to allow moves)
            # Actually, if new_cost < best_cost, we already accepted.
            # If new_cost == best_cost, we should accept to avoid getting stuck.
            # Let's modify:
            # if new_cost <= best_cost: accept
            # else: accept with prob
        
        temp *= cooling
        if temp < 0.01:
            break
            
    return best_seq
