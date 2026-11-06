import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0  # Safety margin

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse rows as bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Input wires
    input_wires = [1 << j for j in range(n)]
    
    # We need to track which wire is which to output the gate list
    # wire_id -> mask
    # mask -> wire_id
    # Inputs have IDs 0..n-1
    # Gate k has ID n+1+k
    
    # Cost function:
    # Given a set of wire masks W, calculate the minimum total cost.
    # Cost = (len(W) - n) + sum(min_wires_needed_for_row_i)
    
    def calculate_cost(wire_masks):
        """
        wire_masks: list of unique integer masks representing available wires.
        Returns: total gate cost + synthesis cost.
        """
        K = len(wire_masks)
        if K == 0:
            return float('inf')
        
        # If K is too large, 2^K is too slow.
        # For n<=16, optimal K is usually < 20.
        # If K > 24, we approximate or return inf to stop expansion.
        if K > 24:
            return float('inf')
            
        # Map mask to index in wire_masks for quick lookup?
        # We need to find min popcount subset sum for each row.
        
        # Precompute subset XORs
        # We can do this iteratively.
        # subset_xor[s] = XOR of wires in subset s
        # This is 2^K memory. For K=24, 16M entries. Too much memory/time in Python.
        
        # Alternative: DP for each row?
        # For each row r, find min popcount subset of W that sums to r.
        # This is a linear system W x = r.
        # We can use Gaussian elimination to find a basis, but we need MIN WEIGHT.
        # Min weight solution to Ax=b is NP-hard, but K is small.
        
        # Brute force over 2^K is too slow for K>20 in Python.
        # We need a faster cost estimator or a limit on K.
        
        # Let's limit K to 20. 2^20 = 1,048,576.
        # Looping 1M times in Python takes ~0.1s.
        # If we have ~100 candidates, 100 * 0.1s = 10s per step.
        # Too slow if many steps.
        
        # Optimization: Only calculate cost for promising candidates.
        # Or, use a heuristic cost: 
        # Heuristic: Sum of (min distance in Hamming space to any wire in W)? No.
        
        # Let's use a "Lazy" exact cost calculation.
        # We only calculate exact cost if K <= 20.
        # If K > 20, we use a greedy approximation for the remaining rows.
        
        if K > 20:
            # Approximate cost
            # For each row, find the wire in W with max dot product?
            # Or just return a large number to discourage growth.
            # Actually, if K > 20, we are likely over-synthesizing.
            # Let's return a penalty.
            return 1000 + (K - n)
            
        # Exact calculation for K <= 20
        # We want min popcount for each row.
        # We can iterate over all subsets s of W.
        # xor_val = 0
        # For s in range(2^K):
        #    compute xor of wires in s
        #    check if xor_val == row_i
        #    if yes, update min_cost[i] = min(min_cost[i], popcount(s))
        
        # To speed up, we can use DP for subset XORs.
        # subset_xor[0] = 0
        # subset_xor[s] = subset_xor[s ^ (1<<i)] ^ wire[i] where i is lowest set bit.
        
        subset_xor = [0] * (1 << K)
        popcounts = [0] * (1 << K)
        
        # This is O(2^K * K) if done naively, but we can do O(2^K).
        for s in range(1, 1 << K):
            # Find lowest set bit
            lsb = s & -s
            i = lsb.bit_length() - 1
            prev = s ^ lsb
            subset_xor[s] = subset_xor[prev] ^ wire_masks[i]
            popcounts[s] = popcounts[prev] + 1
            
        min_costs = [float('inf')] * m
        
        for s in range(1 << K):
            x = subset_xor[s]
            pc = popcounts[s]
            # Check if x matches any row
            # Optimization: Map row_masks to indices
            # If x is in row_to_idx, update min_cost
            if x in row_to_idx:
                idx = row_to_idx[x]
                if pc < min_costs[idx]:
                    min_costs[idx] = pc
            # Also check if x is 0? No, rows are non-zero usually, but could be.
            # If a row is 0, min cost is 0 (empty set).
            
        # Handle zero rows
        for i in range(m):
            if row_masks[i] == 0:
                min_costs[i] = 0
                
        # If any row is not reachable, return inf
        for i in range(m):
            if min_costs[i] == float('inf'):
                return float('inf')
                
        total_cost = (K - n) + sum(min_costs)
        return total_cost

    # Map row masks to indices for quick lookup
    row_to_idx = {}
    for i, r in enumerate(row_masks):
        row_to_idx[r] = i

    best_gates = []
    best_outputs = []
    best_score = float('inf')
    
    # Multi-start greedy
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        current_wires = list(input_wires)
        # Add some random noise to tie-breaking?
        # Or just run deterministic greedy.
        
        gates = []
        
        while time.time() < DEADLINE - 10.0:
            current_cost = calculate_cost(current_wires)
            if current_cost == float('inf'):
                break
                
            best_candidate = None
            best_new_cost = float('inf')
            
            # Generate candidates
            K = len(current_wires)
            candidates = set()
            for i in range(K):
                for j in range(i + 1, K):
                    w = current_wires[i] ^ current_wires[j]
                    if w not in current_wires:
                        candidates.add(w)
                        
            # If no candidates, break
            if not candidates:
                break
                
            # Evaluate candidates
            # To save time, we can limit the number of candidates evaluated if K is large
            # But K is small.
            for w in candidates:
                new_wires = current_wires + [w]
                new_cost = calculate_cost(new_wires)
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_candidate = w
                    
            if best_candidate is None or best_new_cost >= current_cost:
                break
                
            # Accept candidate
            current_wires.append(best_candidate)
            # Find which two wires produced it?
            # We need to record the gate.
            # We need to find u, v in current_wires[:-1] such that u^v = best_candidate.
            found = False
            for i in range(len(current_wires) - 1):
                for j in range(i + 1, len(current_wires) - 1):
                    if current_wires[i] ^ current_wires[j] == best_candidate:
                        # Map masks to wire IDs
                        # We need a mask_to_id map
                        # This is getting complex.
                        pass
            
            # This approach of tracking IDs is messy.
            # Let's switch to a simpler heuristic that is faster.
            break # Skip this complex exact method for now, use the faster greedy from the prompt but improved.

    # Let's use the Improved Greedy from the prompt but with a better gain function.
    # Gain = reduction in total "wire count" across all rows.
    
    wire_masks = {}
    mask_to_wire = {}
    
    for j in range(n):
        mask = 1 << j
        wire_id = j
        wire_masks[wire_id] = mask
        mask_to_wire[mask] = wire_id
        
    zero_wire = n
    wire_masks[zero_wire] = 0
    mask_to_wire[0] = zero_wire
    
    next_wire_id = n + 1
    
    row_wires = [set() for _ in range(m)]
    for i in range(m):
        mask = row_masks[i]
        if mask == 0:
            row_wires[i].add(zero_wire)
        else:
            for j in range(n):
                if (mask >> j) & 1:
                    row_wires[i].add(j)
                    
    gates = []
    
    def elapsed():
        return time.time() - start_time

    # Greedy Loop with Lookahead
    while elapsed() < DEADLINE - 5.0:
        best_gain = 0
        best_u = -1
        best_v = -1
        best_mask = -1
        
        wire_ids = list(wire_masks.keys())
        k = len(wire_ids)
        
        # Filter out zero wire from consideration as input to XOR?
        # XOR with 0 is identity, so it doesn't create new wires.
        
        for i in range(k):
            u = wire_ids[i]
            if u == zero_wire: continue
            for j in range(i + 1, k):
                v = wire_ids[j]
                if v == zero_wire: continue
                
                mask_u = wire_masks[u]
                mask_v = wire_masks[v]
                new_mask = mask_u ^ mask_v
                
                if new_mask in mask_to_wire:
                    continue
                
                # Calculate Gain
                # Gain = number of rows that contain BOTH u and v
                # This reduces the wire count for those rows by 1 (2 wires -> 1 wire).
                count = 0
                for r in range(m):
                    if u in row_wires[r] and v in row_wires[r]:
                        count += 1
                        
                gain = count
                if gain > best_gain:
                    best_gain = gain
                    best_u = u
                    best_v = v
                    best_mask = new_mask
                    
        if best_gain <= 1: # Need at least 1 row to benefit? If gain=1, we replace 2 wires with 1, saving 1.
            # Actually, if gain=1, we save 1 gate in the synthesis of that row.
            # But we ADD 1 gate to create the new wire.
            # Net change: 0.
            # So we need gain >= 2 to have a net gain?
            # Or gain=1 is neutral, but might enable future gains.
            # Let's allow gain >= 1 but prioritize higher.
            if best_gain < 1:
                break
            # If gain == 1, it's a tie. We might skip to avoid bloat.
            # Let's require gain >= 2 for strict improvement, or gain >= 1 if we are stuck.
            # For now, accept gain >= 1.
            pass
        
        # Create new wire
        new_wire_id = next_wire_id
        next_wire_id += 1
        gates.append([best_u, best_v])
        wire_masks[new_wire_id] = best_mask
        mask_to_wire[best_mask] = new_wire_id
        
        # Update rows
        for r in range(m):
            if best_u in row_wires[r] and best_v in row_wires[r]:
                row_wires[r].discard(best_u)
                row_wires[r].discard(best_v)
                row_wires[r].add(new_wire_id)

    # Final Output
    outputs = [0] * m
    for i in range(m):
        if not row_wires[i]:
            outputs[i] = zero_wire
        elif len(row_wires[i]) == 1:
            outputs[i] = list(row_wires[i])[0]
        else:
            wires = list(row_wires[i])
            current_wires = wires
            while len(current_wires) > 1:
                new_current_wires = []
                for k in range(0, len(current_wires), 2):
                    if k + 1 < len(current_wires):
                        w1 = current_wires[k]
                        w2 = current_wires[k + 1]
                        m1 = wire_masks[w1]
                        m2 = wire_masks[w2]
                        nm = m1 ^ m2
                        if nm in mask_to_wire:
                            new_current_wires.append(mask_to_wire[nm])
                        else:
                            new_w = next_wire_id
                            next_wire_id += 1
                            gates.append([w1, w2])
                            wire_masks[new_w] = nm
                            mask_to_wire[nm] = new_w
                            new_current_wires.append(new_w)
                    else:
                        new_current_wires.append(current_wires[k])
                current_wires = new_current_wires
            outputs[i] = current_wires[0]

    return {'gates': gates, 'outputs': outputs}
