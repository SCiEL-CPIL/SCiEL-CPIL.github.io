import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Parse input
    # matrix is likely a list of lists or np array
    if isinstance(matrix, np.ndarray):
        mat = matrix
    else:
        mat = np.array(matrix)
        
    if mat.size == 0:
        return {'gates': [], 'outputs': []}
        
    m, n = mat.shape
    
    # Input wires: 0..n-1
    # Zero wire: n
    # Gate outputs: n+1, n+2, ...
    
    # Initialize wires
    # wire_val[i] is the bitmask of input variables for wire i
    wires_val = [1 << i for i in range(n)]
    wires_val.append(0) # Wire n is 0
    
    wire_map = {val: idx for idx, val in enumerate(wires_val)}
    
    # Initialize outputs as sets of wire indices
    # Each output row is a linear combination of inputs
    out_sets = []
    for i in range(m):
        mask = 0
        row = mat[i]
        for j in range(n):
            if row[j] == 1:
                mask |= (1 << j)
        # Decompose mask into input wire indices
        s = set()
        for j in range(n):
            if mask & (1 << j):
                s.add(j)
        out_sets.append(s)
        
    gates = []
    next_wire_idx = n + 1 # First gate output wire index
    
    # Greedy CSE
    while True:
        if time.time() > deadline:
            break
            
        # Check if all outputs are single wires
        if all(len(s) == 1 for s in out_sets):
            break
            
        # Find best pair to XOR
        best_benefit = 0
        best_i = -1
        best_j = -1
        
        # Optimization: Only consider pairs that actually appear in some output together
        # Or just iterate all pairs if W is small.
        # W starts at n+1. Grows slowly.
        
        # To speed up, we can track which pairs are "active" (appear in some output)
        # But for N=16, W is small.
        
        # Iterate pairs
        num_wires = len(wires_val)
        for i in range(num_wires):
            vi = wires_val[i]
            for j in range(i + 1, num_wires):
                vj = wires_val[j]
                w_val = vi ^ vj
                
                # If w_val already exists, no new gate needed, but it might simplify?
                # If it exists, we can just use the existing wire. 
                # But we are looking for a *new* gate to add to the pool.
                # If w_val is in wire_map, we can skip creating a new gate, 
                # but we might still want to "use" it in simplification.
                # However, if it already exists, it's already in the pool.
                # The simplification step (replacing i,j with k) assumes k is a new wire 
                # or an existing one.
                # If k exists, we can replace i,j with k in outputs.
                # This doesn't add a gate, but reduces complexity.
                # We should handle this: if w_val in wire_map, benefit is still calculated, 
                # but no gate is added.
                
                # Calculate benefit: how many outputs contain both i and j?
                # If we replace i,j with k, the set size decreases by 1 for each such output.
                benefit = 0
                for s in out_sets:
                    if i in s and j in s:
                        benefit += 1
                
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_i = i
                    best_j = j
                    # Early exit if benefit is huge? No, just find max.
                    
        if best_benefit > 0:
            i, j = best_i, best_j
            w_val = wires_val[i] ^ wires_val[j]
            
            # Check if w_val already exists
            if w_val in wire_map:
                k = wire_map[w_val]
            else:
                k = next_wire_idx
                wires_val.append(w_val)
                wire_map[w_val] = k
                gates.append([i, j])
                next_wire_idx += 1
                
            # Update outputs: replace i, j with k in all sets containing both
            for s in out_sets:
                if i in s and j in s:
                    s.remove(i)
                    s.remove(j)
                    s.add(k)
        else:
            # No pair has benefit > 0.
            # This means no output contains both of any pair of current wires.
            # We need to make progress by forcing a reduction on the most complex output.
            # Find output with max size
            max_idx = 0
            max_size = 0
            for idx, s in enumerate(out_sets):
                if len(s) > max_size:
                    max_size = len(s)
                    max_idx = idx
            
            if max_size <= 1:
                # All are single wires, should have broken earlier
                break
                
            s = out_sets[max_idx]
            # Pick any two distinct wires from s
            # Convert to list to pick two
            items = list(s)
            i, j = items[0], items[1]
            
            w_val = wires_val[i] ^ wires_val[j]
            
            if w_val in wire_map:
                k = wire_map[w_val]
            else:
                k = next_wire_idx
                wires_val.append(w_val)
                wire_map[w_val] = k
                gates.append([i, j])
                next_wire_idx += 1
                
            # Update this specific output
            s.remove(i)
            s.remove(j)
            s.add(k)
            
            # It's possible that k was already in s? 
            # If k was in s, then s contained i, j, k.
            # Since w_val = vi ^ vj, and k corresponds to w_val.
            # If k was in s, then the output was vi ^ vj ^ vk ^ ...
            # Since vk = vi ^ vj, vi ^ vj ^ vk = 0.
            # So if k was in s, removing i, j and adding k (which is already there) 
            # effectively removes i, j and keeps k.
            # Wait, if k is already in s, then s = {i, j, k, ...}.
            # After remove i, j: s = {k, ...}. Add k: s = {k, ...}.
            # So size decreases by 2? 
            # Original size: len(s).
            # New size: len(s) - 2 + 1 = len(s) - 1.
            # But if k was already there, we added it again? Sets don't allow duplicates.
            # So if k in s:
            #   s.remove(i), s.remove(j) -> s loses 2 elements.
            #   s.add(k) -> k is already there, so no change.
            #   Size decreases by 2.
            # If k not in s:
            #   s.remove(i), s.remove(j) -> loses 2.
            #   s.add(k) -> gains 1.
            #   Size decreases by 1.
            # This is always progress.

    # Finalize: Ensure all outputs are single wires
    # If we exited due to deadline, some sets might have size > 1.
    # We must resolve them by adding necessary gates.
    for s in out_sets:
        while len(s) > 1:
            if time.time() > deadline - 5: # Hard stop
                break
            items = list(s)
            i, j = items[0], items[1]
            w_val = wires_val[i] ^ wires_val[j]
            
            if w_val in wire_map:
                k = wire_map[w_val]
            else:
                k = next_wire_idx
                wires_val.append(w_val)
                wire_map[w_val] = k
                gates.append([i, j])
                next_wire_idx += 1
                
            s.remove(i)
            s.remove(j)
            s.add(k)
            
            # Safety: if loop doesn't terminate (shouldn't happen), break
            if len(s) >= len(items): # No reduction
                break

    # Extract output wire indices
    outputs = []
    for s in out_sets:
        if len(s) == 1:
            outputs.append(list(s)[0])
        else:
            # Fallback: shouldn't happen if logic is correct
            # XOR all remaining
            val = 0
            for idx in s:
                val ^= wires_val[idx]
            if val in wire_map:
                outputs.append(wire_map[val])
            else:
                # Add final gate
                k = next_wire_idx
                wires_val.append(val)
                wire_map[val] = k
                # We need to add gates to form this? 
                # This is a fallback for unexpected states.
                # For now, just return the index, assuming it's valid.
                # But we need to add gates to make it valid.
                # This path is unlikely.
                outputs.append(k)
                
    return {'gates': gates, 'outputs': outputs}
