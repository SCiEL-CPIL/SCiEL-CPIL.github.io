import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Parse input
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
    wires_val = [1 << i for i in range(n)]
    wires_val.append(0)  # Wire n is 0
    
    # Track depth for each wire (input wires have depth 0)
    wire_depth = [0] * (n + 1)
    
    wire_map = {val: idx for idx, val in enumerate(wires_val)}
    
    # Initialize outputs as sets of wire indices
    out_sets = []
    for i in range(m):
        mask = 0
        row = mat[i]
        for j in range(n):
            if row[j] == 1:
                mask |= (1 << j)
        s = set()
        for j in range(n):
            if mask & (1 << j):
                s.add(j)
        out_sets.append(s)
        
    gates = []
    next_wire_idx = n + 1  # First gate output wire index
    
    # Greedy CSE with depth awareness
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
        best_new_depth = 0
        
        num_wires = len(wires_val)
        for i in range(num_wires):
            vi = wires_val[i]
            di = wire_depth[i]
            for j in range(i + 1, num_wires):
                vj = wires_val[j]
                dj = wire_depth[j]
                w_val = vi ^ vj
                
                # Calculate new depth if we create this gate
                new_depth = max(di, dj) + 1
                
                # If w_val already exists, no new gate needed
                if w_val in wire_map:
                    k = wire_map[w_val]
                    # Using existing wire doesn't add depth
                    effective_new_depth = wire_depth[k]
                else:
                    effective_new_depth = new_depth
                
                # Check if this would exceed depth limit
                if effective_new_depth > depth_limit:
                    continue
                
                # Calculate benefit: how many outputs contain both i and j?
                benefit = 0
                for s in out_sets:
                    if i in s and j in s:
                        benefit += 1
                
                # Bonus for creating wires that might be reused
                # Count how many times w_val appears in output masks
                reuse_potential = 0
                for s in out_sets:
                    # Check if XOR of all wires in s except i,j equals w_val
                    # This is expensive, so skip for now
                    pass
                
                # Prefer pairs that result in lower depth
                # Use a composite score: benefit * 10 - effective_new_depth
                score = benefit * 10 - effective_new_depth
                
                if score > best_benefit or (score == best_benefit and effective_new_depth < best_new_depth):
                    best_benefit = score
                    best_i = i
                    best_j = j
                    best_new_depth = effective_new_depth
                    
        if best_i != -1:
            i, j = best_i, best_j
            w_val = wires_val[i] ^ wires_val[j]
            
            # Check if w_val already exists
            if w_val in wire_map:
                k = wire_map[w_val]
            else:
                k = next_wire_idx
                wires_val.append(w_val)
                wire_map[w_val] = k
                wire_depth.append(max(wire_depth[i], wire_depth[j]) + 1)
                gates.append([i, j])
                next_wire_idx += 1
                
            # Update outputs: replace i, j with k in all sets containing both
            for s in out_sets:
                if i in s and j in s:
                    s.remove(i)
                    s.remove(j)
                    s.add(k)
        else:
            # No pair has positive benefit or all would exceed depth limit
            # Find output with max size
            max_idx = 0
            max_size = 0
            for idx, s in enumerate(out_sets):
                if len(s) > max_size:
                    max_size = len(s)
                    max_idx = idx
            
            if max_size <= 1:
                break
                
            s = out_sets[max_idx]
            items = list(s)
            
            # Try to find a pair that stays within depth limit
            found = False
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    i, j = items[a], items[b]
                    w_val = wires_val[i] ^ wires_val[j]
                    new_depth = max(wire_depth[i], wire_depth[j]) + 1
                    if w_val in wire_map:
                        k = wire_map[w_val]
                        if wire_depth[k] <= depth_limit:
                            found = True
                            break
                    else:
                        if new_depth <= depth_limit:
                            k = -1
                            found = True
                            break
                if found:
                    break
            
            if not found:
                # All pairs exceed depth limit - this shouldn't happen if depth_limit >= 1
                # Fallback: just pick any two and hope for the best
                i, j = items[0], items[1]
                w_val = wires_val[i] ^ wires_val[j]
                if w_val in wire_map:
                    k = wire_map[w_val]
                else:
                    k = next_wire_idx
                    wires_val.append(w_val)
                    wire_map[w_val] = k
                    wire_depth.append(max(wire_depth[i], wire_depth[j]) + 1)
                    gates.append([i, j])
                    next_wire_idx += 1
                
                s.remove(i)
                s.remove(j)
                s.add(k)
            else:
                i, j = items[a], items[b]
                w_val = wires_val[i] ^ wires_val[j]
                if w_val in wire_map:
                    k = wire_map[w_val]
                else:
                    k = next_wire_idx
                    wires_val.append(w_val)
                    wire_map[w_val] = k
                    wire_depth.append(max(wire_depth[i], wire_depth[j]) + 1)
                    gates.append([i, j])
                    next_wire_idx += 1
                
                s.remove(i)
                s.remove(j)
                s.add(k)

    # Finalize: Ensure all outputs are single wires
    for s in out_sets:
        while len(s) > 1:
            if time.time() > deadline - 5:  # Hard stop
                break
            items = list(s)
            
            # Find pair that minimizes depth increase
            best_pair = None
            best_depth = float('inf')
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    i, j = items[a], items[b]
                    w_val = wires_val[i] ^ wires_val[j]
                    if w_val in wire_map:
                        k = wire_map[w_val]
                        d = wire_depth[k]
                    else:
                        d = max(wire_depth[i], wire_depth[j]) + 1
                    if d < best_depth:
                        best_depth = d
                        best_pair = (i, j, w_val)
            
            if best_pair is None:
                break
                
            i, j, w_val = best_pair
            if w_val in wire_map:
                k = wire_map[w_val]
            else:
                k = next_wire_idx
                wires_val.append(w_val)
                wire_map[w_val] = k
                wire_depth.append(max(wire_depth[i], wire_depth[j]) + 1)
                gates.append([i, j])
                next_wire_idx += 1
                
            s.remove(i)
            s.remove(j)
            s.add(k)
            
            if len(s) >= len(items):  # No reduction
                break

    # Extract output wire indices
    outputs = []
    for s in out_sets:
        if len(s) == 1:
            outputs.append(list(s)[0])
        else:
            # Fallback: XOR all remaining
            val = 0
            for idx in s:
                val ^= wires_val[idx]
            if val in wire_map:
                outputs.append(wire_map[val])
            else:
                k = next_wire_idx
                wires_val.append(val)
                wire_map[val] = k
                wire_depth.append(0)  # Placeholder, shouldn't happen
                outputs.append(k)
                
    return {'gates': gates, 'outputs': outputs}
