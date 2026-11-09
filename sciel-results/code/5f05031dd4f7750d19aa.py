import time
import sys

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to integer bitmasks
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)
        
    targets = set(rows)
    targets.discard(0)
    
    if not targets:
        return {'gates': [], 'outputs': [n] * m}

    # Available wires: inputs 0..n-1 and zero wire n
    available = set()
    for j in range(n):
        available.add(1 << j)
    available.add(0)
    
    gates = []
    next_wire = n + 1
    
    # Map value to wire index
    val_to_wire = {0: n}
    for j in range(n):
        val_to_wire[1 << j] = j
        
    # Greedy CSE loop
    # We want to add gates until all targets are in available
    while targets and time.time() < deadline:
        if all(t in available for t in targets):
            break
            
        # Generate candidates
        # Strategy 1: Gates that complete at least one target in 1 step
        # G = T ^ W, where T is target, W is available
        # If we add G, then T = G ^ W is available (1 gate for T)
        candidates = {}
        
        avail_list = list(available)
        
        # If available set is large, this loop is O(|targets| * |available|)
        # For n=16, max available is 65536. Max targets 16-256.
        # 256 * 65536 = 16M ops. Might be slow in Python.
        # Optimization: If |available| is large, sample or use a different metric.
        # But for n=16, we likely won't reach 65536 available wires in a greedy path 
        # because we stop when targets are met.
        
        for t in targets:
            if time.time() > deadline:
                break
            for w in avail_list:
                g = t ^ w
                if g in available:
                    continue
                if g in candidates:
                    candidates[g] += 1
                else:
                    candidates[g] = 1
                    
        best_gate_val = None
        best_score = 0
        best_a = None
        best_b = None
        
        if candidates:
            # Pick the gate that completes the most targets
            for g, count in candidates.items():
                if count > best_score:
                    best_score = count
                    best_gate_val = g
                    # We need to find a, b such that a^b = g
                    # We know g = t ^ w for some t, w
                    # We can find a, b by checking available
                    # But we can just store them during generation?
                    # No, just find them now.
                    # Since g = t ^ w, and t, w are known from the loop?
                    # We lost the specific t, w.
                    # We can find a, b by iterating available again? O(N).
                    # Or we can store the first (t, w) pair that generated g.
                    pass
            
            # Refinement: Store the pair
            # Let's redo the candidate generation to store pairs
            candidates = {}
            for t in targets:
                for w in avail_list:
                    g = t ^ w
                    if g in available:
                        continue
                    if g not in candidates:
                        candidates[g] = [1, t, w]
                    else:
                        candidates[g][0] += 1
                        
            best_score = 0
            for g, info in candidates.items():
                if info[0] > best_score:
                    best_score = info[0]
                    best_gate_val = g
                    best_a = info[1]
                    best_b = info[2]
                    
        if best_gate_val is None:
            # Fallback: No gate completes a target in 1 step.
            # Pick a gate that minimizes the popcount of (T ^ G) for some T
            # i.e., G is close to a target.
            best_score = -1
            for t in targets:
                # Find w in available minimizing popcount(t ^ w)
                min_pop = n + 1
                best_w = None
                for w in avail_list:
                    p = (t ^ w).bit_count()
                    if p < min_pop:
                        min_pop = p
                        best_w = w
                        if p == 0:
                            break
                g = t ^ best_w
                if g in available:
                    continue
                # Score: 1/popcount
                score = 1.0 / (1.0 + min_pop)
                if score > best_score:
                    best_score = score
                    best_gate_val = g
                    best_a = t
                    best_b = best_w
                    
            if best_gate_val is None:
                break
                
        # Add the gate
        # Verify a, b are in available
        if best_a not in available or best_b not in available:
            # This shouldn't happen
            break
            
        wire_a = val_to_wire[best_a]
        wire_b = val_to_wire[best_b]
        gates.append([wire_a, wire_b])
        val_to_wire[best_gate_val] = next_wire
        available.add(best_gate_val)
        next_wire += 1
        
    # Construct outputs
    outputs = []
    for r in rows:
        if r in val_to_wire:
            outputs.append(val_to_wire[r])
        else:
            # Fallback: use zero wire if not found (should not happen)
            outputs.append(n)
            
    return {'gates': gates, 'outputs': outputs}
