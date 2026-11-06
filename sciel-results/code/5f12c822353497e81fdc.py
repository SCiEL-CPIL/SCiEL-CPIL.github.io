import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 50.0
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert rows to bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)
    
    next_wire = n + 1
    gates = []
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n
    
    def elapsed():
        return time.time() - start_time
    
    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w
    
    def build_tree_from_bits(bits):
        if not bits:
            return n
        if len(bits) == 1:
            return bits[0]
        
        wires = list(bits)
        masks = [1 << b for b in bits]
        
        while len(wires) > 1:
            new_wires = []
            new_masks = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    a_w, a_m = wires[i], masks[i]
                    b_w, b_m = wires[i+1], masks[i+1]
                    new_m = a_m ^ b_m
                    if new_m in memo:
                        new_wires.append(memo[new_m])
                    else:
                        new_w = add_gate(a_w, b_w)
                        memo[new_m] = new_w
                        new_wires.append(new_w)
                    new_masks.append(new_m)
                else:
                    new_wires.append(wires[i])
                    new_masks.append(masks[i])
            wires = new_wires
            masks = new_masks
        return wires[0]
    
    # Sort rows: prefer rows that share structure with already-computed masks
    # Use a greedy ordering: at each step, pick the row that minimizes
    # the number of NEW gates needed given current memo
    remaining = list(range(m))
    outputs = [None] * m
    
    # Pre-compute popcounts
    popcounts = [mask.bit_count() for mask in row_masks]
    
    # Order: start with highest popcount rows (they create the most reusable intermediates)
    # but also consider overlap with existing memo
    sorted_indices = sorted(range(m), key=lambda i: (-popcounts[i], i))
    
    for idx in sorted_indices:
        if elapsed() > DEADLINE - 2.0:
            mask = row_masks[idx]
            if mask in memo:
                outputs[idx] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_tree_from_bits(bits)
                memo[mask] = w
                outputs[idx] = w
        else:
            mask = row_masks[idx]
            if mask in memo:
                outputs[idx] = memo[mask]
                continue
            
            # Find best decomposition: mask = A ^ B where A, B are in memo or cheap
            # Try all pairs of memo entries
            best_cost = float('inf')
            best_a = None
            best_b = None
            
            memo_items = list(memo.items())
            
            # If memo is small enough, try all pairs
            if len(memo_items) <= 500:
                for i, (ma, wa) in enumerate(memo_items):
                    for j, (mb, wb) in enumerate(memo_items):
                        if i == j:
                            continue
                        rem = ma ^ mb
                        if rem == mask:
                            # Direct: cost = 1 gate
                            cost = 1
                            if cost < best_cost:
                                best_cost = cost
                                best_a = ma
                                best_b = mb
            else:
                # Greedy: find single best M to XOR with
                best_M = 0
                min_rem_pop = mask.bit_count()
                for M, w in memo_items:
                    rem = mask ^ M
                    pc = rem.bit_count()
                    if pc < min_rem_pop:
                        min_rem_pop = pc
                        best_M = M
                        if pc == 0:
                            break
                        if pc == 1:
                            break
                if best_M == 0:
                    bits = [j for j in range(n) if (mask >> j) & 1]
                    w = build_tree_from_bits(bits)
                    memo[mask] = w
                    outputs[idx] = w
                else:
                    w_m = memo[best_M]
                    rem = mask ^ best_M
                    if rem in memo:
                        w_rem = memo[rem]
                    else:
                        bits = [j for j in range(n) if (rem >> j) & 1]
                        w_rem = build_tree_from_bits(bits)
                        memo[rem] = w_rem
                    w_new = add_gate(w_m, w_rem)
                    memo[mask] = w_new
                    outputs[idx] = w_new
            continue
            
            # Use the best pair found
            if best_a is not None and best_b is not None:
                w_a = memo[best_a]
                w_b = memo[best_b]
                w_new = add_gate(w_a, w_b)
                memo[mask] = w_new
                outputs[idx] = w_new
            else:
                # Fallback
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_tree_from_bits(bits)
                memo[mask] = w
                outputs[idx] = w
    
    # Ensure all outputs are set
    for i in range(m):
        if outputs[i] is None:
            mask = row_masks[i]
            if mask in memo:
                outputs[i] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_tree_from_bits(bits)
                memo[mask] = w
                outputs[i] = w
    
    return {'gates': gates, 'outputs': outputs}
