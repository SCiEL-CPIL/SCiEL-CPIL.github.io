def synthesize(matrix, depth_limit):
    import time
    from collections import defaultdict
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    m, n = matrix.shape
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        rows.append(mask)
    
    # Count pair frequencies
    pair_count = defaultdict(int)
    for mask in rows:
        if mask == 0:
            continue
        bits = []
        tmp = mask
        while tmp:
            lb = tmp & (-tmp)
            idx = lb.bit_length() - 1
            bits.append(idx)
            tmp ^= lb
        for i in range(len(bits)):
            for j in range(i+1, len(bits)):
                pair = (min(bits[i], bits[j]), max(bits[i], bits[j]))
                pair_count[pair] += 1
    
    # Create intermediate wires for pairs with count >= 2
    useful_pairs = [(count, pair) for pair, count in pair_count.items() if count >= 2]
    useful_pairs.sort(reverse=True)
    
    # Build circuit
    gates = []
    bitmask_to_wire = {}
    for j in range(n):
        bitmask_to_wire[(1 << j)] = j
    bitmask_to_wire[0] = n
    
    next_wire = n + 1
    
    for count, (a, b) in useful_pairs:
        if time.time() > deadline:
            break
        pair_mask = (1 << a) | (1 << b)
        if pair_mask not in bitmask_to_wire:
            gates.append([a, b])
            bitmask_to_wire[pair_mask] = next_wire
            next_wire += 1
    
    # Compute outputs
    outputs = []
    for mask in rows:
        if time.time() > deadline:
            # Fallback
            bits = []
            tmp = mask
            while tmp:
                lb = tmp & (-tmp)
                idx = lb.bit_length() - 1
                bits.append(idx)
                tmp ^= lb
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                result = bits[0]
                for w in bits[1:]:
                    gates.append([result, w])
                    result = next_wire
                    next_wire += 1
                outputs.append(result)
            continue
        
        if mask == 0:
            outputs.append(n)
            continue
        
        bits = []
        tmp = mask
        while tmp:
            lb = tmp & (-tmp)
            idx = lb.bit_length() - 1
            bits.append(idx)
            tmp ^= lb
        
        used = [False] * n
        wires_to_xor = []
        
        for i in range(len(bits)):
            if used[bits[i]]:
                continue
            for j in range(i+1, len(bits)):
                if used[bits[j]]:
                    continue
                a, b = bits[i], bits[j]
                pair_mask = (1 << a) | (1 << b)
                if pair_mask in bitmask_to_wire:
                    wires_to_xor.append(bitmask_to_wire[pair_mask])
                    used[a] = True
                    used[b] = True
                    break
        
        for bit in bits:
            if not used[bit]:
                wires_to_xor.append(bit)
        
        if not wires_to_xor:
            outputs.append(n)
        elif len(wires_to_xor) == 1:
            outputs.append(wires_to_xor[0])
        else:
            result = wires_to_xor[0]
            for w in wires_to_xor[1:]:
                gates.append([result, w])
                result = next_wire
                next_wire += 1
            outputs.append(result)
    
    return {'gates': gates, 'outputs': outputs}
