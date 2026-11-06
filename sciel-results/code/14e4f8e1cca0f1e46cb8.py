def synthesize(matrix, depth_limit):
    import numpy as np
    from collections import defaultdict
    
    m, n = matrix.shape
    outputs_support = [frozenset(j for j in range(n) if matrix[i, j]) for i in range(m)]
    
    # Count how many times each pair appears in output supports
    pair_count = defaultdict(int)
    for s in outputs_support:
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                pair = (min(lst[i], lst[j]), max(lst[i], lst[j]))
                pair_count[pair] += 1
    
    # Precompute pairs that appear in >= 2 outputs
    precompute_pairs = {p for p, c in pair_count.items() if c >= 2}
    
    # Assign wires
    wire_to_support = {}
    support_to_wire = {}
    for i in range(n):
        s = frozenset([i])
        wire_to_support[i] = s
        support_to_wire[s] = i
    wire_to_support[n] = frozenset()
    support_to_wire[frozenset()] = n
    
    gates = []
    next_wire = n + 1
    
    # Add precomputed pairs
    for (i, j) in precompute_pairs:
        s = frozenset([i, j])
        if s not in support_to_wire:
            gates.append([i, j])
            wire_to_support[next_wire] = s
            support_to_wire[s] = next_wire
            next_wire += 1
    
    # Now express each output
    outputs = []
    for s in outputs_support:
        if s in support_to_wire:
            outputs.append(support_to_wire[s])
        elif len(s) == 0:
            outputs.append(n)  # zero wire
        else:
            # Express s using available wires
            # Greedy: repeatedly XOR with available wire that reduces popcount
            remaining = set(s)
            wires_used = []
            # Find a sequence
            # Simple: pick elements one by one
            # If |s| is small, just XOR inputs
            # If there's a precomputed pair in s, use it
            s_set = set(s)
            # Try to use precomputed pairs
            used_pairs = []
            temp = set(s)
            for (i, j) in precompute_pairs:
                if i in temp and j in temp:
                    used_pairs.append((i, j))
                    temp.discard(i)
                    temp.discard(j)
            # Now temp has remaining single elements
            # Build expression
            # Start with zero
            # For each used pair, XOR the pair wire
            # For each remaining element, XOR the input wire
            # This needs len(used_pairs) + len(temp) - 1 gates if total > 0
            # But we need to create the intermediate wires
            # Actually, we just need to list the wires to XOR
            # The circuit will XOR them in sequence
            # Number of gates = (len(used_pairs) + len(temp)) - 1 if total > 0
            # But we need to add these gates
            parts = []
            for (i, j) in used_pairs:
                parts.append(support_to_wire[frozenset([i, j])])
            for i in temp:
                parts.append(i)  # input wire
            
            if not parts:
                outputs.append(n)
            elif len(parts) == 1:
                outputs.append(parts[0])
            else:
                # XOR all parts
                # This needs len(parts) - 1 gates
                # Create intermediate wires
                current = parts[0]
                for p in parts[1:]:
                    gates.append([current, p])
                    s_new = wire_to_support[current] ^ wire_to_support[p]
                    wire_to_support[next_wire] = s_new
                    support_to_wire[s_new] = next_wire
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
