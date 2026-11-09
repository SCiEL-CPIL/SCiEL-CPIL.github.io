"""Exact XOR-circuit synthesis with Greedy Common Subexpression Elimination."""

def synthesize(matrix, depth_limit):
    import numpy as np
    from collections import defaultdict

    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    # Wire numbering: 0..n-1 are inputs, n is constant zero
    # Gates produce wires n+1, n+2, ...
    
    gates = []
    
    def new_gate(a, b):
        gates.append([int(a), int(b)])
        return n + 1 + (len(gates) - 1)
        
    # Initialize output supports as sets of input indices
    # If a row is all zeros, support is empty (output n)
    # If a row has 1 bit, support is {i} (output i)
    # Otherwise, support is the set of 1s.
    
    supports = []
    for row in matrix:
        indices = set(np.where(row > 0)[0].tolist())
        supports.append(indices)
        
    # Greedy CSE
    # We repeatedly look for a pair (u, v) that appears in the most number of supports.
    # If count >= 2, it is beneficial to create a gate u XOR v.
    # Why >= 2? 
    # Cost: 1 gate for u^v.
    # Benefit: In each of the k supports, we replace {u, v} with {w}. 
    # The final XOR for that support reduces by 1 operation (size k-1 -> k-2).
    # Total gain: k - 1 gates saved in final phase.
    # Net: 1 - k. If k=1, net 0 (but we added a gate, so strictly worse or equal if we count the gate). 
    # Actually, if k=1, we add 1 gate, save 1 gate in final XOR. Net 0. But we increase wire count. 
    # Usually we want strict improvement, so k >= 2 gives net saving of at least 1.
    
    while True:
        # Count frequencies of all pairs in all supports
        freq = defaultdict(int)
        for sup in supports:
            if len(sup) < 2:
                continue
            # Get all pairs in this support
            # Optimization: if support is large, this is O(|sup|^2)
            items = list(sup)
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    u, v = items[i], items[j]
                    if u > v:
                        u, v = v, u
                    freq[(u, v)] += 1
        
        # Find max frequency pair
        max_freq = 0
        best_pair = None
        for pair, count in freq.items():
            if count > max_freq:
                max_freq = count
                best_pair = pair
                
        if max_freq < 2:
            break
            
        u, v = best_pair
        w = new_gate(u, v)
        
        # Update all supports containing u and v
        for i, sup in enumerate(supports):
            if u in sup and v in sup:
                sup.remove(u)
                sup.remove(v)
                sup.add(w)
                
    # Build final outputs
    outputs = []
    for sup in supports:
        if len(sup) == 0:
            outputs.append(n)
        elif len(sup) == 1:
            outputs.append(list(sup)[0])
        else:
            # XOR all wires in the support
            wires = list(sup)
            result = wires[0]
            for w in wires[1:]:
                result = new_gate(result, w)
            outputs.append(result)
            
    return {'gates': gates, 'outputs': outputs}
