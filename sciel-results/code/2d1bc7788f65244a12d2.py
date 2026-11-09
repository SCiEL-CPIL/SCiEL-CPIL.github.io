"""XOR circuit synthesis with optimized greedy approach."""
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to integer bitmasks for fast XOR
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)
    
    # Map from bitmask value to wire index
    val_to_wires = {}
    
    # Initialize with input wires
    for j in range(n):
        val = 1 << j
        val_to_wires[val] = j
    
    # Constant zero
    val_to_wires[0] = n
    
    gates = []
    next_wire = n + 1
    
    # Sort rows by popcount (sparse first) for better sharing
    row_indices = sorted(range(m), key=lambda i: bin(rows[i]).count('1'))
    
    # Track which rows have been computed
    row_to_wire = [None] * m
    
    for idx in row_indices:
        if time.time() > deadline:
            break
        
        target = rows[idx]
        
        # Check if already available
        if target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target]
            continue
        
        # Try to find two existing wires whose XOR equals target
        # i.e., find a in available such that target ^ a is also available
        found = False
        
        for a, a_wire in val_to_wires.items():
            b = target ^ a
            if b in val_to_wires:
                b_wire = val_to_wires[b]
                # Avoid using the same wire twice (unless it's zero)
                if a_wire == b_wire and target != 0:
                    continue
                # Create gate: a_wire XOR b_wire = new_wire
                gates.append([a_wire, b_wire])
                row_to_wire[idx] = next_wire
                val_to_wires[target] = next_wire
                next_wire += 1
                found = True
                break
        
        if not found:
            # Fallback: build target from inputs
            if target == 0:
                row_to_wire[idx] = n
            else:
                # Find lowest set bit
                lowest_bit = (target & (-target)).bit_length() - 1
                input_wire = lowest_bit
                remaining = target ^ (1 << lowest_bit)
                
                if remaining == 0:
                    row_to_wire[idx] = input_wire
                else:
                    # Build chain of XORs with inputs
                    current_wire = input_wire
                    current_val = 1 << lowest_bit
                    rem = remaining
                    while rem:
                        if time.time() > deadline:
                            break
                        lb = (rem & (-rem)).bit_length() - 1
                        gates.append([current_wire, lb])
                        current_wire = next_wire
                        next_wire += 1
                        current_val ^= (1 << lb)
                        val_to_wires[current_val] = current_wire
                        rem ^= (1 << lb)
                    row_to_wire[idx] = current_wire
    
    # Build outputs in original order
    for i in range(m):
        if row_to_wire[i] is None:
            row_to_wire[i] = n
    
    outputs = row_to_wire
    
    return {'gates': gates, 'outputs': outputs}
