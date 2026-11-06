import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 54.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert to list of lists of 1s indices for each row
    rows = []
    for i in range(m):
        ones = [j for j in range(n) if matrix[i][j] % 2 == 1]
        rows.append(ones)

    gates = []
    wire_cache = {} # frozenset of input indices -> wire id
    
    # Constant zero wire
    zero_wire = n
    
    def get_wire(indices):
        """
        Returns a wire ID for the XOR of the given input indices.
        Uses global cache and balanced tree construction.
        """
        if not indices:
            return zero_wire
        if len(indices) == 1:
            return indices[0]
            
        key = frozenset(indices)
        if key in wire_cache:
            return wire_cache[key]
        
        # Check deadline
        if time.time() > deadline:
            # Fallback: just chain them, might violate depth but returns something
            # Actually, if we time out, we should return best effort.
            # But for small n, this shouldn't happen.
            pass

        # Balanced split
        # Sort to ensure consistent splitting? 
        # The order of indices doesn't matter for XOR, but for cache key consistency, 
        # frozenset handles that. For splitting, we just need two non-empty lists.
        # Sorting helps if we want to split by value, but random/arbitrary split is fine 
        # as long as it's balanced.
        # To maximize cache hits across different rows, it's better to have a canonical 
        # way to split. Sorting the indices and splitting in the middle is canonical.
        sorted_indices = sorted(indices)
        mid = len(sorted_indices) // 2
        left_indices = sorted_indices[:mid]
        right_indices = sorted_indices[mid:]
        
        left_wire = get_wire(left_indices)
        right_wire = get_wire(right_indices)
        
        gate_id = len(gates)
        gates.append([left_wire, right_wire])
        wire_id = n + 1 + gate_id
        
        wire_cache[key] = wire_id
        return wire_id

    outputs = []
    for i in range(m):
        if time.time() > deadline:
            # Emergency fallback: naive linear XOR without caching to avoid infinite recursion/complexity
            # This is unlikely to be hit for n<=16
            ones = rows[i]
            if not ones:
                outputs.append(zero_wire)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = n + 1 + len(gates) - 1
                outputs.append(current)
        else:
            outputs.append(get_wire(rows[i]))

    return {'gates': gates, 'outputs': outputs}
