import numpy as np
import time

N = 17

def construct():
        # 1. Generate Seed Network (Odd-Even Transposition)
        # Phases: 0 to N-1
        # Phase p: compare (p%2, p%2+2, ...), (p%2+1, p%2+3, ...)
        # Actually, standard Odd-Even Transposition:
        # For step in range(N):
        #   For i in range(step % 2, N-1, 2):
        #     Compare (i, i+1)
        
        seed_net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                seed_net.append((i, i + 1))
        
        # 2. Precompute Inputs
        # 2^17 inputs
        # Use np.indices or binary representation
        # Create array of all 17-bit numbers
        inputs = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(2**N)], dtype=np.int8)
        # This list comprehension might be slow.
        # Faster:
        inputs = np.array([[(i >> k) & 1 for k in range(N)] for i in range(2**N)], dtype=np.int8)
        # Even faster:
        # inputs = np.arange(2**N).reshape(-1, 1)
        # inputs = (inputs >> np.arange(N)) & 1
        
        # 3. Optimization Loop
        current_net = seed_net[:]
        start_time = time.time()
        deadline = start_time + 85.0 # 5s margin
        
        def is_valid(net):
            wires = inputs.copy()
            for i, j in net:
                min_val = np.minimum(wires[:, i], wires[:, j])
                max_val = np.maximum(wires[:, i], wires[:, j])
                wires[:, i] = min_val
                wires[:, j] = max_val
            # Check sorted
            return np.all(wires[:, :-1] <= wires[:, 1:])
        
        # Initial check
        if not is_valid(current_net):
            return current_net # Should not happen
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove each comparator
            # Iterate from 0 to len-1
            # If removal succeeds, restart loop from 0
            i = 0
            while i < len(current_net):
                if time.time() > deadline:
                    return current_net
                candidate = current_net[:i] + current_net[i+1:]
                if is_valid(candidate):
                    current_net = candidate
                    improved = True
                    i = 0 # Restart
                else:
                    i += 1
                    
        return current_net
