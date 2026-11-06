import numpy as np
import time

N = 17

def construct():
    # 1. Generate a good seed network (Bitonic Sort)
    def bitonic_network(n):
        comps = []
        def bitonic_merge(size, offset, direction):
            if size == 1:
                return
            half = size // 2
            bitonic_merge(half, offset, direction)
            bitonic_merge(half, offset + half, direction)
            for i in range(half):
                a = offset + i
                b = offset + i + half
                if direction == 0:  # ascending
                    comps.append((a, b))
                else:  # descending
                    comps.append((b, a))
        
        def bitonic_sort(size, offset, direction):
            if size <= 1:
                return
            half = size // 2
            bitonic_sort(half, offset, 1)       # sort first half ascending
            bitonic_sort(half, offset + half, 0) # sort second half descending
            bitonic_merge(size, offset, direction)
        
        bitonic_sort(n, 0, 0)
        return comps
    
    seed_net = bitonic_network(N)
    
    # 2. Precompute all 2^17 inputs as uint8 array for speed
    # inputs[bit_pattern, wire] = 0 or 1
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        # For bit k, create pattern where bit k is set
        # Pattern repeats: 2^k zeros, 2^k ones, 2^k zeros, 2^k ones, ...
        pattern = np.zeros(2**N, dtype=np.uint8)
        period = 2**(k+1)
        half = 2**k
        for start in range(0, 2**N, period):
            pattern[start+half:start+period] = 1
        inputs[:, k] = pattern
    
    # 3. Fast validation using numpy
    def is_valid(net):
        wires = inputs.copy()
        for i, j in net:
            wi = wires[:, i]
            wj = wires[:, j]
            # min to i, max to j
            wires[:, i] = np.minimum(wi, wj)
            wires[:, j] = np.maximum(wi, wj)
        # Check if sorted: wires[:, 0] <= wires[:, 1] <= ... <= wires[:, N-1]
        # For binary inputs, sorted means all 0s come before all 1s
        # So for each row, count the number of 1s, and check position
        # Actually just check: no 1 appears before a 0
        # Equivalent: for each adjacent pair, left <= right
        return np.all(wires[:, :-1] <= wires[:, 1:])
    
    # 4. Optimization with multiple passes
    current_net = seed_net[:]
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    if not is_valid(current_net):
        return current_net
    
    # Pass 1: Remove redundant comparators (multiple rounds)
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        i = 0
        while i < len(current_net):
            if time.time() > deadline:
                return current_net
            candidate = current_net[:i] + current_net[i+1:]
            if is_valid(candidate):
                current_net = candidate
                improved = True
                i = 0  # Restart from beginning
            else:
                i += 1
        if not improved:
            break
    
    # Pass 2: Try moving comparators to different positions
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            i = 0
            while i < len(current_net):
                if time.time() > deadline:
                    return current_net
                comp = current_net[i]
                # Try moving to each other position
                for j in range(len(current_net)):
                    if j == i:
                        continue
                    if time.time() > deadline:
                        return current_net
                    # Move comp from position i to position j
                    if j < i:
                        candidate = current_net[:j] + [comp] + current_net[j:i] + current_net[i+1:]
                    else:
                        candidate = current_net[:i] + current_net[i+1:j+1] + [comp]
                    if is_valid(candidate):
                        current_net = candidate
                        improved = True
                        i = 0
                        break
                if time.time() > deadline:
                    return current_net
                if improved:
                    break
                i += 1
    
    # Pass 3: Another round of removals after moves
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            improved = False
            i = 0
            while i < len(current_net):
                if time.time() > deadline:
                    return current_net
                candidate = current_net[:i] + current_net[i+1:]
                if is_valid(candidate):
                    current_net = candidate
                    improved = True
                    i = 0
                else:
                    i += 1
            if not improved:
                break
    
    return current_net
