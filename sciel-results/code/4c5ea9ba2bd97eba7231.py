import time
import random
import numpy as np

N = 17

def _generate_bitonic_16():
    """Generates a bitonic sorting network for 16 wires."""
    comparators = []
    
    def bitonic_sort(k, n, rev=False):
        if k <= 1:
            return
        bitonic_sort(k // 2, n, rev)
        bitonic_sort(k // 2, n + k // 2, not rev)
        bitonic_merge(k, n, rev)

    def bitonic_merge(k, n, rev=False):
        if k <= 1:
            return
        for i in range(n, n + k - 1):
            j = n + k - (i - n + 1)
            if i == j:
                continue
            if (i < j) != rev:
                comparators.append((i, j))
            else:
                comparators.append((j, i))
        bitonic_merge(k // 2, n, rev)
        bitonic_merge(k // 2, n + k // 2, rev)

    # Sort 16 wires (0 to 15)
    bitonic_sort(16, 0)
    return comparators

def _generate_merge_17():
    """Generates comparators to merge wire 16 into sorted wires 0-15."""
    comparators = []
    for i in range(15, -1, -1):
        comparators.append((i, 16))
    return comparators

def _verify_network(net):
    """Verifies if the network sorts all 2^17 inputs using NumPy."""
    # Create all possible inputs as a matrix of shape (2^N, N)
    # Using bit extraction
    inputs = np.array([list(map(int, f"{i:017b}")) for i in range(2**N)])
    # Note: f"{i:017b}" gives MSB first. We want wire 0 to be LSB or MSB?
    # The problem statement says "min to wire i".
    # Usually wire 0 is index 0.
    # If we use standard binary string, left is MSB.
    # Let's stick to wire 0 being the first element in the list.
    # My previous logic: wire 0 is LSB.
    # Let's be consistent.
    # If wire 0 is LSB, then input i has wire k = (i >> k) & 1.
    
    # Let's use the LSB convention for consistency with the bitmask check.
    # inputs[j, k] = (j >> k) & 1
    inputs = np.array([[(j >> k) & 1 for k in range(N)] for j in range(2**N)])
    
    for i, j in net:
        # Extract columns
        col_i = inputs[:, i]
        col_j = inputs[:, j]
        # min to i, max to j
        # if col_i > col_j, swap
        mask = col_i > col_j
        inputs[:, i] = np.where(mask, col_j, col_i)
        inputs[:, j] = np.where(mask, col_i, col_j)
        
    # Check if sorted
    # Sorted means 0s at low indices, 1s at high indices.
    # So for each row, it must be 0...01...1
    # This is equivalent to: row is non-decreasing?
    # 0,0,1,1 is non-decreasing.
    # Check: np.diff(row) >= 0
    # Vectorized check:
    # diff = np.diff(inputs, axis=1)
    # return np.all(diff >= 0)
    
    diff = np.diff(inputs, axis=1)
    return np.all(diff >= 0)

def construct():
    # Initial network
    net_16 = _generate_bitonic_16()
    merge_17 = _generate_merge_17()
    current_net = net_16 + merge_17
    
    # Verify initial (should be True)
    if not _verify_network(current_net):
        # Fallback to odd-even if something went wrong
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = current_net[:]
    
    # Local Search
    deadline = time.time() + 85.0
    random.seed(42)
    
    while time.time() < deadline:
        if not current_net:
            break
            
        # Try removing a random comparator
        idx = random.randint(0, len(current_net) - 1)
        new_net = current_net[:idx] + current_net[idx+1:]
        
        if _verify_network(new_net):
            current_net = new_net
            best_net = current_net[:]
        else:
            # Optional: Try swapping two comparators?
            # For now, just stick to removal to keep it simple and fast.
            pass
            
    return best_net
