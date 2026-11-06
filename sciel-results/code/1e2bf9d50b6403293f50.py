"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 18

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s kill
    
    # Precompute all 2^N Boolean inputs as a (N, 2^N) boolean matrix
    total = 1 << N
    all_inputs = np.zeros((N, total), dtype=np.uint8)
    for i in range(N):
        period = 1 << (i + 1)
        half = 1 << i
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[half:] = 1
        all_inputs[i] = np.tile(pattern, total // period)

    def _verify(network):
        """Vectorized verification: returns True if network sorts all inputs."""
        if (time.time() - start_time) > deadline:
            return False
        x = all_inputs.copy()
        for i, j in network:
            mask = (x[i] > x[j])
            if np.any(mask):
                x[i][mask] = 0
                x[j][mask] = 1
        
        # Check if all columns are sorted (non-decreasing)
        for i in range(N - 1):
            if np.any((x[i] == 1) & (x[i+1] == 0)):
                return False
        return True

    def _bitonic_sort(n):
        """
        Generates a bitonic sorting network for n wires.
        """
        def gen(n, asc, offset=0):
            if n == 1:
                return []
            mid = n // 2
            left = gen(mid, asc, offset)
            right = gen(n - mid, not asc, offset + mid)
            
            merge_comps = []
            k = n // 2
            while k > 0:
                for i in range(n - k):
                    # i is relative to offset
                    # Global indices: offset + i, offset + i + k
                    if (i // k) % 2 == 0:
                        merge_comps.append((offset + i, offset + i + k))
                    else:
                        merge_comps.append((offset + i + k, offset + i))
                k //= 2
            
            return left + right + merge_comps

        return gen(n, True, 0)

    # Generate initial network
    network = _bitonic_sort(N)
    
    # Verify initial network
    if not _verify(network):
        # Fallback to odd-even if bitonic fails (should not happen)
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        if not _verify(network):
            return network

    # Optimization Loop
    improved = True
    while improved and (time.time() - start_time) < deadline - 2:
        improved = False
        
        # 1. Try removing each comparator
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 2:
                break
            
            test_net = network[:idx] + network[idx+1:]
            if _verify(test_net):
                network = test_net
                improved = True
                break
        
        if improved:
            continue

        # 2. Try swapping adjacent comparators
        for idx in range(len(network) - 1):
            if (time.time() - start_time) > deadline - 2:
                break
            
            test_net = network[:]
            test_net[idx], test_net[idx+1] = test_net[idx+1], test_net[idx]
            
            if _verify(test_net):
                network = test_net
                improved = True
                break
        
        if improved:
            continue

        # 3. Try random re-insertion (move a comparator to a new position)
        # This helps escape local minima where order matters
        if len(network) > 1:
            for _ in range(5): # Try a few random moves
                if (time.time() - start_time) > deadline - 2:
                    break
                
                rem_idx = random.randint(0, len(network) - 1)
                comp = network[rem_idx]
                base_net = network[:rem_idx] + network[rem_idx+1:]
                
                ins_idx = random.randint(0, len(base_net))
                test_net = base_net[:ins_idx] + [comp] + base_net[ins_idx:]
                
                if _verify(test_net):
                    network = test_net
                    improved = True
                    break

    return network
# EVOLVE-BLOCK-END
