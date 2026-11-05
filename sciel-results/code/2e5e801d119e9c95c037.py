"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096

def _verify(network):
    """Verify sorting network over all 2^N Boolean inputs using zero-one principle."""
    total = 1 << N
    # Generate all 2^N boolean inputs
    # Use a fast method: for each bit position, create the pattern
    inputs = np.empty((total, N), dtype=np.uint8)
    for i in range(N):
        # Bit i of x for x in 0..total-1
        inputs[:, i] = (np.arange(total, dtype=np.uint32) >> i) & 1
    
    for (i, j) in network:
        a = inputs[:, i].copy()
        b = inputs[:, j].copy()
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    
    # Check sorted: no 1 before 0 in any row
    # Equivalent: for each row, the sequence is non-decreasing
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_remove(network, idx):
    """Try removing comparator at index idx, return (success, new_network)."""
    candidate = network[:idx] + network[idx+1:]
    if _verify(candidate):
        return True, candidate
    return False, None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Start with odd-even transposition network
    network = _odd_even_transposition(N)
    
    # Verify it's correct
    if not _verify(network):
        return network
    
    best_network = list(network)
    best_count = len(network)
    
    # Phase 1: Greedy removal - repeatedly remove redundant comparators
    improved = True
    while improved:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > 75.0:
                return best_network
            
            success, candidate = _try_remove(network, idx)
            if success:
                network = candidate
                improved = True
                best_network = list(network)
                best_count = len(network)
    
    # Phase 2: Local search - try swapping adjacent comparators to enable removals
    if time.time() - start_time < 70.0:
        for _pass in range(5):
            if time.time() - start_time > 70.0:
                break
            for idx in range(len(network) - 1):
                if time.time() - start_time > 70.0:
                    break
                # Try swapping
                candidate = network[:idx] + [network[idx+1], network[idx]] + network[idx+2:]
                if len(candidate) != len(network):
                    continue
                # After swap, try removing each of the two
                for rem in [idx, idx]:
                    cand2 = candidate[:rem] + candidate[rem+1:]
                    if _verify(cand2):
                        network = cand2
                        best_network = list(network)
                        best_count = len(network)
                        break
    
    # Phase 3: Try inserting a comparator to enable removal of two others
    # This is more expensive, so do it sparingly
    if time.time() - start_time < 60.0:
        # Try inserting at each position, then see if we can remove two
        for ins_pos in range(len(network) + 1):
            if time.time() - start_time > 60.0:
                break
            # Try all possible new comparators
            for i in range(N):
                if time.time() - start_time > 60.0:
                    break
                for j in range(i + 1, N):
                    if time.time() - start_time > 60.0:
                        break
                    new_comp = (i, j)
                    # Skip if already present at adjacent positions
                    candidate = network[:ins_pos] + [new_comp] + network[ins_pos:]
                    # Now try to remove two comparators
                    for rem1 in range(len(candidate) - 1):
                        if time.time() - start_time > 60.0:
                            break
                        cand1 = candidate[:rem1] + candidate[rem1+1:]
                        if len(cand1) >= len(network):
                            continue
                        for rem2 in range(len(cand1) - 1):
                            if time.time() - start_time > 60.0:
                                break
                            cand2 = cand1[:rem2] + cand1[rem2+1:]
                            if len(cand2) < len(network) and _verify(cand2):
                                network = cand2
                                best_network = list(network)
                                best_count = len(network)
                                break
    
    return best_network
# EVOLVE-BLOCK-END
