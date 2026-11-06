import time
import numpy as np

N = 17

def _verify_network(comps, num_inputs=None):
    """
    Verify if the network sorts all inputs.
    If num_inputs is None, uses all 2^N inputs.
    Otherwise, uses num_inputs random samples.
    """
    if num_inputs is None:
        num_inputs = 1 << N
    
    # Generate inputs
    # For full verification, we can use a more efficient method if needed, 
    # but 131k * 17 is small enough for random sampling or full enumeration.
    # Full enumeration of 2^17 is 131072.
    # We can generate them using np.arange and bit operations.
    
    if num_inputs == (1 << N):
        # Generate all 0-1 inputs
        # Using np.arange for speed
        inputs = np.arange(num_inputs, dtype=np.uint8).reshape(-1, 1)
        # This creates 0, 1, 2, ... 
        # We need to expand to binary representation
        # Bitwise: (inputs >> k) & 1
        # But inputs is 1D. 
        # Let's use a standard method:
        # np.unpackbits is fast.
        # But we need 17 bits.
        # Let's just use random sampling for speed in the search loop, 
        # and full enumeration for the final check.
        # Actually, 131072 is small. Let's use random for the search 
        # and full for the candidate.
        pass

    # For the search loop, we use random sampling.
    # For the final check, we use full enumeration.
    
    # Generate random inputs
    inputs = np.random.randint(0, 2, size=(num_inputs, N), dtype=np.uint8)
    
    # Apply comparators
    for i, j in comps:
        a = inputs[:, i]
        b = inputs[:, j]
        # min to i, max to j
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
        
    # Check if sorted
    # diff >= 0
    diffs = np.diff(inputs, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _full_verify(comps):
    """
    Full verification over all 2^N inputs.
    """
    num_inputs = 1 << N
    # Generate all inputs
    # Method: np.arange(2^N) -> binary
    # This is memory efficient if done right.
    # 131072 * 17 bytes = ~2.2 MB. Very small.
    
    # Create indices
    idx = np.arange(num_inputs, dtype=np.uint32)
    # Expand to bits
    # We can use np.unpackbits but it works on uint8.
    # Let's do it in chunks or just use random if full is too slow?
    # 131k rows is fast.
    
    # Actually, a faster way to generate all 0-1 inputs:
    # inputs = np.zeros((num_inputs, N), dtype=np.uint8)
    # for k in range(N):
    #     inputs[:, k] = (idx >> k) & 1
    
    # This loop is N=17 iterations. Fast.
    inputs = np.zeros((num_inputs, N), dtype=np.uint8)
    for k in range(N):
        inputs[:, k] = (idx >> k) & 1
        
    # Apply comparators
    for i, j in comps:
        a = inputs[:, i]
        b = inputs[:, j]
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
        
    # Check sorted
    diffs = np.diff(inputs, axis=1)
    if np.any(diffs < 0):
        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    # Seed: Odd-even transposition
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    # If time is very tight, return seed
    # But we have 60s.
    
    t_start = time.time()
    deadline = t_start + 55  # 5s margin
    
    best_net = seed
    best_size = len(seed)
    current_net = list(seed)
    
    # Local Search
    # Moves: Remove, Swap
    # We use random sampling for quick checks in the loop.
    # We use full verification when a smaller network is found.
    
    import random
    random.seed(42)
    
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() < deadline:
        if len(current_net) == 0:
            break
            
        # Choose a move
        move_type = random.choice(['remove', 'swap'])
        
        if move_type == 'remove':
            if len(current_net) <= 1:
                continue
            idx = random.randint(0, len(current_net) - 1)
            comp = current_net[idx]
            new_net = current_net[:idx] + current_net[idx+1:]
            
            # Quick check
            if _verify_network(new_net, num_inputs=1024):
                # Full check
                if _full_verify(new_net):
                    current_net = new_net
                    if len(current_net) < best_size:
                        best_net = current_net
                        best_size = len(current_net)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
                
        elif move_type == 'swap':
            if len(current_net) < 2:
                continue
            i1 = random.randint(0, len(current_net) - 1)
            i2 = random.randint(0, len(current_net) - 1)
            if i1 == i2:
                continue
            new_net = list(current_net)
            new_net[i1], new_net[i2] = new_net[i2], new_net[i1]
            
            # Quick check
            if _verify_network(new_net, num_inputs=1024):
                # Full check
                if _full_verify(new_net):
                    current_net = new_net
                    # Size didn't change, but we might be in a better local minimum
                    no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
                
        # Restart if stuck
        if no_improvement_count > max_no_improvement:
            current_net = list(best_net)
            no_improvement_count = 0
            # Maybe perturb?
            # For now, just restart from best.
            
    return best_net
# EVOLVE-BLOCK-END
