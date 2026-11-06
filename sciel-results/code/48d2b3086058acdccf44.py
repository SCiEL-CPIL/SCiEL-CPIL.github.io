"""Size-efficient sorting networks on 17 wires via structural optimization."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 58.0  # Safety margin before 60s limit
START_TIME = time.time()

# Precompute initial states for all 2^N inputs using Python big integers.
# Each wire is represented by an integer where bit m is the value of wire k for input m.
# This allows O(1) bitwise operations for comparators across all 131,072 inputs.
def _init_bit_vectors():
    states = [0] * N
    for k in range(N):
        # Construct integer for wire k
        # Bit m is 1 if (m >> k) & 1 is 1
        # We can build this by iterating or using a pattern.
        # Pattern for wire k: 0...0 1...1 0...0 1...1 ...
        # Block size is 2^k.
        # For efficiency, we can just iterate over a small range if N is small, 
        # but 131k bits is small for big ints.
        val = 0
        # We can generate this using a loop over blocks
        # Actually, simpler: 
        # For m in range(NUM_INPUTS):
        #   if (m >> k) & 1: val |= (1 << m)
        # This loop is 17 * 131k = 2.2M iterations. Acceptable for init.
        # But we can do it faster.
        # Let's stick to the simple loop for clarity, or use a helper.
        # Since this is only done once, it's fine.
        pass
    
    # Faster initialization:
    # Wire k has period 2^(k+1). Half 0, half 1.
    # We can construct the mask for wire k.
    for k in range(N):
        mask = 0
        # Create a block of 2^k ones, then 2^k zeros, repeated.
        # Total bits: 2^17.
        # We can use a recursive or iterative construction.
        # Let's just use the bit shift method in a loop, it's fast enough in C under the hood? 
        # No, Python loop is slow.
        # Let's use NumPy to generate the initial big ints?
        # Or just use the property:
        # val = 0
        # for m in range(NUM_INPUTS):
        #     if (m >> k) & 1:
        #         val |= 1 << m
        # This is 131k * 17 = 2.2M operations. In Python, this might take ~1-2 seconds.
        # Let's optimize init.
        
        # Alternative: Use NumPy to create the array, then pack into big ints?
        # Or just accept the 1-2s cost? The deadline is 58s.
        # Let's try to optimize the loop.
        
        # Actually, we can compute it as:
        # block_ones = (1 << (1 << k)) - 1
        # pattern = block_ones
        # while pattern.bit_length() < NUM_INPUTS:
        #     pattern |= pattern << (1 << (k+1))
        # This is very fast.
        
        block_len = 1 << k
        block_ones = (1 << block_len) - 1
        val = 0
        shift = 0
        while shift < NUM_INPUTS:
            val |= (block_ones << shift)
            shift += block_len * 2
        states[k] = val
        
    return states

INITIAL_STATES = _init_bit_vectors()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_network_fast(comps):
    """
    Verify network using Python big integer bit-parallel simulation.
    comps: list of (i, j) tuples.
    Returns True if valid.
    """
    if not comps:
        return False
        
    # Copy initial states (list of ints)
    state = INITIAL_STATES[:]
    
    for i, j in comps:
        si = state[i]
        sj = state[j]
        
        # Comparator (i, j): min to i, max to j.
        # min(si, sj) bitwise is si & sj
        # max(si, sj) bitwise is si | sj
        state[i] = si & sj
        state[j] = si | sj
        
    # Check if sorted: for all k, state[k] <= state[k+1]
    # This means no bit is 1 in state[k] and 0 in state[k+1].
    # i.e., state[k] & ~state[k+1] == 0
    for k in range(N - 1):
        if state[k] & ~state[k + 1]:
            return False
            
    return True

def make_seed():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def greedy_remove(comps):
    """
    Try to remove each comparator if the network remains valid.
    Uses fast verification.
    """
    n = len(comps)
    improved = True
    while improved:
        improved = False
        for idx in range(n - 1, -1, -1):
            if _time_left() < 2.0:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if verify_network_fast(candidate):
                comps = candidate
                n = len(comps)
                improved = True
                break # Restart scan
    return comps

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Start with odd-even transposition
    comps = make_seed()
    
    # First pass: aggressive greedy removal
    comps = greedy_remove(comps)
    
    best = list(comps)
    best_len = len(best)
    
    # Local Search with Structural Moves
    # We use a combination of:
    # 1. Removal (primary goal)
    # 2. Swap (to escape local minima where removal is blocked)
    # 3. Move (to reorder comparators, potentially enabling new removals)
    # 4. "Simulated Annealing" style acceptance of same-size valid moves to explore
    
    current = list(best)
    current_len = best_len
    temperature = 10.0  # Initial "temperature" for accepting non-improving moves
    
    last_improvement = time.time()
    
    while _time_left() > 1.0:
        # Generate a candidate via mutation
        candidate = list(current)
        n = len(candidate)
        
        # Choose operation
        # Bias towards removal if we are getting stuck
        if n > 0:
            # 60% remove, 20% swap, 20% move
            op = rng.random()
            if op < 0.6:
                op = 'remove'
            elif op < 0.8:
                op = 'swap'
            else:
                op = 'move'
        else:
            op = 'remove'
            
        if op == 'remove':
            if n == 0:
                continue
            idx = rng.randrange(n)
            candidate.pop(idx)
        elif op == 'swap' and n >= 2:
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            if i1 != i2:
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        elif op == 'move' and n >= 1:
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            if i1 != i2:
                elem = candidate.pop(i1)
                candidate.insert(i2, elem)
        
        # Evaluate candidate
        is_valid = verify_network_fast(candidate)
        
        if is_valid:
            new_len = len(candidate)
            
            if new_len < current_len:
                # Improvement
                current = candidate
                current_len = new_len
                last_improvement = time.time()
                
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
                    # Reset temperature on improvement
                    temperature = max(1.0, temperature * 0.5)
                    
                # Try greedy removal immediately after improvement
                current = greedy_remove(current)
                current_len = len(current)
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
                    
            elif new_len == current_len:
                # Same size, valid.
                # Accept with some probability to explore (Simulated Annealing style)
                # This helps escape local minima where no single removal works,
                # but a sequence of swaps/moves followed by a removal does.
                if rng.random() < 0.3:  # 30% chance to accept neutral move
                    current = candidate
                    # Run a quick greedy check to see if this new configuration allows removal
                    temp = greedy_remove(current)
                    if len(temp) < current_len:
                        current = temp
                        current_len = len(current)
                        last_improvement = time.time()
                        if current_len < best_len:
                            best = list(current)
                            best_len = current_len
                            temperature = max(1.0, temperature * 0.5)
                    else:
                        # No immediate gain, but we moved.
                        # Slight cooling
                        temperature = max(0.5, temperature * 0.95)
            else:
                # Larger size (should not happen with remove/swap/move, but just in case)
                pass
        else:
            # Invalid move
            # Cool down slightly
            temperature = max(0.5, temperature * 0.98)
            
        # If no improvement for a long time, increase temperature (restart exploration)
        if time.time() - last_improvement > 5.0:
            temperature = min(20.0, temperature * 1.2)
            # Occasionally, restart from a slightly perturbed best to escape deep local minima
            if rng.random() < 0.1:
                current = list(best)
                # Perturb best
                for _ in range(rng.randint(1, 3)):
                    if len(current) > 0:
                        idx = rng.randrange(len(current))
                        current.pop(idx)
                        if not verify_network_fast(current):
                            current.insert(idx, best[idx])
                current_len = len(current)
                last_improvement = time.time()

    # Final greedy pass
    best = greedy_remove(best)
    
    return best

if __name__ == "__main__":
    # Quick test
    net = construct()
    print(f"Network size: {len(net)}")
    if verify_network_fast(net):
        print("Network is valid.")
    else:
        print("Network is INVALID!")
