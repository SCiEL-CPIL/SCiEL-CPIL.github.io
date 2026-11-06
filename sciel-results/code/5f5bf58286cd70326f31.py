"""
Size-efficient sorting networks on 17 wires.

Strategy:
1. Start from a known-good seed: odd-even transposition network (136 comparators).
2. Use simulated annealing / hill climbing with random mutations to reduce comparator count.
3. Mutations: remove a random comparator, swap order of two comparators, insert a new comparator.
4. Validate correctness using the zero-one principle (all 2^17 = 131072 Boolean inputs).
5. Keep an explicit deadline guard to finish well within the 60s limit.
"""

import time
import random
import itertools

N = 17
SEED_COMPARATORS = None
DEADLINE = 55.0  # seconds, safety margin before 60s limit

def _get_seed():
    """Generate the odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _verify_network(comps):
    """
    Verify that the comparator network sorts all 2^N Boolean inputs.
    Uses the zero-one principle.
    Returns True if the network is correct, False otherwise.
    """
    n = len(comps)
    if n == 0:
        return N <= 1
    
    # Precompute the comparator indices as tuples for faster access
    comp_list = list(comps)
    
    # We test all 2^N Boolean inputs
    # For efficiency, we can use bit operations
    for mask in range(1 << N):
        # Initialize the wires with 0/1 based on the mask
        wires = [(mask >> i) & 1 for i in range(N)]
        
        # Apply each comparator
        for a, b in comp_list:
            if wires[a] > wires[b]:
                wires[a], wires[b] = wires[b], wires[a]
        
        # Check if sorted: wires should be non-decreasing
        for i in range(N - 1):
            if wires[i] > wires[i + 1]:
                return False
    return True

def _verify_network_fast(comps):
    """
    Faster verification using vectorized bit operations.
    Tests all 2^N Boolean inputs at once using integer bit manipulation.
    """
    n = len(comps)
    if n == 0:
        return N <= 1
    
    # We'll process all 2^N inputs simultaneously.
    # Represent the state of all inputs as a list of integers, where each integer
    # encodes the wire values for all inputs at once.
    # Actually, let's use a different approach: for each wire, maintain a bitmask
    # of length 2^N where bit k is 1 if wire i is 1 for input k.
    
    # Initialize: wire i is 1 for input k iff bit i of k is 1
    # So wire_mask[i] has bits set for all k where (k >> i) & 1 == 1
    # That is, wire_mask[i] = sum of 2^k for k where bit i of k is 1
    # This is a pattern: for i, the mask has 2^i consecutive 1s, then 2^i 0s, repeating.
    
    # Precompute wire masks
    wire_masks = []
    for i in range(N):
        m = 0
        period = 1 << (i + 1)
        for block_start in range(0, 1 << N, period):
            for k in range(block_start + (1 << i), block_start + period):
                if k < (1 << N):
                    m |= (1 << k)
        wire_masks.append(m)
    
    # Apply comparators
    for a, b in comps:
        # min goes to a, max goes to b
        # min(wire_a, wire_b) = wire_a & wire_b
        # max(wire_a, wire_b) = wire_a | wire_b
        wa = wire_masks[a]
        wb = wire_masks[b]
        wire_masks[a] = wa & wb
        wire_masks[b] = wa | wb
    
    # Check if sorted: for each adjacent pair (i, i+1), we need wire_i <= wire_{i+1}
    # i.e., wire_i & wire_{i+1} == wire_i (wire_i is subset of wire_{i+1})
    for i in range(N - 1):
        if wire_masks[i] & ~wire_masks[i + 1] != 0:
            return False
    return True

def construct():
    global SEED_COMPARATORS
    start_time = time.time()
    
    if SEED_COMPARATORS is None:
        SEED_COMPARATORS = _get_seed()
    
    # Verify seed is correct
    # For speed, use the fast verification
    if not _verify_network_fast(SEED_COMPARATORS):
        # Fallback to slow verification
        if not _verify_network(SEED_COMPARATORS):
            raise ValueError("Seed network is incorrect")
    
    best_comps = list(SEED_COMPARATORS)
    best_count = len(best_comps)
    
    # Simulated annealing
    current_comps = list(best_comps)
    current_count = len(current_comps)
    
    temperature = 0.05
    cooling = 0.9995
    min_temperature = 0.001
    
    random.seed(42)
    
    while time.time() - start_time < DEADLINE:
        # Check if we should continue
        if temperature < min_temperature:
            # Restart with higher temperature if we've been stuck
            temperature = 0.02
            current_comps = list(best_comps)
            current_count = len(current_comps)
        
        # Generate a mutation
        mutation_type = random.random()
        
        if mutation_type < 0.5 and len(current_comps) > 1:
            # Remove a random comparator
            idx = random.randrange(len(current_comps))
            new_comps = current_comps[:idx] + current_comps[idx+1:]
        elif mutation_type < 0.75 and len(current_comps) >= 2:
            # Swap two comparators
            i = random.randrange(len(current_comps))
            j = random.randrange(len(current_comps))
            if i != j:
                new_comps = list(current_comps)
                new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
            else:
                new_comps = list(current_comps)
        elif mutation_type < 0.9:
            # Insert a random comparator
            a = random.randrange(N)
            b = random.randrange(N)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            idx = random.randrange(len(current_comps) + 1)
            new_comps = current_comps[:idx] + [(a, b)] + current_comps[idx:]
        else:
            # Move a comparator to a new position
            if len(current_comps) < 2:
                continue
            i = random.randrange(len(current_comps))
            j = random.randrange(len(current_comps))
            if i == j:
                continue
            new_comps = list(current_comps)
            comp = new_comps.pop(i)
            new_comps.insert(j, comp)
        
        if len(new_comps) > 4096:
            continue
        
        # Verify the new network
        is_correct = _verify_network_fast(new_comps)
        
        if is_correct:
            new_count = len(new_comps)
            delta = new_count - current_count
            
            if delta < 0:
                # Accept improvement
                current_comps = new_comps
                current_count = new_count
                if new_count < best_count:
                    best_comps = list(new_comps)
                    best_count = new_count
            else:
                # Accept worse with probability exp(-delta / temperature)
                if random.random() < (2.718281828 ** (-delta / max(temperature, 1e-10))):
                    current_comps = new_comps
                    current_count = new_count
        
        temperature *= cooling
    
    return best_comps

# Pre-generate seed at module load to save time in construct()
SEED_COMPARATORS = _get_seed()
