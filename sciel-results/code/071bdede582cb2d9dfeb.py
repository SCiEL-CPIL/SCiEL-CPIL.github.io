import numpy as np
import time
import random

N = 18
NUM_INPUTS = 2 ** N

# Precompute all Boolean inputs
# Shape: (2**18, 18)
# We generate indices 0 to 2**18 - 1
# Bit 0 is wire 0? Or wire 17?
# The problem statement says wires 0 to N-1.
# Usually, bit k of the index corresponds to wire k.
# Let's stick to standard: index bit k -> wire k.
# So input 1 (0b00...01) has wire 0 = 1, others 0.
# Input 2 (0b00...10) has wire 1 = 1, others 0.

def generate_inputs():
    # arange gives 0 to 2**18 - 1
    idx = np.arange(NUM_INPUTS, dtype=np.uint32)[:, None]
    # We want bits 0 to 17
    bits = np.arange(17, -1, -1, dtype=np.uint32) # 17, 16, ..., 0
    # Shift right by bits, and mask with 1
    # This gives columns for wire 17 down to wire 0
    # We want wire 0 to be the first column?
    # The problem says 0 <= i < j < N.
    # It doesn't specify which bit corresponds to which wire in the input generation,
    # as long as the network sorts *all* inputs.
    # So the mapping doesn't matter for correctness, as long as it's a bijection of the space.
    # Let's make column 0 correspond to bit 0 (LSB).
    bits_asc = np.arange(0, N, dtype=np.uint32)
    X = ((idx >> bits_asc) & 1).astype(np.uint8)
    return X

INPUTS = generate_inputs()

def is_valid(net):
    # Copy inputs to avoid modifying original
    # Using a view might be faster if we restore, but copy is safer for parallelism or reuse
    # For speed, we can operate in place on a copy
    X = INPUTS.copy()
    for i, j in net:
        a = X[:, i]
        b = X[:, j]
        # min to i, max to j
        X[:, i] = np.minimum(a, b)
        X[:, j] = np.maximum(a, b)
    
    # Check if sorted
    # X[:, k] <= X[:, k+1] for all k
    return np.all(X[:, :-1] <= X[:, 1:])

def construct():
    # Seed: Odd-even transposition
    # Phases: 0 to N-1
    # Phase p: comparators (i, i+1) where i % 2 == p % 2
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    
    best_net = net[:]
    best_cost = len(net)
    
    # Deadline: 85 seconds
    deadline = time.time() + 85
    
    # Helper to try a mutation and update best
    def try_mutation(current_net):
        nonlocal best_net, best_cost
        # Pick a random mutation type
        r = random.random()
        new_net = current_net[:]
        
        if r < 0.3:
            # Delete a comparator
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
        elif r < 0.6:
            # Swap endpoints of a comparator
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                i, j = new_net[idx]
                # Try to find a new valid pair
                # Random swap
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i == new_j:
                    return False
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                new_net[idx] = (new_i, new_j)
        elif r < 0.8:
            # Insert a comparator
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i == new_j:
                return False
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            idx = random.randint(0, len(new_net))
            new_net.insert(idx, (new_i, new_j))
        else:
            # Move a comparator
            if len(new_net) > 1:
                idx1 = random.randint(0, len(new_net) - 1)
                idx2 = random.randint(0, len(new_net) - 1)
                if idx1 != idx2:
                    comp = new_net.pop(idx1)
                    new_net.insert(idx2, comp)
        
        # Validate
        if is_valid(new_net):
            cost = len(new_net)
            if cost < best_cost:
                best_cost = cost
                best_net = new_net[:]
            # Accept if better or equal, or with some probability if worse (SA-like)
            # For now, accept if <= current cost to ensure progress
            # But we need to handle the "current" state.
            # The caller manages current_net.
            return True, new_net
        return False, None

    # Phase 1: Greedy Reduction
    # Try to remove comparators one by one
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate backwards to handle indices safely if deleting
        for i in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing net[i]
            test_net = net[:]
            test_net.pop(i)
            if is_valid(test_net):
                net = test_net
                improved = True
                if len(net) < best_cost:
                    best_cost = len(net)
                    best_net = net[:]
        
        # Try swapping endpoints to enable further removals
        for i in range(len(net)):
            if time.time() > deadline:
                break
            orig_i, orig_j = net[i]
            # Try a few random swaps
            for _ in range(10):
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i == new_j:
                    continue
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if (new_i, new_j) == (orig_i, orig_j):
                    continue
                test_net = net[:]
                test_net[i] = (new_i, new_j)
                if is_valid(test_net):
                    net = test_net
                    improved = True
                    if len(net) < best_cost:
                        best_cost = len(net)
                        best_net = net[:]
                    break # Accept first valid swap found

    # Phase 2: Local Search
    current_net = net[:]
    # Use Simulated Annealing or Hill Climbing
    # Since we want to minimize cost, Hill Climbing on valid states is good.
    # But we might get stuck.
    # Let's do a simple random walk on the space of valid networks.
    # If a move is valid and cost <= current cost, accept.
    # If cost > current cost, accept with small prob (to escape local minima).
    
    T = 1.0 # Temperature
    t_start = time.time()
    
    while time.time() < deadline:
        # Generate mutation
        r = random.random()
        new_net = current_net[:]
        
        if r < 0.4:
            # Delete
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
        elif r < 0.7:
            # Swap endpoints
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                i, j = new_net[idx]
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i == new_j:
                    continue
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                new_net[idx] = (new_i, new_j)
        elif r < 0.85:
            # Insert
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i == new_j:
                continue
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            idx = random.randint(0, len(new_net))
            new_net.insert(idx, (new_i, new_j))
        else:
            # Move
            if len(new_net) > 1:
                idx1 = random.randint(0, len(new_net) - 1)
                idx2 = random.randint(0, len(new_net) - 1)
                if idx1 != idx2:
                    comp = new_net.pop(idx1)
                    new_net.insert(idx2, comp)
                else:
                    continue

        if is_valid(new_net):
            new_cost = len(new_net)
            curr_cost = len(current_net)
            
            if new_cost <= curr_cost:
                current_net = new_net
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_net = new_net[:]
            else:
                # Accept worse solution with probability
                # delta = new_cost - curr_cost
                # prob = exp(-delta / T)
                # T decreases over time?
                # Simple: accept with 0.1 probability
                if random.random() < 0.1:
                    current_net = new_net
                    # Do not update best if worse
        else:
            # Invalid, ignore
            pass

    return best_net

# EVOLVE-BLOCK-START
# The above code is the logic. I need to wrap it in the required format.
# The prompt asks for the complete program.
# I will place the helper functions and the construct function inside the block or outside?
# "Keep the function names and signatures of the seed program."
# Seed program has `construct()`.
# I can define helpers outside `construct` if I want, but they need to be defined before `construct` is called?
# Actually, `construct` is called by the evaluator.
# So I should define everything necessary.
# I will put the NumPy setup and helpers in the global scope or inside construct?
# Global scope is better for `INPUTS` precomputation so it's not repeated if construct is called multiple times (though likely called once).
# However, the evaluator might restart the process.
# Precomputing `INPUTS` takes a few ms. It's safe to do it globally.

# Let's re-structure to fit the "EVOLVE-BLOCK" style if required, or just output the whole script.
# The prompt says "Output the COMPLETE program in a single 