"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 18
TOTAL_INPUTS = 1 << N  # 262144
DEADLINE = 50.0  # seconds, safety margin before 90s limit (evaluator kills at 90s)

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires using Bit-Parallel verification.
    Uses Python big integers to represent the state of each wire across all 2^18 inputs.
    min(a,b) is bitwise AND, max(a,b) is bitwise OR.
    Sorted check: (wire[i] & ~wire[i+1]) == 0 for all i.
    """
    start_time = time.time()
    
    def get_remaining_time():
        return DEADLINE - (time.time() - start_time)

    # --- Initialize Bit-Parallel Wires ---
    # wire[i] is a Python int where bit k is 1 if input k has 1 on wire i.
    # Input k is the integer k. Wire i corresponds to bit i of k.
    # So wire[i] has bits set at positions k where (k >> i) & 1 == 1.
    
    def make_mask(i):
        # Create a big int with 1s at positions where bit i is set.
        # Pattern: 2^i zeros, 2^i ones.
        # Block size: 2^(i+1).
        # Value of one block: (2^i - 1) << 2^i.
        
        L = 1 << (i + 1)
        num_blocks = TOTAL_INPUTS // L
        ones_val = (1 << (1 << i)) - 1
        block = ones_val << (1 << i)
        
        # G = 1 + 2^L + 2^(2L) + ... + 2^((num_blocks-1)L)
        # G = (2^(L * num_blocks) - 1) / (2^L - 1)
        # L * num_blocks = TOTAL_INPUTS
        G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        mask = block * G
        return mask

    # Pre-compute masks
    masks = [make_mask(i) for i in range(N)]

    def verify_network(network):
        """Verify network sorts all inputs. Returns True if correct."""
        # Copy the initial state
        wires = masks[:]
        
        for i, j in network:
            # min to i, max to j
            # min is AND, max is OR
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
            
        # Check sorted: wire[i] <= wire[i+1] for all i
        # This means no bit is 1 in wire[i] if it is 0 in wire[i+1].
        # So (wire[i] & ~wire[i+1]) must be 0.
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return False
        return True

    def get_inversion_wires(network):
        """
        Returns the wires array after applying network, or None if sorted.
        If not sorted, returns the wires array so we can inspect inversions.
        """
        wires = masks[:]
        for i, j in network:
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
            
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return wires
        return None

    # --- Start with odd-even transposition network ---
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    # Verify initial network
    assert verify_network(network), "Initial network should be correct"

    best_network = network[:]
    best_size = len(network)
    current = network[:]

    # --- Phase 1: Aggressive Greedy Removal with Randomization ---
    # We alternate between systematic sweep and random probing to escape local minima.
    
    max_greedy_time = 10.0
    greedy_start = time.time()
    
    while (time.time() - greedy_start) < max_greedy_time:
        removed = False
        # Randomly choose between forward sweep, backward sweep, or random probe
        mode = random.choice(['forward', 'backward', 'random'])
        
        if mode == 'forward':
            indices = list(range(len(current)))
        elif mode == 'backward':
            indices = list(range(len(current) - 1, -1, -1))
        else:
            indices = [random.randint(0, len(current) - 1)]
            
        for idx in indices:
            if time.time() - start_time > DEADLINE - 10:
                break
            if idx < 0 or idx >= len(current):
                continue
            candidate = current[:]
            candidate.pop(idx)
            if verify_network(candidate):
                current = candidate
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
                removed = True
                if mode != 'random':
                    break # Restart sweep if we removed one systematically
                
        if not removed and mode == 'random':
            # If random probe failed, try a few more random probes
            for _ in range(5):
                if time.time() - start_time > DEADLINE - 10:
                    break
                idx = random.randint(0, len(current) - 1)
                candidate = current[:]
                candidate.pop(idx)
                if verify_network(candidate):
                    current = candidate
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)
                    break

    # --- Phase 2: Error-Guided Repair and Insertion ---
    # Try to reduce size by removing a comparator, finding the specific inversion it caused,
    # and then trying to fix it with a *different* comparator or by reordering.
    # If we can't fix it with a swap, we accept the removal only if we can repair it.
    # Actually, a better strategy: 
    # 1. Remove a comparator.
    # 2. If invalid, find the inversion (i, j) where wire[i] has 1 and wire[j] has 0 for some input.
    # 3. Try inserting a comparator (i, j) or nearby (i, k) or (k, j) at various positions.
    
    repair_start = time.time()
    max_repair_time = 15.0
    
    while (time.time() - repair_start) < max_repair_time:
        if time.time() - start_time > DEADLINE - 5:
            break
        if len(current) <= 70:
            break
            
        # Random removal
        idx = random.randint(0, len(current) - 1)
        candidate = current[:]
        candidate.pop(idx)
        
        # If it's still valid, accept
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
            continue
            
        # If broken, try to repair by inserting a comparator
        repaired = False
        
        # Compute the state of the broken network to find the error
        wires_err = get_inversion_wires(candidate)
        
        if wires_err is not None:
            # Find all inversions
            inversions = []
            for i in range(N - 1):
                if (wires_err[i] & ~wires_err[i+1]) != 0:
                    inversions.append((i, i+1))
                    
            # We can also look for non-adjacent inversions? 
            # Usually adjacent is enough for local repair.
            # Let's try to fix the first few found inversions.
            
            # Shuffle inversions to try different ones
            random.shuffle(inversions)
            
            for inv in inversions[:3]: # Try up to 3 different inversion fixes
                i, j = inv
                # Candidate comparators: (i, j) and maybe (i, j+1), (i-1, j) if valid
                cands = [(i, j)]
                if i > 0: cands.append((i-1, j))
                if j < N-1: cands.append((i, j+1))
                
                for ci, cj in cands:
                    if ci >= cj: continue
                    # Try inserting at random positions
                    positions = list(range(len(candidate) + 1))
                    random.shuffle(positions)
                    
                    for pos in positions[:10]: # Limit position checks
                        if time.time() - start_time > DEADLINE - 5:
                            break
                        temp = candidate[:]
                        temp.insert(pos, (ci, cj))
                        
                        if verify_network(temp):
                            current = temp
                            repaired = True
                            if len(current) < best_size:
                                best_network = current[:]
                                best_size = len(current)
                            break
                    if repaired:
                        break
                if repaired:
                    break

    # --- Phase 3: Swap Search (Topology Optimization) ---
    # Swapping comparators can change the flow of data and might allow removals.
    swap_start = time.time()
    max_swap_time = 10.0
    
    while (time.time() - swap_start) < max_swap_time:
        if time.time() - start_time > DEADLINE - 5:
            break
        improved = False
        # Try a few random swaps
        for _ in range(10):
            if len(current) < 2: break
            p = random.randint(0, len(current) - 1)
            q = random.randint(0, len(current) - 1)
            if p == q: continue
            
            candidate = current[:]
            candidate[p], candidate[q] = candidate[q], candidate[p]
            
            if verify_network(candidate):
                # Accept swap if it leads to a better "state"? 
                # For now, just accept if valid. It might unlock removals.
                # To avoid cycling, we only accept if it's "different" or just accept and hope removals follow.
                current = candidate
                improved = True
                break
        
        if not improved:
            break

    # --- Phase 4: Final Greedy Polish ---
    # One more pass of systematic removal on the current best.
    final_net = best_network[:]
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 2:
        removed = False
        for idx in range(len(final_net)):
            if time.time() - start_time > DEADLINE - 1:
                break
            candidate = final_net[:]
            candidate.pop(idx)
            if verify_network(candidate):
                final_net = candidate
                removed = True
                break

    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)

    # Ensure we return a valid network
    if not verify_network(best_network):
        # Fallback to seed if something went wrong (should not happen)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    return best_network
# EVOLVE-BLOCK-END
