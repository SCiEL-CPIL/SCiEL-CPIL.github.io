import time
import numpy as np

N = 18
TIME_LIMIT = 85  # Safety margin for 90s limit

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states for fast verification
    # Number of 64-bit blocks needed for 2^18 inputs
    num_blocks = 2**N // 64
    
    # states[w] is an array of num_blocks uint64s
    # Bit b of states[w][k] is the value of wire w for input (k*64 + b)
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Fill states
    # We can do this efficiently by iterating through inputs
    # Or using bit manipulation on the input index
    for k in range(2**N):
        block_idx = k >> 6
        bit_idx = k & 63
        mask = 1 << bit_idx
        for w in range(N):
            if (k >> w) & 1:
                states[w][block_idx] |= mask

    def _fast_verify(comparators):
        """
        Verify network using bit-parallel simulation.
        Returns True if network sorts all inputs.
        """
        # Copy states
        # Using list of arrays copy is fast enough
        current = [s.copy() for s in states]
        
        for i, j in comparators:
            a = current[i]
            b = current[j]
            # min goes to i, max goes to j
            current[i] = a & b
            current[j] = a | b
            
        # Check if sorted: for all w, current[w] <= current[w+1]
        # This means if current[w] has a 1, current[w+1] must have a 1.
        # Violation: current[w] == 1 and current[w+1] == 0.
        # Bitwise: current[w] & ~current[w+1] should be 0 for all blocks.
        for w in range(N - 1):
            # If any block has a violation, return False
            # np.any(arr != 0) is fast
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    # Verify seed
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    # Phase 1: Greedy Removal (Backward Scan)
    # This usually yields significant improvements quickly
    improved = True
    while improved and (time.time() < deadline - 10):
        improved = False
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            trial = current_network[:idx] + current_network[idx+1:]
            if _fast_verify(trial):
                current_network = trial
                improved = True
                break # Restart scan to account for new dependencies

    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    # Phase 2: Local Search with Swaps and Wire Mutations
    # To escape local minima of removal-only search
    current_network = list(best_network)
    current_len = len(current_network)
    
    # If we are already very small, we might not need this, but let's try
    # We will perform random local moves
    
    # Helper to pick a random valid move
    def _try_local_move(net):
        """
        Attempts a random local move: swap, wire change, or removal.
        Returns the new network if the move is valid and improves/maintains size,
        or None if no valid move found in this attempt.
        """
        n = len(net)
        if n == 0:
            return None
            
        move_type = np.random.choice(3, p=[0.5, 0.3, 0.2]) # Remove, Swap, WireChange
        
        if move_type == 0: # Removal
            idx = np.random.randint(n)
            trial = net[:idx] + net[idx+1:]
            if _fast_verify(trial):
                return trial
                
        elif move_type == 1: # Swap
            if n < 2:
                return None
            idx = np.random.randint(n - 1)
            trial = net.copy()
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _fast_verify(trial):
                return trial
                
        else: # Wire Change
            idx = np.random.randint(n)
            i, j = net[idx]
            # Pick a new wire j' != i
            candidates = [x for x in range(N) if x != i]
            new_j = np.random.choice(candidates)
            if new_j == j:
                return None
            trial = net.copy()
            trial[idx] = (i, new_j) if i < new_j else (new_j, i)
            if _fast_verify(trial):
                return trial
    # Note: Wire change might increase size if it was redundant? No, size is count.
    # Wire change keeps size same. We only accept if valid.
    # We should prefer moves that reduce size.
    # So, only accept Wire Change if it allows a subsequent removal?
    # Or just accept any valid move to explore landscape?
    # Since we want to minimize size, we should only accept moves that lead to a state
    # from which we can remove more.
    # Simplest: Only accept moves that reduce size (Removal).
    # But Swaps and Wire Changes don't reduce size directly.
    # However, they might enable future removals.
    # To ensure progress, we can use a "tabu" or just accept if it doesn't increase size (it never does)
    # and we are stuck.
    
    # Let's refine: 
    # 1. Try Removal. If valid, accept.
    # 2. If no removal found in last K tries, try Swap/WireChange.
    
    stuck_counter = 0
    max_stuck = 50
    
    while time.time() < deadline - 5:
        # Try to remove
        found_removal = False
        # Try a few random removals
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            idx = np.random.randint(len(current_network))
            trial = current_network[:idx] + current_network[idx+1:]
            if _fast_verify(trial):
                current_network = trial
                found_removal = True
                stuck_counter = 0
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                break
        
        if found_removal:
            continue
            
        stuck_counter += 1
        
        # If stuck for a while, try a swap or wire change to escape local minimum
        if stuck_counter >= max_stuck:
            # Try a random swap
            if len(current_network) > 1:
                idx = np.random.randint(len(current_network) - 1)
                trial = current_network.copy()
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                if _fast_verify(trial):
                    current_network = trial
                    stuck_counter = 0 # Reset, we moved
            
            # Try a random wire change
            if len(current_network) > 0:
                idx = np.random.randint(len(current_network))
                i, j = current_network[idx]
                candidates = [x for x in range(N) if x != i]
                new_j = np.random.choice(candidates)
                if new_j != j:
                    trial = current_network.copy()
                    trial[idx] = (i, new_j) if i < new_j else (new_j, i)
                    if _fast_verify(trial):
                        current_network = trial
                        stuck_counter = 0 # Reset, we moved

    # Final check
    if len(current_network) < best_len:
        best_network = current_network
        
    return best_network
# EVOLVE-BLOCK-END
