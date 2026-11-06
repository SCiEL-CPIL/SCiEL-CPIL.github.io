import time
import random

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    # 3. Local Search with Add-Remove
    # We keep the best network found so far.
    # We operate on `current_network`.
    
    current_network = best_network[:]
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to remove
        # Scan all indices
        for idx in range(len(current_network)):
            if time.time() > deadline:
                break
            
            candidate = current_network[:idx] + current_network[idx+1:]
            
            if is_valid(candidate):
                current_network = candidate
                improved = True
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
                break # Restart scan
        
        if not improved:
            # Phase 2: Mutate
            # Strategy: 
            # 50% Swap/Replace (size constant)
            # 50% Add (size +1) then try to remove multiple
            
            if random.random() < 0.5:
                # Constant size mutation
                if len(current_network) > 1 and random.random() < 0.5:
                    # Swap
                    i = random.randint(0, len(current_network) - 1)
                    j = random.randint(0, len(current_network) - 1)
                    if i != j:
                        candidate = current_network[:]
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        if is_valid(candidate):
                            current_network = candidate
                else:
                    # Replace
                    i = random.randint(0, len(current_network) - 1)
                    new_i = random.randint(0, N - 1)
                    new_j = random.randint(0, N - 1)
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    if new_i != new_j:
                        candidate = current_network[:]
                        candidate[i] = (new_i, new_j)
                        if is_valid(candidate):
                            current_network = candidate
            else:
                # Add a comparator
                # Insert at random position
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i != new_j:
                    pos = random.randint(0, len(current_network))
                    candidate = current_network[:]
                    candidate.insert(pos, (new_i, new_j))
                    
                    if is_valid(candidate):
                        # Now try to remove as many as possible
                        # We don't update best_network yet, we just see if we can get back to a smaller size
                        # Actually, if we added one, we are at size S+1.
                        # If we can remove 2, we get S-1.
                        # So we should run the removal loop on this candidate.
                        # But that's expensive.
                        # Let's just accept the add, and let the main loop handle removals.
                        current_network = candidate
                        # Note: We do NOT update best_network here because size increased.
                        # But we might be in a better position to remove later.
                        
    return best_network
# EVOLVE-BLOCK-END
