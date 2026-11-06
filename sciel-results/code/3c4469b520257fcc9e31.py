import time
import random

N = 17

# Precompute initial state bitmasks
INITIAL_STATE = [0] * N
for i in range(N):
    mask = 0
    bit_val = 1 << i
    for k in range(1 << N):
        if k & bit_val:
            mask |= (1 << k)
    INITIAL_STATE[i] = mask

def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1
    
    state = list(INITIAL_STATE)
    n = N
    
    for i, j in network:
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj
        
    for i in range(n - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _local_search(network, deadline):
    """
    Greedy deletion local search.
    Returns the minimized network.
    """
    current = list(network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Try removing this comparator
            test_network = current[:idx] + current[idx+1:]
            
            if _is_valid(test_network):
                current = test_network
                improved = True
                break # Restart loop with new list
                
    return current

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Initial Network
    network = _construct_odd_even_transposition()
    
    # 2. Initial Local Search
    best_network = _local_search(network, deadline)
    best_len = len(best_network)
    current_best = list(best_network)
    
    iterations_since_improvement = 0
    
    # 3. Iterated Local Search
    while time.time() < deadline - 2.0:
        # Perturbation
        trial = list(current_best)
        
        # Stronger perturbation if stuck
        if iterations_since_improvement > 10:
            # Add 3-5 random comparators
            num_adds = random.randint(3, 5)
            for _ in range(num_adds):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(trial))
                trial.insert(pos, (i, j))
            # Shuffle to mix them
            random.shuffle(trial)
            # We don't check validity here, we rely on local search? 
            # No, local search requires valid start.
            # So we must ensure trial is valid.
            # Adding comparators to a valid network keeps it valid.
            # So trial is valid.
        else:
            move_type = random.random()
            if move_type < 0.4:
                # Swap
                if len(trial) >= 2:
                    i, j = random.sample(range(len(trial)), 2)
                    trial[i], trial[j] = trial[j], trial[i]
                    if not _is_valid(trial):
                        continue
            elif move_type < 0.8:
                # Move
                if len(trial) >= 2:
                    i = random.randint(0, len(trial) - 1)
                    j = random.randint(0, len(trial) - 1)
                    if i != j:
                        c = trial.pop(i)
                        trial.insert(j, c)
                        if not _is_valid(trial):
                            continue
            else:
                # Add one
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(trial))
                trial.insert(pos, (i, j))
                # Valid by construction

        # Local Search
        sub_deadline = min(deadline - 2.0, time.time() + 8.0)
        new_network = _local_search(trial, sub_deadline)
        
        if len(new_network) < best_len:
            best_len = len(new_network)
            best_network = list(new_network)
            current_best = list(new_network)
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1
            
    return best_network
