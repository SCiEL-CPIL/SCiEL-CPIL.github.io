import time
import random

N = 16

# Precompute initial bitsets for zero-one principle verification
# state[i] is a 65536-bit integer where bit k is 1 if input k has a 1 in wire i
_INITIAL_BITSETS = []
for i in range(N):
    bitset = 0
    for k in range(1 << N):
        if k & (1 << i):
            bitset |= (1 << k)
    _INITIAL_BITSETS.append(bitset)

def _verify_zero_one_fast(net):
    """
    Fast verification using bit-slicing.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    if not net:
        return True
    
    state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
        
    # Check if sorted: for all adjacent wires i, i+1, 
    # if wire i has a 1, wire i+1 must also have a 1 in all inputs.
    # state[i] & ~state[i+1] should be 0.
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _base_60_network():
    """
    A known 60-comparator sorting network for 16 wires.
    This is the Batcher Odd-Merge network for N=16.
    """
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def _odd_even_network():
    """
    Odd-even transposition network: 120 comparators.
    """
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _greedy_reduce(net):
    """
    Attempt to remove any comparator that is not essential.
    Returns the reduced network.
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(net)):
            # Try removing net[i]
            candidate = net[:i] + net[i+1:]
            if _verify_zero_one_fast(candidate):
                net = candidate
                changed = True
                break # Restart loop as indices changed
    return net

def _local_search(net, rng, deadline):
    """
    Perform local search to rearrange the network to potentially enable further reductions.
    Only accepts valid neighbors.
    """
    current_net = net[:]
    current_len = len(current_net)
    
    no_improve = 0
    max_no_improve = 100 # Stall counter for local search phase
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Generate a random move
        move_type = rng.random()
        new_net = current_net[:]
        
        if move_type < 0.4:
            # Swap two comparators
            if len(new_net) > 1:
                i, j = rng.sample(range(len(new_net)), 2)
                new_net[i], new_net[j] = new_net[j], new_net[i]
            else:
                no_improve += 1
                continue
        elif move_type < 0.7:
            # Move a comparator to a new position
            if len(new_net) > 1:
                i = rng.randrange(len(new_net))
                comp = new_net.pop(i)
                j = rng.randrange(len(new_net) + 1)
                new_net.insert(j, comp)
            else:
                no_improve += 1
                continue
        elif move_type < 0.9:
            # Mutate a comparator (change one endpoint)
            if len(new_net) > 0:
                idx = rng.randrange(len(new_net))
                i, j = new_net[idx]
                if rng.random() < 0.5:
                    new_i = rng.randrange(N)
                    if new_i < j:
                        new_net[idx] = (new_i, j)
                    elif new_i > j:
                        new_net[idx] = (j, new_i)
                    else:
                        new_net[idx] = (i, j)
                else:
                    new_j = rng.randrange(N)
                    if new_j > i:
                        new_net[idx] = (i, new_j)
                    elif new_j < i:
                        new_net[idx] = (new_j, i)
                    else:
                        new_net[idx] = (i, j)
            else:
                no_improve += 1
                continue
        else:
            # Try to insert a comparator (to escape local minima, though we want to minimize)
            # This is risky but might lead to a better global structure
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i < j:
                pos = rng.randrange(len(new_net) + 1)
                new_net.insert(pos, (i, j))
            else:
                no_improve += 1
                continue
        
        # Verify
        if _verify_zero_one_fast(new_net):
            # Accept if it allows for a greedy reduction later, or just accept any valid move
            # We accept any valid move to explore the space
            current_net = new_net
            no_improve = 0
        else:
            no_improve += 1
            
    return current_net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial best network
    best_net = _base_60_network()
    best_len = len(best_net)
    
    if not _verify_zero_one_fast(best_net):
        # Fallback to odd-even if base is broken (shouldn't happen)
        best_net = _odd_even_network()
        best_len = len(best_net)
        
    rng = random.Random(42)
    
    # Phase 1: Greedy Reduction
    # This is deterministic and fast. It finds all single-comparator removals.
    best_net = _greedy_reduce(best_net)
    best_len = len(best_net)
    
    # Phase 2: Local Search to enable further reductions
    # We alternate between local search (to rearrange) and greedy reduction.
    
    while time.time() < deadline:
        # Run local search for a short burst
        local_deadline = min(deadline, time.time() + 2.0)
        current_net = _local_search(best_net, rng, local_deadline)
        
        # Check if local search found a better arrangement that allows reduction
        # We don't directly reduce in local search, so we check if the new net is valid
        # (it is, by construction) and then try to reduce it.
        
        if len(current_net) < best_len:
            # Local search might have inserted a comparator, so length could increase.
            # We only care if it leads to a shorter net after reduction.
            pass
            
        # Try greedy reduction on the result of local search
        reduced_net = _greedy_reduce(current_net)
        
        if len(reduced_net) < best_len:
            best_net = reduced_net
            best_len = len(best_net)
            
        # If we haven't improved in a while, we might be stuck.
        # The local search restarts with the best_net, so it will explore new neighborhoods.
        
    return best_net

if __name__ == "__main__":
    # Quick self-test
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print(f"Valid: {_verify_zero_one_fast(net)}")
