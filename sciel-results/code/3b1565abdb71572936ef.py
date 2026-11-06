import time
import random
import math

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55  # Safety margin

# Precompute the initial state for each wire as a Python integer bitmask.
# Bit k of wire i is 1 if input k has a 1 on wire i.
# Input k is an integer from 0 to 2^N - 1.
# For wire i, the bits set are those k where (k >> i) & 1 == 1.
def _get_initial_wire_masks():
    masks = []
    for i in range(N):
        mask = 0
        for k in range(1 << N):
            if (k >> i) & 1:
                mask |= (1 << k)
        masks.append(mask)
    return masks

INITIAL_MASKS = _get_initial_wire_masks()

def _odd_even_transposition(n):
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _simulate(net):
    """
    Simulate the network using Python integer masks.
    Returns the final state of the wires as a list of integers.
    """
    wires = INITIAL_MASKS[:]
    for i, j in net:
        wires[i], wires[j] = wires[i] & wires[j], wires[i] | wires[j]
    return wires

def _is_sorted(wires):
    """
    Check if the wires are sorted for all boolean inputs.
    Wire i must be a subset of Wire i+1.
    """
    for i in range(N - 1):
        if (wires[i] & wires[i+1]) != wires[i]:
            return False
    return True

def _verify_full(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    """
    if len(net) > MAX_COMPARATORS:
        return False
    if len(net) == 0:
        return True
    wires = _simulate(net)
    return _is_sorted(wires)

def _simulate_with_prefixes(net):
    """
    Simulate the network and record the wire state BEFORE each comparator.
    Returns a list of length len(net), where prefix[k] is a tuple of N ints
    representing the wire state just before comparator k is applied.
    """
    L = len(net)
    if L == 0:
        return []
    wires = INITIAL_MASKS[:]
    prefixes = []
    for k in range(L):
        prefixes.append(tuple(wires))
        i, j = net[k]
        wires[i], wires[j] = wires[i] & wires[j], wires[i] | wires[j]
    return prefixes

def _prune_fast(net, deadline):
    """
    Iteratively prune redundant comparators using prefix simulation.
    In each pass, find ALL redundant comparators at once (O(L) per pass),
    remove them, and repeat.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        L = len(net)
        if L == 0:
            break
        prefixes = _simulate_with_prefixes(net)
        # Find all redundant indices
        to_remove = set()
        for k in range(L):
            i, j = net[k]
            w = prefixes[k]
            if (w[i] & w[j]) == w[i]:
                to_remove.add(k)
        if to_remove:
            new_net = [net[k] for k in range(L) if k not in to_remove]
            net = new_net
            changed = True
    return net

def _try_single_removal(net, deadline):
    """
    Try removing each comparator one at a time (in reverse order to preserve
    indices). Return the best shortened network found.
    Uses prefix simulation for O(1) per removal check.
    """
    L = len(net)
    if L == 0:
        return net
    prefixes = _simulate_with_prefixes(net)
    best = net[:]
    best_len = L
    # Try removing from the end first (indices remain valid)
    # Actually, we need to try each removal independently.
    # For each k, check if removing net[k] still yields a sorted network.
    # We can't just check redundancy of net[k] in the original network,
    # because removing it changes the states of subsequent comparators.
    # So we must verify the full network after removal.
    # But we can use a trick: simulate the network with comparator k skipped.
    # This is O(L) per candidate, so O(L^2) total for one pass.
    # For L ~ 80-100, that's ~10000 operations, which is fast.
    
    for k in range(L - 1, -1, -1):
        if time.time() > deadline:
            break
        # Simulate network with comparator k removed
        wires = INITIAL_MASKS[:]
        for idx in range(L):
            if idx == k:
                continue
            i, j = net[idx]
            wires[i], wires[j] = wires[i] & wires[j], wires[i] | wires[j]
        if _is_sorted(wires):
            # This removal works; commit it and restart the scan
            net = net[:k] + net[k+1:]
            L = len(net)
            best = net[:]
            best_len = L
            break  # Restart to re-evaluate
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition(N)
    
    # Phase 1: Fast structural pruning
    if time.time() < deadline - 15:
        net = _prune_fast(net, deadline - 15)
    
    # Phase 1b: Try single removals iteratively
    if time.time() < deadline - 10:
        net = _try_single_removal(net, deadline - 10)
    
    best_net = net[:]
    best_count = len(net)
    
    # Phase 2: Simulated Annealing with faster verification
    if time.time() < deadline - 3:
        T = 1.0
        min_T = 0.01
        cooling = 0.9995
        current_count = len(net)
        moves_since_improve = 0
        max_moves_no_improve = 5000
        
        while time.time() < deadline - 2:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 1.0
                moves_since_improve = 0
            
            # Generate move
            r = random.random()
            if r < 0.5:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.75:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            else:
                # Mutate: change a comparator to a nearby one
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                # Perturb the comparator
                ni = max(0, min(N - 2, i + random.randint(-2, 2)))
                nj = max(ni + 1, min(N - 1, j + random.randint(-2, 2)))
                if nj <= ni:
                    nj = ni + 1
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            
            # Verify
            if not _verify_full(new_net):
                moves_since_improve += 1
                T *= cooling
                if moves_since_improve > max_moves_no_improve:
                    T = max(T, 0.5)
                    moves_since_improve = 0
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                net = new_net
                current_count = new_count
                moves_since_improve = 0
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    # After improvement, try to prune
                    if time.time() < deadline - 3:
                        net = _prune_fast(net, deadline - 3)
                        current_count = len(net)
                        if current_count < best_count:
                            best_net = net[:]
                            best_count = current_count
                    
            T *= cooling
    
    # Final: try to prune the best network one more time
    if time.time() < deadline - 1:
        final_net = _prune_fast(best_net, deadline - 1)
        if len(final_net) < best_count and _verify_full(final_net):
            best_net = final_net
            best_count = len(best_net)
    
    # Final validation
    if not _verify_full(best_net):
        best_net = _odd_even_transposition(N)
        
    return best_net
