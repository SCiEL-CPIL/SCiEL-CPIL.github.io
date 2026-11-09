import time
import random
import sys

N = 18

# EVOLVE-BLOCK-START

# Precompute constants for fast verification using bit-parallelism
# State is a list of N integers. Integer i represents the state of wire i 
# across all 2^N inputs. Bit 'm' in integer i is 1 if wire i is 1 for input m.

def _precompute_targets():
    """
    Precompute the target state for each wire for all 2^N inputs.
    Target_i has bit 'm' set if and only if the sorted output for input 'm' 
    has a 1 on wire i.
    Wire 0 is the minimum, Wire N-1 is the maximum.
    For input m with k ones, sorted output has 0s in positions 0..N-1-k and 1s in N-k..N-1.
    So wire i is 1 iff k >= N - i.
    """
    targets = [0] * N
    # Optimization: We can build these masks more efficiently by grouping by popcount
    # or simply iterating. 2^18 is small enough for a simple loop if optimized.
    # Let's use a slightly faster method: 
    # For each wire i, it is 1 in the sorted output if the number of 1s in input >= N-i.
    # We can precompute popcount for all masks? No, just iterate.
    # 262144 iterations * 18 checks is ~4.7M ops. In Python this might take ~0.5s.
    # Let's try to be smarter.
    
    # Actually, let's just compute it directly.
    for m in range(1 << N):
        k = bin(m).count('1')
        # Wire i is 1 if k >= N - i
        # i.e., N - i <= k  =>  i >= N - k
        # So wires N-k, N-k+1, ..., N-1 are 1.
        start_wire = N - k
        for i in range(start_wire, N):
            targets[i] |= (1 << m)
    return targets

def _precompute_initials():
    """
    Initial state: Wire i has bit m set if the i-th bit of input m is 1.
    """
    initials = [0] * N
    for i in range(N):
        mask = 0
        # Bit i of m is 1.
        # We can construct this mask using bit tricks or simple loop.
        # Simple loop:
        for m in range(1 << N):
            if (m >> i) & 1:
                mask |= (1 << m)
        initials[i] = mask
    return initials

_TARGETS = None
_INITIALS = None

def _init_constants():
    global _TARGETS, _INITIALS
    if _TARGETS is None:
        _TARGETS = _precompute_targets()
        _INITIALS = _precompute_initials()

def _verify_network_bitparallel(comparators):
    """
    Verifies the network using bit-parallel simulation.
    Returns True if the network sorts all inputs.
    """
    if _TARGETS is None:
        _init_constants()
    
    # Copy initials
    states = list(_INITIALS)
    
    for i, j in comparators:
        si = states[i]
        sj = states[j]
        # min goes to i, max goes to j
        # For 0-1 values:
        # min(a,b) = a & b
        # max(a,b) = a | b
        states[i] = si & sj
        states[j] = si | sj
        
    # Check against targets
    for i in range(N):
        if states[i] != _TARGETS[i]:
            return False
    return True

def _count_inversions_bitparallel(states):
    """
    Counts the total number of inversions (unsorted pairs) across all inputs.
    An inversion exists if for some input m, state[i][m] > state[j][m] for i < j.
    Since states are 0/1, state[i][m] > state[j][m] means state[i][m]=1 and state[j][m]=0.
    This corresponds to the bits set in (states[i] & ~states[j]).
    We sum the popcount of these values for all pairs i < j.
    """
    inv = 0
    for i in range(N):
        si = states[i]
        for j in range(i + 1, N):
            sj = states[j]
            # Count bits where si is 1 and sj is 0
            inv += (si & ~sj).bit_count()
    return inv

def _seed_network():
    """Odd-Even Transposition Sort."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _greedy_construction(deadline):
    """
    Constructs a network greedily layer by layer.
    """
    _init_constants()
    states = list(_INITIALS)
    network = []
    
    # We want to reach the target states.
    # We will build layers. In each layer, we pick a set of non-conflicting comparators
    # that reduces the inversion count the most.
    
    # To keep it simple and effective, we'll just pick the single best comparator
    # that reduces inversions, add it, update states, and repeat.
    # This creates a deep, narrow network initially, which we then prune.
    
    # Optimization: Precompute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline:
        best_comp = None
        best_inv = _count_inversions_bitparallel(states)
        
        # If we are already sorted (0 inversions), we can stop? 
        # No, 0 inversions means sorted. But we need to ensure it stays sorted?
        # Actually, if inv is 0, states == targets. We can stop construction.
        if best_inv == 0:
            break
            
        # Find the comparator that reduces inversions the most
        # To save time, we can sample or check all. 
        # N=18, ~153 comparators. Checking all is 153 * (18 bit ops) which is fast.
        
        for i, j in all_comps:
            # Simulate comparator
            si = states[i]
            sj = states[j]
            new_si = si & sj
            new_sj = si | sj
            
            # Calculate change in inversions involving i and j
            # This is complex to calculate incrementally without full recount.
            # Full recount is O(N^2) bit ops. 153 * 153 * 18 ops is ~400k ops. 
            # This is feasible per step if we don't do too many steps.
            
            # Let's just do a quick check: if the comparator changes the state,
            # we need to check if it reduces inversions.
            # Actually, applying a comparator always reduces or keeps inversions non-increasing?
            # No, it reduces inversions for the pair (i,j) if they were inverted.
            # It might increase inversions with other wires? 
            # Actually, sorting networks are monotonic. 
            # Let's just simulate and count inversions for the candidate.
            
            # To speed up, we can just check if the pair (i,j) is "inverted" in some inputs.
            # If (si & ~sj) is 0, the comparator does nothing. Skip.
            if (si & ~sj) == 0:
                continue
                
            # Apply and count
            # We can optimize inversion count by only checking affected wires.
            # But for simplicity and correctness in this structural change, 
            # let's use a heuristic: pick the one that maximizes (si & ~sj).bit_count()
            # This is a proxy for how much "work" this comparator does.
            
            work = (si & ~sj).bit_count()
            if best_comp is None or work > best_inv: # Wait, best_inv is total inversions.
                # We want to maximize work? Or minimize resulting total inversions?
                # Let's stick to minimizing total inversions for robustness.
                # But full recount is expensive.
                # Let's use the heuristic: Maximize immediate inversion removal.
                pass
        
        # Let's refine the greedy step:
        # 1. Identify all "active" comparators (where si & ~sj != 0).
        # 2. Pick the one with the highest "weight" (popcount of si & ~sj).
        # 3. Apply it.
        
        best_comp = None
        max_work = 0
        for i, j in all_comps:
            si = states[i]
            sj = states[j]
            work = (si & ~sj).bit_count()
            if work > max_work:
                max_work = work
                best_comp = (i, j)
                
        if best_comp is None:
            break # No active comparators, should be sorted
            
        # Apply best_comp
        i, j = best_comp
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        network.append((i, j))
        
        # Check if we are done
        if _count_inversions_bitparallel(states) == 0:
            break
            
    return network

def _prune(comps, deadline):
    """
    Aggressively remove redundant comparators.
    """
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                improved = True
                break
    return comps

def construct():
    _init_constants()
    
    # Deadline: 60s limit, keep 5s margin
    deadline = time.time() + 55.0
    
    # Strategy 1: Greedy Construction
    # This might produce a network that is not minimal but is a valid starting point 
    # potentially different from the odd-even seed.
    greedy_net = _greedy_construction(time.time() + 10.0) # Give it 10s
    
    # Prune the greedy network
    if greedy_net:
        greedy_net = _prune(greedy_net, deadline)
    
    # Strategy 2: Seed Network (Odd-Even)
    seed_net = _seed_network()
    seed_net = _prune(seed_net, deadline)
    
    # Compare and pick the best
    if len(greedy_net) < len(seed_net):
        return greedy_net
    else:
        return seed_net

# EVOLVE-BLOCK-END
