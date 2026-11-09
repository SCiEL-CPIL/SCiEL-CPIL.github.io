import time
import random
import sys

N = 18
SEED_COUNT = 153

# EVOLVE-BLOCK-START

# Precompute constants for fast verification using bit-parallelism
# State is a list of N integers. Integer i represents the state of wire i 
# across all 2^N inputs. Bit 'm' in integer i is 1 if wire i is 1 for input m.

def _precompute_targets():
    """
    Precompute the target state for each wire for all 2^N inputs.
    Wire 0 is the minimum, Wire N-1 is the maximum.
    For input m with k ones, sorted output has 0s in positions 0..N-1-k and 1s in N-k..N-1.
    So wire i is 1 iff k >= N - i.
    """
    targets = [0] * N
    for m in range(1 << N):
        k = bin(m).count('1')
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
        for m in range(1 << N):
            if (m >> i) & 1:
                mask |= (1 << m)
        initials[i] = mask
    return initials

_TARGETS = None
_INITIALS = None
_ALL_COMPS = None

def _init_constants():
    global _TARGETS, _INITIALS, _ALL_COMPS
    if _TARGETS is None:
        _TARGETS = _precompute_targets()
        _INITIALS = _precompute_initials()
        _ALL_COMPS = [(i, j) for i in range(N) for j in range(i+1, N)]

def _verify_network_bitparallel(comparators):
    """
    Verifies the network using bit-parallel simulation.
    Returns True if the network sorts all inputs.
    """
    if _TARGETS is None:
        _init_constants()
    
    states = list(_INITIALS)
    
    for i, j in comparators:
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        
    for i in range(N):
        if states[i] != _TARGETS[i]:
            return False
    return True

def _count_inversions_bitparallel(states):
    """
    Counts the total number of inversions across all inputs.
    """
    inv = 0
    for i in range(N):
        si = states[i]
        for j in range(i + 1, N):
            sj = states[j]
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
    Constructs a network greedily by always picking the comparator
    that minimizes the total inversion count.
    Returns the network and whether it's verified to be sorted.
    """
    if _INITIALS is None:
        _init_constants()
    
    states = list(_INITIALS)
    network = []
    
    all_comps = _ALL_COMPS
    
    while time.time() < deadline:
        current_inv = _count_inversions_bitparallel(states)
        
        if current_inv == 0:
            break
        
        best_comp = None
        best_inv = current_inv
        best_delta = 0
        
        for i, j in all_comps:
            si = states[i]
            sj = states[j]
            
            # Quick check: if no inversion on this pair, skip
            if (si & ~sj) == 0:
                continue
            
            new_si = si & sj
            new_sj = si | sj
            
            # Compute delta in inversions for affected pairs only
            delta = 0
            
            # Pair (i,j)
            old_ij = (si & ~sj).bit_count()
            new_ij = (new_si & ~new_sj).bit_count()
            delta += new_ij - old_ij
            
            # Pairs (a,i) for a < i
            for a in range(i):
                sa = states[a]
                old_ai = (sa & ~si).bit_count()
                new_ai = (sa & ~new_si).bit_count()
                delta += new_ai - old_ai
            
            # Pairs (i,b) for b > i, b != j
            for b in range(i + 1, N):
                if b == j:
                    continue
                sb = states[b]
                old_ib = (si & ~sb).bit_count()
                new_ib = (new_si & ~sb).bit_count()
                delta += new_ib - old_ib
            
            # Pairs (a,j) for a < j, a != i
            for a in range(j):
                if a == i:
                    continue
                sa = states[a]
                old_aj = (sa & ~sj).bit_count()
                new_aj = (sa & ~new_sj).bit_count()
                delta += new_aj - old_aj
            
            # Pairs (j,b) for b > j
            for b in range(j + 1, N):
                sb = states[b]
                old_jb = (sj & ~sb).bit_count()
                new_jb = (new_sj & ~sb).bit_count()
                delta += new_jb - old_jb
            
            new_inv = current_inv + delta
            
            if new_inv < best_inv:
                best_inv = new_inv
                best_comp = (i, j)
                best_delta = delta
        
        if best_comp is None:
            break
        
        i, j = best_comp
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        network.append((i, j))
        
        # Check if we are done
        if _count_inversions_bitparallel(states) == 0:
            break
    
    # Verify the network is actually valid
    is_valid = _verify_network_bitparallel(network)
    return network, is_valid

def _prune(comps, deadline):
    """
    Aggressively remove redundant comparators.
    """
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing from the end first (often less critical)
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                improved = True
                break
    return comps

def _random_network(num_comps):
    """Generate a random network with the given number of comparators."""
    comps = []
    for _ in range(num_comps):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comps.append((i, j))
    return comps

def _mutate(comps, deadline):
    """
    Perform random mutations on a valid network.
    Returns a potentially better (shorter) valid network.
    """
    comps = list(comps)
    best = comps
    
    for _ in range(2000):
        if time.time() > deadline:
            break
        
        mutated = list(comps)
        op = random.choice(['swap', 'insert', 'remove'])
        
        if op == 'swap' and len(mutated) >= 2:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        elif op == 'insert':
            pos = random.randint(0, len(mutated))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            mutated.insert(pos, (i, j))
        elif op == 'remove' and len(mutated) > 0:
            pos = random.randint(0, len(mutated) - 1)
            mutated.pop(pos)
        
        if _verify_network_bitparallel(mutated):
            if len(mutated) < len(best):
                best = mutated
                # Try to prune the improved version
                pruned = _prune(best, min(time.time() + 0.5, deadline))
                if len(pruned) < len(best):
                    best = pruned
    
    return best

def construct():
    _init_constants()
    
    # Deadline: 90s limit, keep 5s margin
    deadline = time.time() + 85.0
    
    best_network = None
    best_len = SEED_COUNT
    
    # Strategy 1: Seed Network with pruning
    seed_net = _seed_network()
    seed_pruned = _prune(seed_net, min(time.time() + 10, deadline))
    if len(seed_pruned) < best_len:
        best_len = len(seed_pruned)
        best_network = seed_pruned
    
    # Strategy 2: Greedy Construction with pruning
    if time.time() < deadline:
        greedy_deadline = min(time.time() + 15, deadline - 5)
        greedy_net, greedy_valid = _greedy_construction(greedy_deadline)
        if greedy_valid and greedy_net:
            greedy_pruned = _prune(greedy_net, min(time.time() + 10, deadline))
            if len(greedy_pruned) < best_len:
                best_len = len(greedy_pruned)
                best_network = greedy_pruned
    
    # Strategy 3: Random restarts with mutation
    while time.time() < deadline - 5:
        # Start from a random network or from the best so far
        if random.random() < 0.5 and best_network:
            start_net = list(best_network)
        else:
            start_net = _random_network(random.randint(60, 120))
        
        # Prune first
        start_net = _prune(start_net, min(time.time() + 2, deadline))
        
        if len(start_net) < best_len:
            best_len = len(start_net)
            best_network = start_net
        
        # Mutate
        mutated = _mutate(start_net, min(time.time() + 3, deadline))
        if len(mutated) < best_len:
            best_len = len(mutated)
            best_network = mutated
    
    # Final prune of the best network
    if best_network is None:
        return _seed_network()
    
    final = _prune(best_network, deadline)
    return final

# EVOLVE-BLOCK-END

