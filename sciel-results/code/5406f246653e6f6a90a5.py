import time
import numpy as np
import random

N = 18
TIME_LIMIT = 55  # Safety margin for 90s wall-clock limit

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _bit_parallel_init():
    """Initialize bit-parallel states for all 2^N Boolean inputs.
    
    Returns a list of N arrays, each of shape (num_blocks,), dtype uint64.
    states[w][block] bit b is 1 if input (block*64+b) has 1 on wire w.
    """
    num_blocks = 2**N // 64
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    for k in range(2**N):
        block_idx = k >> 6
        bit_idx = k & 63
        mask = 1 << bit_idx
        for w in range(N):
            if (k >> w) & 1:
                states[w][block_idx] |= mask
    return states

def _verify(states, comparators):
    """Verify that the comparator network sorts all Boolean inputs."""
    current = [s.copy() for s in states]
    for i, j in comparators:
        a = current[i]
        b = current[j]
        current[i] = a & b
        current[j] = a | b
    
    # Check sorted: for all w, if wire w is 1 then wire w+1 must be 1
    for w in range(N - 1):
        if np.any(current[w] & ~current[w + 1]):
            return False
    return True

def _greedy_reduce(states, net, deadline):
    """Remove redundant comparators by forward removal."""
    cur_net = list(net)
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        idx = 0
        while idx < len(cur_net):
            if time.time() > deadline - 2:
                return cur_net
            trial = cur_net[:idx] + cur_net[idx+1:]
            if _verify(states, trial):
                cur_net = trial
                changed = True
            else:
                idx += 1
    return cur_net

def _compute_residual(states, net):
    """Compute which inputs are not sorted by the given network.
    
    Returns a list of N uint64 arrays where bit b in block i indicates
    input i*64+b is not sorted (has a violation somewhere).
    """
    current = [s.copy() for s in states]
    for i, j in net:
        a = current[i]
        b = current[j]
        current[i] = a & b
        current[j] = a | b
    
    # Find violations: wire w is 1 but wire w+1 is 0
    # An input is "unsorted" if any such violation exists
    unsorted = np.zeros(len(current[0]), dtype=np.uint64)
    for w in range(N - 1):
        violations = current[w] & ~current[w + 1]
        unsorted |= violations
    return unsorted

def _construct_from_residual(states, residual, deadline):
    """Given a set of unsorted inputs (bit-parallel), build a network to fix them.
    
    Uses a greedy approach: find comparators that fix the most unsorted inputs.
    """
    # We need to build a network that sorts all inputs currently in residual.
    # Start with empty network, add comparators that help.
    net = []
    remaining = residual.copy()
    
    # Precompute for each possible comparator (i,j) how many unsorted inputs it would fix
    # if added to the current network. But this is expensive to recompute.
    # Instead, use a simpler heuristic: try all comparators and pick the one that
    # reduces the number of unsorted inputs the most.
    
    max_iter = 200  # Safety
    iter_count = 0
    
    while time.time() < deadline - 3 and iter_count < max_iter:
        iter_count += 1
        if len(remaining) == 0 or np.all(remaining == 0):
            break
        
        # Find the best comparator to add
        best_cmp = None
        best_reduction = 0
        
        for i in range(N):
            for j in range(i + 1, N):
                # Simulate adding (i,j) to net and see how many unsorted inputs are fixed
                # We only care about the inputs in remaining
                # Apply net + (i,j) to the inputs in remaining
                
                # This is expensive. Let's use a cheaper heuristic.
                # For each wire w, look at the pattern of bits in remaining.
                # A comparator (i,j) helps if it can fix some inversions.
                
                # Simpler: just try a random subset of comparators
                pass
        
        # For now, use a simpler approach: add comparators from a template
        # based on the pattern of unsorted inputs.
        
        # Alternative: use the full odd-even transposition as a fallback
        # and greedy reduce it, but only on the subset of unsorted inputs.
        
        # Let's try: for each pair (i,j), compute how many unsorted inputs
        # would become sorted if we applied (i,j) after the current net.
        # We can do this by simulating only on the unsorted inputs.
        
        # To keep it fast, limit to a few candidate comparators per iteration.
        candidates = []
        for i in range(N):
            for j in range(i + 1, N):
                candidates.append((i, j))
        
        random.shuffle(candidates)
        candidates = candidates[:50]  # Limit to 50 candidates
        
        for i, j in candidates:
            # Simulate net + (i,j) on all inputs, check how many in remaining are now sorted
            current = [s.copy() for s in states]
            for a, b in net:
                current[a] = current[a] & current[b]
                current[b] = current[a] | current[b]  # Bug: should use original values
            # Fix the above
            current = [s.copy() for s in states]
            for a, b in net:
                va = current[a].copy()
                vb = current[b].copy()
                current[a] = va & vb
                current[b] = va | vb
            va = current[i].copy()
            vb = current[j].copy()
            current[i] = va & vb
            current[j] = va | vb
            
            # Check which of the remaining unsorted inputs are now sorted
            new_unsorted = np.zeros(len(current[0]), dtype=np.uint64)
            for w in range(N - 1):
                new_unsorted |= current[w] & ~current[w + 1]
            
            # Count how many bits in remaining are no longer in new_unsorted
            fixed = remaining & ~new_unsorted
            # Count bits
            # np.bitwise_count is not available, use a trick
            reduction = sum(bin(b).count('1') for b in fixed)
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_cmp = (i, j)
        
        if best_cmp is not None and best_reduction > 0:
            net.append(best_cmp)
            # Update remaining
            current = [s.copy() for s in states]
            for a, b in net:
                va = current[a].copy()
                vb = current[b].copy()
                current[a] = va & vb
                current[b] = va | vb
            new_unsorted = np.zeros(len(current[0]), dtype=np.uint64)
            for w in range(N - 1):
                new_unsorted |= current[w] & ~current[w + 1]
            remaining = new_unsorted
        else:
            break
    
    return net

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states
    states = _bit_parallel_init()
    
    # Initial network: odd-even transposition
    best_network = _odd_even_transposition()
    best_network = _greedy_reduce(states, best_network, deadline)
    best_len = len(best_network)
    
    # Local search with different move types
    current_network = list(best_network)
    stagnation_counter = 0
    max_stagnation = 300
    
    while time.time() < deadline - 5:
        move_type = random.random()
        
        if move_type < 0.35:
            # Mutation: Change a random comparator
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                a, b = random.sample(range(N), 2)
                if a > b: a, b = b, a
                trial = list(current_network)
                trial[idx] = (a, b)
                
        elif move_type < 0.55:
            # Swap: Swap two comparators (not just adjacent)
            if len(current_network) > 1:
                idx1, idx2 = random.sample(range(len(current_network)), 2)
                trial = list(current_network)
                trial[idx1], trial[idx2] = trial[idx2], trial[idx1]
                
        elif move_type < 0.75:
            # Insert: Add a random comparator
            idx = random.randint(0, len(current_network))
            a, b = random.sample(range(N), 2)
            if a > b: a, b = b, a
            trial = current_network[:idx] + [(a, b)] + current_network[idx:]
            
        elif move_type < 0.9:
            # Delete: Remove a random comparator
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                trial = current_network[:idx] + current_network[idx+1:]
        else:
            # Residual-based reconstruction
            residual = _compute_residual(states, current_network)
            if np.any(residual != 0):
                new_net = _construct_from_residual(states, residual, deadline)
                if new_net:
                    trial = list(current_network) + new_net
                else:
                    trial = list(current_network)
            else:
                trial = list(current_network)
        
        # Apply move and reduce
        reduced = _greedy_reduce(states, trial, deadline)
        
        # Update best
        if len(reduced) < best_len:
            best_network = reduced
            best_len = len(reduced)
            current_network = reduced
            stagnation_counter = 0
        elif len(reduced) < len(current_network):
            current_network = reduced
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
            if stagnation_counter >= max_stagnation:
                # Kick: Perturb the best network
                K = random.randint(2, 6)
                current_network = list(best_network)
                for _ in range(K):
                    if not current_network: break
                    idx = random.randint(0, len(current_network) - 1)
                    a, b = random.sample(range(N), 2)
                    if a > b: a, b = b, a
                    current_network[idx] = (a, b)
                
                current_network = _greedy_reduce(states, current_network, deadline)
                stagnation_counter = 0

    return best_network
