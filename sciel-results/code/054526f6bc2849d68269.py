"""Size-efficient sorting network on 17 wires - batched greedy construction."""
import time
import random
import numpy as np

N = 17
TIME_LIMIT = 80.0  # seconds (safety margin from 90s)


def _seed_network():
    """Odd-even transposition network: N*(N-1)/2 = 136 comparators."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def _simulate_all(network):
    """Simulate network on all 2^N Boolean inputs.
    Returns list of (input_mask, output_state) for all inputs."""
    results = []
    for mask in range(1 << N):
        state = mask
        for i, j in network:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i > bit_j:
                state = (state & ~(1 << i)) | (1 << j)
        results.append((mask, state))
    return results


def _is_sorted(state):
    """Check if a Boolean state is sorted (0...01...1)."""
    if state == 0 or state == (1 << N) - 1:
        return True
    # Count trailing zeros
    tz = 0
    tmp = state
    while (tmp & 1) == 0:
        tz += 1
        tmp >>= 1
        if tz >= N:
            break
    expected = ((1 << N) - 1) ^ ((1 << tz) - 1)
    return state == expected


def _get_unsorted_indices(results):
    """Return list of indices into results where the output is unsorted."""
    unsorted = []
    for idx, (_, state) in enumerate(results):
        if not _is_sorted(state):
            unsorted.append(idx)
    return unsorted


def _find_best_comparator(results, unsorted_indices, network):
    """Find the comparator that, when added to network, fixes the most unsorted inputs.
    Returns (i, j, count) or None."""
    best_count = 0
    best_pair = None
    
    # For each candidate comparator, count how many unsorted inputs it would fix
    # A comparator (i,j) fixes input mask if: after applying network, the state has
    # bit_i=1, bit_j=0, AND after swapping, the result is sorted.
    
    # Pre-compute the current output states for unsorted inputs
    unsorted_states = [results[idx][1] for idx in unsorted_indices]
    unsorted_masks = [results[idx][0] for idx in unsorted_indices]
    
    # Try all possible comparators
    for i in range(N):
        for j in range(i + 1, N):
            count = 0
            for state in unsorted_states:
                bit_i = (state >> i) & 1
                bit_j = (state >> j) & 1
                if bit_i > bit_j:
                    new_state = (state & ~(1 << i)) | (1 << j)
                    if _is_sorted(new_state):
                        count += 1
            if count > best_count:
                best_count = count
                best_pair = (i, j)
    
    if best_pair is None or best_count == 0:
        return None
    return best_pair[0], best_pair[1], best_count


def _construct_batched():
    """Batched greedy construction: iteratively add comparators to fix unsorted inputs."""
    # Start with empty network and build up
    # Actually, start with seed and remove, or start empty and add?
    # Let's start with the seed and try to reduce via batched removal.
    
    # Alternative: start with a minimal valid network and grow.
    # Better: start with seed, identify which comparators are "critical", remove non-critical ones in batches.
    
    # Strategy: 
    # 1. Start with seed network
    # 2. Simulate all inputs
    # 3. For each unsorted input, determine which comparators in the network are "responsible" for fixing it
    # 4. Remove comparators that are not needed for any input
    
    # Actually, let's use a different approach:
    # Start with seed. Try removing groups of comparators simultaneously.
    
    # Even better: use the "add from scratch" approach.
    # Start with empty network. Repeatedly find the comparator that fixes the most
    # currently-unsorted inputs, add it, repeat until all sorted.
    
    network = []
    deadline = time.time() + TIME_LIMIT * 0.7  # Leave time for verification
    
    while time.time() < deadline:
        results = _simulate_all(network)
        unsorted = _get_unsorted_indices(results)
        
        if not unsorted:
            break
        
        # Find the best comparator to add
        best = _find_best_comparator(results, unsorted, network)
        if best is None:
            # No single comparator can fix any unsorted input - shouldn't happen with valid seed
            break
        
        i, j, count = best
        network.append((i, j))
    
    # Verify
    if network:
        results = _simulate_all(network)
        unsorted = _get_unsorted_indices(results)
        if not unsorted:
            return network
    
    return None


def _construct_greedy_removal():
    """Greedy removal: iteratively remove the comparator whose removal breaks the fewest inputs."""
    network = _seed_network()
    deadline = time.time() + TIME_LIMIT * 0.7
    
    while time.time() < deadline:
        results = _simulate_all(network)
        unsorted = _get_unsorted_indices(results)
        
        if not unsorted:
            # All sorted, try to remove more
            pass
        
        # Try removing each comparator and see how many become unsorted
        best_remove_idx = -1
        best_bad_count = float('inf')
        
        for idx in range(len(network)):
            candidate = network[:idx] + network[idx+1:]
            if not candidate:
                continue
            
            cand_results = _simulate_all(candidate)
            cand_unsorted = _get_unsorted_indices(cand_results)
            
            if len(cand_unsorted) < best_bad_count:
                best_bad_count = len(cand_unsorted)
                best_remove_idx = idx
        
        if best_remove_idx >= 0 and best_bad_count <= len(unsorted):
            # Removing this comparator doesn't make things worse (or improves)
            network = network[:best_remove_idx] + network[best_remove_idx+1:]
            
            if best_bad_count == 0:
                # Still fully sorted, continue removing
                continue
            else:
                # This made things worse, put it back
                break
        else:
            break
    
    # Verify final network
    results = _simulate_all(network)
    if not _get_unsorted_indices(results):
        return network
    return None


def _verify_network_fast(network):
    """Verify network sorts all 2^N Boolean inputs."""
    for mask in range(1 << N):
        state = mask
        for i, j in network:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i > bit_j:
                state = (state & ~(1 << i)) | (1 << j)
        if not _is_sorted(state):
            return False
    return True


def _local_search_refine(network, time_budget):
    """Refine a valid network via random local search (swap, insert, delete)."""
    best_network = network[:]
    best_len = len(network)
    start = time.time()
    deadline = start + time_budget
    
    # Pre-compute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline:
        # Try a random mutation
        mutation_type = random.choice(['delete', 'swap', 'insert'])
        
        if mutation_type == 'delete' and len(best_network) > 0:
            idx = random.randint(0, len(best_network) - 1)
            candidate = best_network[:idx] + best_network[idx+1:]
        elif mutation_type == 'swap' and len(best_network) >= 2:
            idx1 = random.randint(0, len(best_network) - 1)
            idx2 = random.randint(0, len(best_network) - 1)
            if idx1 == idx2:
                continue
            candidate = best_network[:]
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        elif mutation_type == 'insert':
            new_comp = random.choice(all_pairs)
            idx = random.randint(0, len(best_network))
            candidate = best_network[:idx] + [new_comp] + best_network[idx:]
        else:
            continue
        
        # Verify candidate
        if _verify_network_fast(candidate):
            if len(candidate) < best_len:
                best_network = candidate
                best_len = len(candidate)
    
    return best_network


def construct():
    """Construct a sorting network for N=17 wires."""
    start_time = time.time()
    
    # Strategy 1: Batched greedy construction from scratch
    batched_net = _construct_batched()
    if batched_net and _verify_network_fast(batched_net):
        best = batched_net
    else:
        # Fallback to seed
        best = _seed_network()
    
    # Strategy 2: Greedy removal from seed
    greedy_net = _construct_greedy_removal()
    if greedy_net and _verify_network_fast(greedy_net):
        if len(greedy_net) < len(best):
            best = greedy_net
    
    # Strategy 3: Local search refinement on the best found
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 5:
        refined = _local_search_refine(best, remaining_time - 3)
        if _verify_network_fast(refined):
            if len(refined) < len(best):
                best = refined
    
    # Final verification
    if not _verify_network_fast(best):
        # Return seed as fallback
        return _seed_network()
    
    return best
