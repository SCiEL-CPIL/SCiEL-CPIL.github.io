"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
SEED_COMPARATORS = 153
DEADLINE_S = 85.0  # Safety margin before 90s kill

# EVOLVE-BLOCK-START

def _odd_even_transposition():
    """Generate the standard odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _apply_network(network, input_vec):
    """Apply a compare-exchange network to a boolean input vector.
    Returns the output vector."""
    out = input_vec.copy()
    for (i, j) in network:
        a, b = out[i], out[j]
        if a > b:
            out[i], out[j] = b, a
    return out


def _is_correct(network):
    """Check if the network sorts all 2^N boolean inputs."""
    n_comps = len(network)
    # Check all 2^N boolean inputs
    for mask in range(1 << N):
        inp = np.zeros(N, dtype=np.uint8)
        for i in range(N):
            inp[i] = (mask >> i) & 1
        out = _apply_network(network, inp)
        # Output should be sorted: all 0s then all 1s
        # i.e., out should be non-increasing (0s at high indices? No, min to i, max to j)
        # Actually, after sorting, the array should be sorted in non-decreasing order
        # Wait: min goes to wire i, max to wire j. So the network sorts in non-decreasing order.
        # For boolean inputs, sorted means all 0s first, then all 1s.
        # So out[0] <= out[1] <= ... <= out[N-1]
        for i in range(N - 1):
            if out[i] > out[i + 1]:
                return False
    return True


def _depth(network):
    """Compute the parallel depth of the network."""
    wire_depth = [0] * N
    for (i, j) in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth)


def _bitonic_sort_network():
    """Generate Batcher's bitonic sorting network for N=18 (pad to 32? No, use recursive).
    For non-power-of-2, we can pad to next power of 2 and then remove dummy wires.
    Actually, let's just use a known construction. For 18 wires, we can use:
    - Split into two halves of 9, sort each, merge.
    - But 9 is not power of 2 either.
    
    Let's use a simpler approach: generate a valid network using odd-even transposition
    as the base, which is known to be correct.
    """
    return _odd_even_transposition()


def _remove_comparator(network, idx):
    """Remove comparator at index idx and return new network."""
    new_net = network[:idx] + network[idx+1:]
    return new_net


def _add_comparator(network, i, j):
    """Add comparator (i,j) to the network."""
    if i > j:
        i, j = j, i
    new_net = network + [(i, j)]
    return new_net


def _swap_comparators(network, idx1, idx2):
    """Swap two comparators in the network."""
    new_net = network.copy()
    new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
    return new_net


def _mutate_network(network, rng):
    """Apply a random mutation to the network."""
    new_net = network.copy()
    mutation_type = rng.random()
    
    if mutation_type < 0.3 and len(new_net) > 1:
        # Remove a random comparator
        idx = rng.randint(0, len(new_net) - 1)
        new_net = _remove_comparator(new_net, idx)
    elif mutation_type < 0.5:
        # Add a random comparator
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        new_net = _add_comparator(new_net, i, j)
    elif mutation_type < 0.7:
        # Swap two comparators
        if len(new_net) >= 2:
            idx1 = rng.randint(0, len(new_net) - 1)
            idx2 = rng.randint(0, len(new_net) - 1)
            if idx1 != idx2:
                new_net = _swap_comparators(new_net, idx1, idx2)
    else:
        # Change a comparator's endpoints
        if len(new_net) > 0:
            idx = rng.randint(0, len(new_net) - 1)
            i, j = new_net[idx]
            # Slightly perturb one endpoint
            if rng.random() < 0.5:
                i = max(0, min(N - 2, i + rng.randint(-1, 1)))
                j = min(N - 1, max(i + 1, j))
            else:
                j = min(N - 1, max(i + 1, j + rng.randint(-1, 1)))
            new_net[idx] = (i, j)
    
    return new_net


def _crossover(net1, net2, rng):
    """Crossover: take prefix from one and suffix from another."""
    if not net1 or not net2:
        return net1 if net1 else net2
    cut1 = rng.randint(0, len(net1))
    cut2 = rng.randint(0, len(net2))
    child = net1[:cut1] + net2[cut2:]
    return child


def _greedy_removal(network, rng):
    """Try to remove comparators greedily, checking correctness."""
    n = len(network)
    # Try removing from the end first (end-biased), then random
    order = list(range(n - 1, -1, -1))
    rng.shuffle(order)
    # Put end indices first
    end_indices = [i for i in range(n // 2, n)]
    mid_indices = [i for i in range(n // 2)]
    rng.shuffle(end_indices)
    rng.shuffle(mid_indices)
    order = end_indices + mid_indices
    
    current_net = network.copy()
    removed = 0
    for idx in order:
        if idx >= len(current_net):
            continue
        candidate = _remove_comparator(current_net, idx)
        if _is_correct(candidate):
            current_net = candidate
            removed += 1
        else:
            # Try a few more random removals
            for _ in range(3):
                ridx = rng.randint(0, len(current_net) - 1)
                candidate = _remove_comparator(current_net, ridx)
                if _is_correct(candidate):
                    current_net = candidate
                    removed += 1
                    break
    return current_net


def _targeted_replacement(network, rng):
    """Try replacing a comparator with a nearby one."""
    current_net = network.copy()
    improved = False
    for idx in range(len(current_net)):
        i, j = current_net[idx]
        # Try replacing with a comparator involving a nearby wire
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni = i + di
                nj = j + dj
                if ni < 0 or ni >= N - 1 or nj < 0 or nj >= N:
                    continue
                if ni >= nj:
                    continue
                if (ni, nj) == (i, j):
                    continue
                candidate = current_net.copy()
                candidate[idx] = (ni, nj)
                if _is_correct(candidate):
                    current_net = candidate
                    improved = True
                    break
            if improved:
                break
        if improved:
            break
    return current_net


def construct():
    """Construct a sorting network for N=18 wires with as few comparators as possible."""
    start_time = time.time()
    rng = random.Random(42)
    
    # Start with odd-even transposition network
    best_net = _odd_even_transposition()
    best_count = len(best_net)
    
    # Verify it's correct (it should be)
    if not _is_correct(best_net):
        # Fallback: return the standard network
        return best_net
    
    # Try greedy removal first
    deadline_check = time.time() - start_time
    if deadline_check < DEADLINE_S - 10:
        improved = _greedy_removal(best_net, rng)
        if len(improved) < best_count:
            best_net = improved
            best_count = len(best_net)
    
    # Try targeted replacements and more removals
    deadline_check = time.time() - start_time
    if deadline_check < DEADLINE_S - 5:
        for _ in range(50):
            if time.time() - start_time > DEADLINE_S - 5:
                break
            # Try targeted replacement
            candidate = _targeted_replacement(best_net, rng)
            if len(candidate) < best_count:
                best_net = candidate
                best_count = len(candidate)
            # Try greedy removal on the current best
            candidate = _greedy_removal(best_net, rng)
            if len(candidate) < best_count:
                best_net = candidate
                best_count = len(candidate)
    
    # Local search with random mutations
    deadline_check = time.time() - start_time
    if deadline_check < DEADLINE_S - 2:
        current_net = best_net.copy()
        no_improve_count = 0
        while time.time() - start_time < DEADLINE_S - 2:
            candidate = _mutate_network(current_net, rng)
            if len(candidate) < len(current_net) and _is_correct(candidate):
                current_net = candidate
                if len(current_net) < best_count:
                    best_net = current_net.copy()
                    best_count = len(best_net)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            elif _is_correct(candidate):
                current_net = candidate
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > 200:
                # Reset to best
                current_net = best_net.copy()
                no_improve_count = 0
                # Try a different mutation strategy
                for _ in range(10):
                    candidate = _mutate_network(current_net, rng)
                    if _is_correct(candidate) and len(candidate) < best_count:
                        best_net = candidate
                        best_count = len(best_net)
                        current_net = candidate
                        break
    
    return best_net

# EVOLVE-BLOCK-END
