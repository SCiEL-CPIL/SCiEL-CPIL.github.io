import time
import random
import numpy as np

N = 17
MAX_TIME = 90.0  # wall-clock budget
SAFETY_MARGIN = 10.0  # seconds to leave before deadline


def _apply_comparator(state, i, j):
    """Apply a compare-exchange to a state vector. Writes min to i, max to j."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state


def _check_all_boolean(network):
    """Check if the network sorts all 2^17 boolean inputs using zero-one principle."""
    for mask in range(1 << N):
        state = np.array([(mask >> k) & 1 for k in range(N)], dtype=np.int8)
        for i, j in network:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        # Check if sorted (non-decreasing)
        if np.any(state[1:] < state[:-1]):
            return False
    return True


def _check_random(network, num_trials=2000):
    """Quick check with random inputs."""
    for _ in range(num_trials):
        state = np.random.randint(0, 1000, size=N)
        state_orig = state.copy()
        for i, j in network:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        if np.any(state[1:] < state[:-1]):
            return False
    return True


def _check_sorted_input(network):
    """Check that the network doesn't break already-sorted inputs."""
    for trial in range(100):
        state = np.sort(np.random.randint(0, 10000, size=N))
        for i, j in network:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        if np.any(state[1:] < state[:-1]):
            return False
    return True


def _odd_even_transposition():
    """Standard odd-even transposition sorting network for N wires."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network


def _bitonic_merge_down(a, b):
    """Bitonic merge of two sorted halves (ascending) into one ascending sequence."""
    network = []
    # Standard bitonic merge for 2*n wires
    n = len(a)
    for k in range(n - 1, -1, -1):
        for i in range(k):
            network.append((i, i + 2 * (n - k)))
            network.append((i + 2 * (n - k), i))
    # This is getting complex; let me use a simpler approach
    return network


def _build_bitonic_network(n):
    """Build a bitonic sorting network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    
    network = []
    
    def build_bitonic(wires, ascending=True):
        """Build bitonic network for given wire indices."""
        if len(wires) <= 1:
            return []
        if len(wires) == 2:
            i, j = wires[0], wires[1]
            if ascending:
                return [(min(i,j), max(i,j))]
            else:
                return [(min(i,j), max(i,j))]
        
        half = len(wires) // 2
        left = wires[:half]
        right = wires[half:]
        
        # Sort left ascending, right descending (or vice versa)
        if ascending:
            left_net = build_bitonic(left, True)
            right_net = build_bitonic(right, False)
        else:
            left_net = build_bitonic(left, False)
            right_net = build_bitonic(right, True)
        
        result = left_net + right_net
        
        # Merge phase: compare elements symmetrically
        for k in range(half - 1, -1, -1):
            for i in range(k + 1):
                li = left[i]
                rj = right[half - 1 - i]
                if ascending:
                    result.append((min(li, rj), max(li, rj)))
                else:
                    result.append((min(li, rj), max(li, rj)))
        
        return result
    
    return build_bitonic(list(range(n)), True)


def _search_improvements(seed_network, time_limit):
    """Try to improve the seed network by removing redundant comparators."""
    start_time = time.time()
    best_network = list(seed_network)
    best_count = len(best_network)
    
    # First verify the seed is correct
    if not _check_all_boolean(best_network):
        print("Seed network is not correct!")
        return best_network
    
    # Try greedy removal: for each comparator, check if removing it still gives a valid network
    for pass_num in range(50):  # Multiple passes of removal
        if time.time() - start_time > time_limit:
            break
        
        removed_any = False
        # Try to remove each comparator
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            
            # Skip if we've already spent too much time
            if len(best_network) <= 63:  # Can't beat the lower bound
                break
            
            # Try removing this comparator
            candidate = best_network[:idx] + best_network[idx+1:]
            
            # Quick check with random inputs first
            if not _check_random(candidate, 500):
                continue
            
            # Full boolean check
            if _check_all_boolean(candidate):
                best_network = candidate
                removed_any = True
                print(f"  Removed comparator at index {idx}, now {len(best_network)} comparators")
        
        if not removed_any:
            # Try a different order
            for idx in range(len(best_network)):
                if time.time() - start_time > time_limit:
                    break
                if len(best_network) <= 63:
                    break
                
                candidate = best_network[:idx] + best_network[idx+1:]
                
                if not _check_random(candidate, 500):
                    continue
                
                if _check_all_boolean(candidate):
                    best_network = candidate
                    print(f"  Removed comparator at index {idx} (fwd pass), now {len(best_network)} comparators")
                    break  # Restart forward scan
        
        if not removed_any:
            break
    
    return best_network


def _hill_climbing(seed_network, time_limit):
    """Hill climbing: try swapping, moving, or removing comparators."""
    start_time = time.time()
    best_network = list(seed_network)
    best_count = len(best_network)
    
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        if len(best_network) <= 63:
            break
        
        # Try various local moves
        moves = []
        
        # 1. Try removing a comparator
        for idx in range(len(best_network)):
            moves.append(('remove', idx, None))
        
        # 2. Try swapping adjacent comparators (if they're on different wires)
        for idx in range(len(best_network) - 1):
            i1, j1 = best_network[idx]
            i2, j2 = best_network[idx+1]
            if i1 != i2 and j1 != j2:  # Non-overlapping wires
                moves.append(('swap', idx, idx+1))
        
        # 3. Try moving a comparator slightly earlier or later
        for idx in range(1, len(best_network)):
            i_cur, j_cur = best_network[idx]
            i_prev, j_prev = best_network[idx-1]
            if i_cur != i_prev and j_cur != j_prev:
                moves.append(('move_up', idx, None))
        
        random.shuffle(moves)
        
        made_progress = False
        for move in moves:
            if time.time() - start_time > time_limit:
                break
            if len(best_network) <= 63:
                break
            
            candidate = list(best_network)
            
            if move[0] == 'remove':
                idx = move[1]
                candidate = best_network[:idx] + best_network[idx+1:]
            elif move[0] == 'swap':
                i, j = move[1], move[2]
                candidate[i], candidate[j] = candidate[j], candidate[i]
            elif move[0] == 'move_up':
                idx = move[1]
                candidate = best_network[:idx-1] + [best_network[idx]] + \
                           best_network[idx-1:idx] + best_network[idx+1:]
            
            # Quick validity check
            if not _check_random(candidate, 300):
                continue
            
            if _check_all_boolean(candidate):
                if move[0] == 'remove' and len(candidate) < len(best_network):
                    best_network = candidate
                    made_progress = True
                    print(f"  Iter {iteration}: removed comparator, now {len(best_network)}")
                    break
                elif move[0] in ('swap', 'move_up'):
                    # Only accept if it enables future removals; skip for now
                    pass
        
        if not made_progress and iteration % 100 == 0:
            # Try more aggressive random perturbation
            if len(best_network) > 70:
                # Randomly shuffle some comparators
                candidate = list(best_network)
                for _ in range(3):
                    idx1 = random.randint(0, len(candidate) - 2)
                    idx2 = idx1 + 1
                    i1, j1 = candidate[idx1]
                    i2, j2 = candidate[idx2]
                    if i1 != i2 and j1 != j2:
                        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                
                if _check_random(candidate, 300) and _check_all_boolean(candidate):
                    best_network = candidate
                    print(f"  Iter {iteration}: perturbed, now {len(best_network)}")
    
    return best_network


def construct():
    """Construct a sorting network for N=17 wires with minimal comparators."""
    start_time = time.time()
    deadline = start_time + MAX_TIME - SAFETY_MARGIN
    
    # Start with odd-even transposition (136 comparators)
    network = _odd_even_transposition()
    print(f"Seed network: {len(network)} comparators")
    
    # Verify seed
    if not _check_all_boolean(network):
        print("ERROR: Seed network failed validation")
        return network
    
    # Phase 1: Greedy removal
    if time.time() < deadline - 20:
        network = _search_improvements(network, min(20, deadline - time.time()))
        print(f"After greedy removal: {len(network)} comparators")
    
    # Phase 2: Hill climbing with swaps and moves
    if time.time() < deadline - 5:
        remaining = min(25, deadline - time.time())
        network = _hill_climbing(network, remaining)
        print(f"After hill climbing: {len(network)} comparators")
    
    # Final validation
    if time.time() < deadline:
        if not _check_all_boolean(network):
            print("WARNING: Final network failed validation, returning seed")
            return _odd_even_transposition()
    
    print(f"Final network: {len(network)} comparators")
    return network
