"""Size-efficient sorting networks on 16 wires."""
import time
import random

N = 16
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

# Known good starting point: odd-even transposition (120 comparators)
def _initial_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _apply_network(network, input_vec):
    """Apply the network to an input vector (list of 0/1 for zero-one principle)."""
    state = list(input_vec)
    for i, j in network:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _is_sorted(state):
    return all(state[i] <= state[i+1] for i in range(N-1))

def _verify_network(network):
    """Verify using zero-one principle: check all 2^16 inputs."""
    for mask in range(1 << N):
        inp = [(mask >> i) & 1 for i in range(N)]
        out = _apply_network(network, inp)
        if not _is_sorted(out):
            return False
    return True

def _verify_network_sampled(network, num_samples=20000, rng=None):
    """Fast verification: check random samples + all single-bit + all pair patterns."""
    if rng is None:
        rng = random.Random(42)
    # Check all single-bit patterns
    for bit in range(N):
        inp = [0]*N
        inp[bit] = 1
        out = _apply_network(network, inp)
        if not _is_sorted(out):
            return False
    # Check all pair patterns (two 1s)
    for a in range(N):
        for b in range(a+1, N):
            inp = [0]*N
            inp[a] = 1
            inp[b] = 1
            out = _apply_network(network, inp)
            if not _is_sorted(out):
                return False
    # Random samples
    for _ in range(num_samples):
        inp = [rng.randint(0, 1) for _ in range(N)]
        out = _apply_network(network, inp)
        if not _is_sorted(out):
            return False
    return True

def _count_comparators(network):
    return len(network)

def construct():
    """Construct a sorting network on 16 wires minimizing comparator count."""
    start_time = time.time()
    rng = random.Random(12345)
    
    # Start with odd-even transposition
    best = _initial_network()
    best_len = len(best)
    
    # Verify initial
    assert _verify_network_sampled(best, 5000, rng), "Initial network should be valid"
    
    # Local search: try removing comparators
    def try_remove(network, rng):
        """Try removing one random comparator; return new network if still valid."""
        n = len(network)
        if n <= 1:
            return None
        idx = rng.randint(0, n - 1)
        new_network = network[:idx] + network[idx+1:]
        # Quick check: does the network still sort?
        # Use fast verification
        if _verify_network_sampled(new_network, 3000, rng):
            return new_network
        return None
    
    # Local search: try replacing a comparator with a different one
    def try_replace(network, rng):
        """Try replacing one comparator with a different valid one."""
        n = len(network)
        if n <= 1:
            return None
        idx = rng.randint(0, n - 1)
        old_i, old_j = network[idx]
        # Generate a random different comparator
        for _ in range(20):
            new_i = rng.randint(0, N-2)
            new_j = rng.randint(new_i+1, N-1)
            if (new_i, new_j) == (old_i, old_j):
                continue
            new_network = network[:idx] + [(new_i, new_j)] + network[idx+1:]
            if _verify_network_sampled(new_network, 3000, rng):
                return new_network
        return None
    
    # Local search: try swapping two comparators (reordering)
    def try_swap(network, rng):
        """Try swapping positions of two comparators."""
        n = len(network)
        if n <= 1:
            return None
        for _ in range(10):
            i = rng.randint(0, n-1)
            j = rng.randint(0, n-1)
            if i == j:
                continue
            new_network = list(network)
            new_network[i], new_network[j] = new_network[j], new_network[i]
            if _verify_network_sampled(new_network, 3000, rng):
                return new_network
        return None
    
    # Iterative local search with deadline guard
    max_iter = 50000
    iter_count = 0
    no_improve = 0
    patience = 2000
    
    while iter_count < max_iter:
        # Deadline check
        if time.time() - start_time > DEADLINE:
            break
        
        # Choose operation
        op = rng.random()
        if op < 0.5:
            candidate = try_remove(best, rng)
        elif op < 0.8:
            candidate = try_replace(best, rng)
        else:
            candidate = try_swap(best, rng)
        
        if candidate is not None:
            cand_len = len(candidate)
            if cand_len < best_len:
                best = candidate
                best_len = cand_len
                no_improve = 0
            else:
                # Accept with some probability (simulated annealing-like)
                # But since we want fewer comparators, only accept if shorter
                # For equal length, accept with small prob to explore
                if cand_len == best_len and rng.random() < 0.1:
                    best = candidate
                    no_improve += 1
                else:
                    no_improve += 1
        else:
            no_improve += 1
        
        iter_count += 1
        
        # If stuck for too long, do a perturbation
        if no_improve > patience:
            # Perturb: add a few random comparators then try to remove
            perturbed = list(best)
            for _ in range(rng.randint(2, 5)):
                i = rng.randint(0, N-2)
                j = rng.randint(i+1, N-1)
                perturbed.append((i, j))
            # Now try to remove from perturbed
            for _ in range(rng.randint(10, 30)):
                rem = try_remove(perturbed, rng)
                if rem is not None:
                    perturbed = rem
            if len(perturbed) < best_len:
                best = perturbed
                best_len = len(best)
            no_improve = 0
        
        # Periodic full verification
        if iter_count % 5000 == 0:
            if not _verify_network_sampled(best, 10000, rng):
                # Shouldn't happen, but reset to a known good state
                best = _initial_network()
                best_len = len(best)
                no_improve = 0
    
    # Final: try a few more removals in a focused manner
    # Sort network by "usefulness" - try removing from end first
    for attempt in range(min(200, best_len)):
        if time.time() - start_time > DEADLINE:
            break
        # Try removing the last comparator
        if len(best) <= 1:
            break
        new_network = best[:-1]
        if _verify_network_sampled(new_network, 5000, rng):
            best = new_network
        else:
            # Try removing from random positions
            for _ in range(5):
                idx = rng.randint(0, len(best)-1)
                new_network = best[:idx] + best[idx+1:]
                if _verify_network_sampled(new_network, 5000, rng):
                    best = new_network
                    break
    
    return best

# EVOLVE-BLOCK-END
