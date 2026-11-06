"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
SEED_COMPARATORS = 153
MAX_COMPARATORS = 4096
TIME_LIMIT = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _apply_network(net, x):
    """Apply compare-exchange network to a vector x. Returns sorted check."""
    x = list(x)
    for i, j in net:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _verify_network(net):
    """Verify network sorts all 2^N boolean inputs. Returns True if correct."""
    n = len(net)
    if n == 0:
        return N <= 1
    # Fast path: check all-ones, all-zeros, and a few structured patterns first
    # Then do full verification with numpy bit manipulation
    
    # Full verification: iterate over all 2^N inputs
    # For N=18, that's 262144 inputs
    # Represent each input as an integer, extract bits
    # Apply network, check if output is sorted (non-increasing or all zeros then ones)
    
    # Pre-compute wire indices for fast bit access
    for mask in range(1 << N):
        # Extract bits: bit i is wire i
        # We need to check if after applying the network, the result is sorted
        # A boolean sequence is sorted if it's of the form 111...1000...0
        # i.e., there exists k such that bits 0..k-1 are 1 and bits k..N-1 are 0
        
        # Simulate the network
        val = mask
        for i, j in net:
            bi = (val >> i) & 1
            bj = (val >> j) & 1
            if bi < bj:  # need to swap: i should get max, j should get min
                # i gets 1, j gets 0
                val |= (1 << i)
                val &= ~(1 << j)
            # else: i already >= j, no change needed
        
        # Check if val is sorted: bits should be 1...10...0
        # Find the position where bits change from 1 to 0
        # A sorted boolean array has the form: some number of 1s followed by 0s
        # Check: for each position, if bit is 0, all higher bits must be 0
        # Equivalent: val must be of the form (1 << k) - 1 for some k, or 0
        
        # Check: val & (val + 1) == 0 means val is of the form 2^k - 1 (all ones)
        # Or val == 0
        # But we also need to handle the case where it's like 111000 which is (1<<3)-1 = 7? No.
        # 111000 in binary (6 bits) = 56. 56 & 57 = 56 & 111001 = 111000 & 111001 = 111000 != 0
        # Hmm, that doesn't work. Let me think differently.
        
        # A sorted boolean array of length N is one where there's a split point k
        # such that bits 0..k-1 are 1 and bits k..N-1 are 0.
        # So the value must be (1 << k) - 1 for some k in 0..N.
        
        # Check if val is of the form (1 << k) - 1
        # (1<<k) - 1 has k ones. So val + 1 should be a power of 2, or val == 0.
        if val == 0:
            continue
        if (val & (val + 1)) != 0:
            return False
        # Also need to make sure val < (1 << N) which is always true
    
    return True

def _verify_network_fast(net):
    """Faster verification using numpy for batch processing."""
    n = len(net)
    if n == 0:
        return N <= 1
    
    # Process in batches for speed
    batch_size = 65536
    total = 1 << N
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        masks = np.arange(start, end, dtype=np.uint64)
        
        # Apply each comparator
        for i, j in net:
            bi = (masks >> i) & 1
            bj = (masks >> j) & 1
            # If bi < bj, swap: set bit i to 1, bit j to 0
            swap_mask = (bj > bi)
            masks = np.where(swap_mask, 
                            (masks | (1 << i)) & ~(1 << j), 
                            masks)
        
        # Check if all are sorted: val & (val+1) == 0
        # For uint64, need to be careful
        vals = masks.astype(np.uint64)
        bad = (vals & (vals + 1)) != 0
        if np.any(bad):
            return False
    
    return True

def _get_depth(net):
    """Compute the parallel depth of the network."""
    if not net:
        return 0
    depth = [0] * N
    max_d = 0
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
        max_d = max(max_d, d)
    return max_d

def _batcher_bitonic_sort(n):
    """Generate Batcher's bitonic sorting network for n wires (n should be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # For non-power-of-2, we'll handle separately
    # Standard bitonic sort for power of 2
    comparators = []
    
    def bitonic_merge(k, start, direction):
        """Merge a bitonic sequence of length k starting at 'start'."""
        if k <= 1:
            return
        # First, recursively merge halves in opposite directions
        # For a bitonic sequence of length k:
        # Split into two halves of length k/2
        # Merge first half ascending, second half descending (or vice versa)
        # Then compare elements i and i+k/2
        
        half = k // 2
        # The two halves should already be sorted in opposite directions
        # We compare element start+i with start+i+half for i in 0..half-1
        for i in range(half):
            a = start + i
            b = start + i + half
            if a < b:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
    
    # Build up the bitonic sort
    # Standard algorithm:
    # For k = 2, 4, 8, ..., n:
    #   For j = n, n-2, ..., k:
    #     For i = 0, 1, ..., j-k:
    #       bitonic_merge(k, i, direction depending on i/k parity)
    
    # Actually, let me use the standard recursive approach
    # Bitonic sort network:
    # 1. Split into two halves
    # 2. Sort first half ascending, second half descending
    # 3. Bitonic merge the two halves
    
    def sort_network(size, start, ascending):
        """Generate comparators for sorting 'size' elements starting at 'start'."""
        if size <= 1:
            return
        if size == 2:
            a, b = start, start + 1
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
            return
        
        half = size // 2
        # Sort first half ascending, second half descending (for ascending overall)
        sort_network(half, start, ascending)
        sort_network(half, start + half, not ascending)
        # Bitonic merge
        bitonic_merge(size, start, ascending)
    
    # For 18 wires, we can't use a pure power-of-2 bitonic sort
    # We'll sort 16 wires with bitonic, then merge in the remaining 2
    # Or we can use a different approach
    
    # Let's generate for the largest power of 2 <= N, then handle the rest
    # Actually, for simplicity and to get a valid starting point, let's just
    # use odd-even transposition as the base and improve from there
    
    return None  # We'll use a different approach

def construct():
    """Construct a sorting network on N wires."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Start with odd-even transposition
    best_net = _odd_even_transposition()
    best_len = len(best_net)
    
    # Verify the seed
    if not _verify_network_fast(best_net):
        raise RuntimeError("Seed network is invalid")
    
    # Genetic algorithm with local search refinement
    rng = np.random.default_rng(42)
    
    def mutate(net, rate=0.1):
        """Apply random mutations to the network."""
        net = list(net)
        # Try several mutation types
        for _ in range(max(1, int(len(net) * rate))):
            op = rng.integers(0, 4)
            if op == 0 and net:
                # Remove a random comparator
                idx = rng.integers(0, len(net))
                net.pop(idx)
            elif op == 1:
                # Add a random comparator
                i, j = sorted(rng.choice(N, size=2, replace=False))
                net.append((i, j))
            elif op == 2 and len(net) >= 2:
                # Swap two comparators
                i, j = rng.choice(len(net), size=2, replace=False)
                net[i], net[j] = net[j], net[i]
            elif op == 3 and net:
                # Move a comparator to a random position
                idx = rng.integers(0, len(net))
                comp = net.pop(idx)
                new_idx = rng.integers(0, len(net) + 1)
                net.insert(new_idx, comp)
        
        # Ensure no duplicate comparators and valid indices
        seen = set()
        cleaned = []
        for c in net:
            if c[0] < c[1] and c not in seen:
                seen.add(c)
                cleaned.append(c)
        return cleaned
    
    def local_search(net):
        """Apply local search to minimize comparator count."""
        net = list(net)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            n = len(net)
            # Try removing each comparator
            for i in range(n):
                if time.time() > deadline - 2.0:
                    break
                candidate = net[:i] + net[i+1:]
                if _verify_network_fast(candidate):
                    net = candidate
                    improved = True
                    break
            if not improved:
                # Try swapping pairs of comparators
                for i in range(n):
                    if time.time() > deadline - 2.0:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline - 2.0:
                            break
                        candidate = net[:]
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        if _verify_network_fast(candidate):
                            net = candidate
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                # Try moving a comparator
                for i in range(n):
                    if time.time() > deadline - 2.0:
                        break
                    for j in range(n):
                        if j == i or j == i + 1:
                            continue
                        if time.time() > deadline - 2.0:
                            break
                        candidate = net[:]
                        comp = candidate.pop(i)
                        candidate.insert(j, comp)
                        if _verify_network_fast(candidate):
                            net = candidate
                            improved = True
                            break
                    if improved:
                        break
        return net
    
    # Run local search on the seed
    best_net = local_search(best_net)
    best_len = len(best_net)
    
    # Genetic algorithm
    population = [best_net]
    # Add some diverse individuals
    for _ in range(5):
        pop = mutate(best_net, rate=0.3)
        if len(pop) < best_len and _verify_network_fast(pop):
            population.append(pop)
    
    generation = 0
    while time.time() < deadline - 3.0:
        generation += 1
        
        # Evaluate population
        valid_pops = []
        for pop in population:
            if _verify_network_fast(pop):
                valid_pops.append(pop)
        
        if not valid_pops:
            population = [best_net]
            continue
        
        # Select best
        best_in_gen = min(valid_pops, key=len)
        if len(best_in_gen) < best_len:
            best_net = best_in_gen
            best_len = len(best_net)
        
        # Create next generation
        new_population = [best_net]  # Elitism
        
        # Crossover and mutation
        while len(new_population) < 10 and time.time() < deadline - 3.0:
            if len(valid_pops) >= 2:
                # Crossover
                p1, p2 = rng.choice(len(valid_pops), size=2, replace=False)
                parent1 = valid_pops[p1]
                parent2 = valid_pops[p2]
                
                # Simple crossover: take prefix from one, suffix from other
                if len(parent1) > 0 and len(parent2) > 0:
                    cut = rng.integers(1, min(len(parent1), len(parent2)))
                    child = parent1[:cut] + parent2[cut:]
                else:
                    child = parent1 + parent2
                
                child = mutate(child, rate=0.15)
            else:
                # Mutation only
                child = mutate(best_net, rate=0.2)
            
            if len(child) < best_len and _verify_network_fast(child):
                new_population.append(child)
            elif len(new_population) < 10:
                new_population.append(child)
        
        population = new_population[:10]
        
        # Periodically run local search on the best
        if generation % 5 == 0 and time.time() < deadline - 5.0:
            refined = local_search(best_net)
            if len(refined) < best_len:
                best_net = refined
                best_len = len(best_net)
    
    # Final local search
    if time.time() < deadline - 1.0:
        final = local_search(best_net)
        if len(final) < best_len:
            best_net = final
            best_len = len(best_net)
    
    # Final verification
    if not _verify_network_fast(best_net):
        # Fallback to seed
        return _odd_even_transposition()
    
    return best_net

# EVOLVE-BLOCK-END
