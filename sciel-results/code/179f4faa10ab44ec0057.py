"""Rewritten program: constructive heuristic + genetic algorithm for 18-wire sorting network."""
import numpy as np
import random
import time

N = 18
NUM_COMPARATORS_LIMIT = 4096
TIME_LIMIT = 55  # seconds, leaving buffer for 60s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 18 wires using a hybrid approach:
    1. Generate initial population using Batcher's bitonic sort (valid, ~130 comparators)
    2. Apply genetic algorithm with crossover and mutation to reduce comparator count
    3. Return the best valid network found
    
    The network is a list of (i, j) pairs where i < j, and each comparator
    writes min to wire i and max to wire j.
    """
    random.seed(42)
    np.random.seed(42)
    
    start_time = time.time()
    
    # --- Helper functions ---
    
    def verify_network(network):
        """Verify that the network sorts all 2^18 Boolean inputs."""
        n = len(network)
        if n == 0:
            return True
        
        # For each of the 2^18 possible Boolean inputs, check if sorted
        # Use vectorized operations for speed
        # We'll test in chunks to manage memory
        
        chunk_size = 4096
        total_inputs = 1 << N
        
        for start in range(0, total_inputs, chunk_size):
            end = min(start + chunk_size, total_inputs)
            # Generate all inputs from start to end
            inputs = np.arange(start, end, dtype=np.int32)
            
            # Create bit matrix: each row is an input, each column is a wire
            # Bit k of input value goes to wire k
            bits = np.zeros((end - start, N), dtype=np.uint8)
            for k in range(N):
                bits[:, k] = (inputs >> k) & 1
            
            # Apply each comparator
            for i, j in network:
                # For each input, compare wire i and wire j
                # min goes to i, max goes to j
                min_val = np.minimum(bits[:, i], bits[:, j])
                max_val = np.maximum(bits[:, i], bits[:, j])
                bits[:, i] = min_val
                bits[:, j] = max_val
            
            # Check if sorted: for each input, bits should be non-increasing
            # i.e., bits[:, k] >= bits[:, k+1] for all k
            diffs = np.diff(bits, axis=1)
            if np.any(diffs > 0):
                return False
        
        return True
    
    def count_comparators(network):
        return len(network)
    
    def get_depth(network):
        """Compute the parallel depth of the network."""
        if not network:
            return 0
        depth = np.zeros(N, dtype=np.int32)
        for i, j in network:
            new_depth = max(depth[i], depth[j]) + 1
            depth[i] = new_depth
            depth[j] = new_depth
        return int(np.max(depth))
    
    def bitonic_sort_network(n):
        """
        Generate a Batcher bitonic sorting network for n wires (n must be power of 2).
        Returns list of (i, j) comparators.
        """
        if n <= 1:
            return []
        
        comparators = []
        
        def bitonic_merge(start, length, direction):
            """Merge two adjacent sorted sequences of 'length' each into a bitonic sequence."""
            if length <= 1:
                return
            half = length // 2
            # Compare elements that are 'half' apart
            for i in range(start, start + length):
                j = i + half
                if i < j:
                    if direction == 1:  # ascending
                        comparators.append((i, j))
                    else:  # descending
                        comparators.append((i, j))  # same, since we always put min at i
            # Recursively merge the two halves
            bitonic_merge(start, half, direction)
            bitonic_merge(start + half, half, direction)
        
        def bitonic_sort(start, length, direction=1):
            """Sort a sequence of 'length' elements starting at 'start'."""
            if length <= 1:
                return
            half = length // 2
            # Sort first half ascending, second half descending (or vice versa)
            bitonic_sort(start, half, 1)
            bitonic_sort(start + half, half, -1)
            # Merge the two halves
            bitonic_merge(start, length, direction)
        
        bitonic_sort(0, n, 1)
        return comparators
    
    def bitonic_sort_for_n(n):
        """
        Generate a bitonic sorting network for n wires where n is not necessarily a power of 2.
        We pad to next power of 2, generate network, then remove comparators involving padded wires.
        """
        # Find next power of 2
        n_pow2 = 1
        while n_pow2 < n:
            n_pow2 *= 2
        
        # Generate bitonic sort for n_pow2 wires
        full_network = bitonic_sort_network(n_pow2)
        
        # Filter out comparators that involve wires >= n
        filtered = [(i, j) for i, j in full_network if i < n and j < n]
        
        return filtered
    
    def odd_even_transposition(n):
        """Generate odd-even transposition sorting network for n wires."""
        comparators = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comparators.append((i, i + 1))
        return comparators
    
    def crossover(parent1, parent2):
        """
        Crossover: take a prefix from parent1 and suffix from parent2,
        then try to repair validity by removing redundant comparators.
        """
        if not parent1 or not parent2:
            return parent1 if parent1 else parent2
        
        # Random crossover point
        point = random.randint(0, min(len(parent1), len(parent2)))
        
        child = parent1[:point] + parent2[point:]
        
        # Remove duplicate comparators (keep first occurrence)
        seen = set()
        unique = []
        for comp in child:
            if comp not in seen:
                seen.add(comp)
                unique.append(comp)
        
        return unique
    
    def mutate(network, mutation_rate=0.1):
        """
        Mutate the network by:
        1. Randomly removing some comparators
        2. Randomly adding some comparators
        """
        if not network:
            return network
        
        child = list(network)
        
        # Random removal
        for i in range(len(child)):
            if random.random() < mutation_rate:
                # Remove this comparator
                child.pop(i)
                i -= 1
        
        # Random addition
        num_additions = random.randint(0, 3)
        for _ in range(num_additions):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            child.append((i, j))
        
        # Remove duplicates
        seen = set()
        unique = []
        for comp in child:
            if comp not in seen:
                seen.add(comp)
                unique.append(comp)
        
        return unique
    
    def simplify(network):
        """
        Try to simplify the network by removing redundant comparators.
        A comparator (i, j) is redundant if removing it still leaves a valid network.
        We use a greedy approach: try removing each comparator and check validity.
        This is expensive, so we only do it on promising candidates.
        """
        if not network:
            return network
        
        # Only try simplification if network is small enough
        if len(network) > 200:
            return network
        
        child = list(network)
        i = 0
        while i < len(child):
            # Try removing comparator at position i
            test_network = child[:i] + child[i+1:]
            if verify_network(test_network):
                child = test_network
                # Don't increment i since we removed an element
            else:
                i += 1
        
        return child
    
    # --- Generate initial population ---
    
    # Start with Batcher's bitonic sort for 18 wires
    bitonic_net = bitonic_sort_for_n(N)
    
    # Also start with odd-even transposition
    oet_net = odd_even_transposition(N)
    
    # Create initial population
    population = []
    
    # Add the two base networks
    population.append(bitonic_net)
    population.append(oet_net)
    
    # Add some mutated versions
    for _ in range(10):
        base = random.choice([bitonic_net, oet_net])
        mutated = mutate(base, mutation_rate=0.05)
        population.append(mutated)
    
    # --- Evaluate initial population ---
    
    best_network = None
    best_count = float('inf')
    
    for net in population:
        if len(net) >= best_count:
            continue
        if verify_network(net):
            best_network = net
            best_count = len(net)
    
    # --- Genetic algorithm ---
    
    max_generations = 50
    pop_size = 20
    
    for gen in range(max_generations):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Create next generation
        next_population = []
        
        # Elitism: keep the best
        if best_network:
            next_population.append(list(best_network))
        
        # Generate new individuals
        while len(next_population) < pop_size:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Choose parents
            parent1 = random.choice(population) if population else bitonic_net
            parent2 = random.choice(population) if population else oet_net
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate=0.05)
            
            # Limit size
            if len(child) > 200:
                child = child[:200]
            
            next_population.append(child)
        
        # Evaluate new population
        for net in next_population:
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            if len(net) >= best_count:
                continue
            if verify_network(net):
                # Try to simplify
                simplified = simplify(net)
                if len(simplified) < len(net):
                    net = simplified
                best_network = net
                best_count = len(net)
        
        population = next_population
    
    # If we didn't find any valid network (shouldn't happen), return bitonic
    if best_network is None:
        return bitonic_net
    
    return best_network
# EVOLVE-BLOCK-END
