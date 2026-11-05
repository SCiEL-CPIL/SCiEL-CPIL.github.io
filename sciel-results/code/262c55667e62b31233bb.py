"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0
    
    # 1. Input Generation
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for col in range(N):
        period = 1 << (col + 1)
        half = 1 << col
        block = np.zeros(period, dtype=np.uint8)
        block[half:] = 1
        repeats = (1 << N) // period
        inputs[:, col] = np.tile(block, repeats)
    
    # 2. Fast Verification Setup
    rng_seed = 42
    rng = np.random.RandomState(rng_seed)
    hard_indices = rng.choice(1 << N, size=1024, replace=False)
    hard_inputs = inputs[hard_indices]
    
    state_buffer_full = np.empty_like(inputs)
    state_buffer_hard = np.empty_like(hard_inputs)
    
    def verify_full(net):
        """Verify against all 2^N inputs. Returns True if sorted."""
        np.copyto(state_buffer_full, inputs)
        for i, j in net:
            mi = state_buffer_full[:, i] & state_buffer_full[:, j]
            ma = state_buffer_full[:, i] | state_buffer_full[:, j]
            state_buffer_full[:, i] = mi
            state_buffer_full[:, j] = ma
        return np.all(state_buffer_full[:, :-1] <= state_buffer_full[:, 1:])

    def verify_hard(net):
        """Verify against the subset of hard inputs. Fast filter."""
        np.copyto(state_buffer_hard, hard_inputs)
        for i, j in net:
            mi = state_buffer_hard[:, i] & state_buffer_hard[:, j]
            ma = state_buffer_hard[:, i] | state_buffer_hard[:, j]
            state_buffer_hard[:, i] = mi
            state_buffer_hard[:, j] = ma
        return np.all(state_buffer_hard[:, :-1] <= state_buffer_hard[:, 1:])

    def _odd_even_network():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    def _batcher_bitonic(n):
        """Generate Batcher's bitonic sorting network for n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Recursively build bitonic network
        if n % 2 == 0:
            # Split into two halves
            half = n // 2
            left_net = _batcher_bitonic(half)
            right_net = _batcher_bitonic(half)
            
            # Offset right network
            right_net = [(i + half, j + half) for i, j in right_net]
            
            # Combine
            combined = left_net + right_net
        else:
            # For odd n, use a simpler approach: sort n-1 wires, then insert
            # Actually, let's just use odd-even transposition for simplicity
            # or fall back to a known construction
            return _odd_even_network()
        
        # Add merge comparators
        # For bitonic merge, we need to merge two sorted sequences
        # This is complex; for now, let's just use odd-even as fallback
        # A proper bitonic merge would add O(n log n) comparators
        
        # Since implementing full bitonic is complex, let's use a hybrid approach:
        # Use odd-even transposition but try to optimize it
        return _odd_even_network()
    
    # Start with odd-even transposition as baseline
    best_network = _odd_even_network()
    best_count = len(best_network)
    
    # Verify initial network
    if not verify_full(best_network):
        return best_network
    
    def _is_valid_topology(net):
        """
        Check if the network has a valid topological structure.
        A valid topology means that for any comparator (i,j) at position p,
        no subsequent comparator at position q > p can have a wire that
        'crosses' in a way that violates causality.
        
        Actually, for sorting networks, any sequence of comparators is valid
        in the sense that it defines a network. The question is whether it sorts.
        So this check is just for structural sanity (no self-loops, etc.)
        """
        for i, j in net:
            if i >= j or i < 0 or j >= N:
                return False
        return True
    
    def _crossover(parent1, parent2, rng):
        """
        Topology-preserving crossover.
        We use a 'segment' crossover where we take alternating segments
        from each parent, but ensure the result is still a valid network.
        
        Strategy: Pick a random split point, take first half from parent1,
        second half from parent2, then apply light mutation to fix any issues.
        """
        len1 = len(parent1)
        len2 = len(parent2)
        
        # Choose target length between the two parents
        target_len = int(rng.randint(min(len1, len2), max(len1, len2) + 1))
        
        # Sample comparators from both parents
        child = []
        for _ in range(target_len):
            # Randomly pick from parent1 or parent2
            if rng.random() < 0.5:
                if parent1:
                    idx = rng.randint(len(parent1))
                    child.append(parent1[idx])
            else:
                if parent2:
                    idx = rng.randint(len(parent2))
                    child.append(parent2[idx])
        
        # Remove duplicates to avoid redundant comparators
        seen = set()
        unique_child = []
        for comp in child:
            if comp not in seen:
                seen.add(comp)
                unique_child.append(comp)
        
        return unique_child
    
    def _mutation(net, rng, mutation_rate=0.1):
        """
        Apply mutations to the network.
        Mutation types:
        1. Remove a random comparator (with some probability)
        2. Change a random comparator to a different one
        3. Move a comparator to a different position
        4. Add a new random comparator (with lower probability)
        """
        if not net:
            return net
        
        result = list(net)
        n_muts = max(1, int(len(net) * mutation_rate))
        
        for _ in range(n_muts):
            if not result:
                break
            
            mut_type = rng.random()
            
            if mut_type < 0.3:
                # Remove
                idx = rng.randint(len(result))
                result.pop(idx)
            elif mut_type < 0.6:
                # Change
                idx = rng.randint(len(result))
                new_i = rng.randint(N)
                new_j = rng.randint(N)
                if new_i == new_j:
                    new_j = (new_j + 1) % N
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                result[idx] = (new_i, new_j)
            elif mut_type < 0.8:
                # Move
                idx = rng.randint(len(result))
                comp = result[idx]
                result.pop(idx)
                insert_idx = rng.randint(len(result) + 1)
                result.insert(insert_idx, comp)
            else:
                # Add (only if below some limit)
                if len(result) < 200:  # Safety limit
                    new_i = rng.randint(N)
                    new_j = rng.randint(N)
                    if new_i == new_j:
                        new_j = (new_j + 1) % N
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    insert_idx = rng.randint(len(result) + 1)
                    result.insert(insert_idx, (new_i, new_j))
        
        return result
    
    def _fitness(net):
        """
        Compute fitness: lower is better.
        Primary: number of comparators
        Secondary: whether it passes hard verification (penalty if not)
        """
        if not _is_valid_topology(net):
            return float('inf')
        
        # Check hard verification
        if not verify_hard(net):
            # Penalize invalid networks heavily
            return len(net) + 1000
        
        return len(net)
    
    # Genetic Algorithm
    pop_size = 30
    population = []
    
    # Initialize population with diverse starting points
    # 1. Odd-even transposition
    population.append(_odd_even_network())
    
    # 2. Some random valid networks
    for i in range(pop_size - 1):
        # Start from odd-even and apply random mutations
        base = _odd_even_network()
        mutated = _mutation(base, rng, mutation_rate=0.2)
        population.append(mutated)
    
    # Sort by fitness
    population.sort(key=lambda net: _fitness(net))
    
    best_network = population[0]
    best_count = len(best_network)
    
    # GA loop
    generation = 0
    while time.time() < deadline - 2.0:  # Leave 2s buffer
        generation += 1
        
        # Selection: tournament selection
        def tournament_select(pop, k=3):
            """Select k random individuals, return the best."""
            indices = rng.randint(len(pop), size=k)
            best_idx = indices[0]
            best_fit = _fitness(pop[best_idx])
            for idx in indices[1:]:
                fit = _fitness(pop[idx])
                if fit < best_fit:
                    best_fit = fit
                    best_idx = idx
            return pop[best_idx]
        
        # Generate next generation
        new_population = []
        
        # Elitism: keep top few
        elite_count = max(2, pop_size // 5)
        for i in range(elite_count):
            new_population.append(population[i])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover
            child = _crossover(parent1, parent2, rng)
            
            # Mutation
            child = _mutation(child, rng, mutation_rate=0.15)
            
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        population.sort(key=lambda net: _fitness(net))
        
        # Update best
        if _fitness(population[0]) < best_count:
            # Verify full before updating
            if verify_full(population[0]):
                best_network = population[0]
                best_count = len(best_network)
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Final verification
    if not verify_full(best_network):
        # Fallback to odd-even
        best_network = _odd_even_network()
    
    return best_network

# EVOLVE-BLOCK-END
