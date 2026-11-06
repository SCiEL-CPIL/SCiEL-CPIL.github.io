"""Size-efficient sorting network on 17 wires via genetic search + bit-vector verification."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 82.0  # safety margin before 90s kill

# Precompute all possible comparator pairs
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
NPAIRS = len(ALL_PAIRS)

# Bitmask for each comparator: bits i and j
# When applying comparator (i,j): new[i] = min(old[i],old[j]), new[j] = max(old[i],old[j])
# In bit form: min = a&b, max = a|b
COMP_I = np.array([p[0] for p in ALL_PAIRS], dtype=np.int64)
COMP_J = np.array([p[1] for p in ALL_PAIRS], dtype=np.int64)


def apply_comparators_to_bits(bits, comparators):
    """
    Apply a list of comparator indices to a bit vector.
    bits: numpy array of shape (n_inputs, N) with 0/1 values
    comparators: list of indices into ALL_PAIRS
    Returns new bits array.
    """
    if len(comparators) == 0:
        return bits
    b = bits.copy()
    # Process comparators in order (they are sequential, not parallel layers)
    for ci in comparators:
        i = COMP_I[ci]
        j = COMP_J[ci]
        # min = a & b, max = a | b
        new_i = b[:, i] & b[:, j]
        new_j = b[:, i] | b[:, j]
        b[:, i] = new_i
        b[:, j] = new_j
    return b


def verify_network(comparators):
    """
    Verify a sorting network using the zero-one principle.
    Test all 2^17 boolean inputs.
    Returns True if the network sorts all inputs.
    """
    if len(comparators) == 0:
        # Only valid if N <= 1
        return N <= 1

    # Generate all 2^17 bit patterns
    total = 1 << N
    # Use numpy to generate all patterns efficiently
    # Pattern k has bit i set if (k >> i) & 1
    # We'll build the input matrix column by column
    # bits[k, i] = (k >> i) & 1

    # Build using broadcasting
    powers = np.arange(N, dtype=np.int64)
    # Create array of all values 0..2^N-1
    vals = np.arange(total, dtype=np.int64)
    # bits[k, i] = (vals[k] >> i) & 1
    # Use reshape trick: vals reshaped as (2^N, 1), powers as (1, N)
    # Actually, let's do it more memory-efficiently
    # For N=17, total=131072, so bits is 131072 x 17 = ~2.2M elements, fine

    # Compute all bits at once
    # bits = (vals[:, None] >> powers[None, :]) & 1
    bits = (vals[:, None] >> powers[None, :]) & 1

    # Apply all comparators
    for ci in comparators:
        i = COMP_I[ci]
        j = COMP_J[ci]
        new_i = bits[:, i] & bits[:, j]
        new_j = bits[:, i] | bits[:, j]
        bits[:, i] = new_i
        bits[:, j] = new_j

    # Check if sorted: for each row, bits should be non-increasing
    # i.e., bits[k, 0] >= bits[k, 1] >= ... >= bits[k, N-1]
    # Equivalent: no row has a 0 followed by a 1
    # Check: for all k, for all i < N-1: not (bits[k,i]==0 and bits[k,i+1]==1)
    # This is: (bits[:, :-1] == 0) & (bits[:, 1:] == 1) should be all False
    bad = (bits[:, :-1] == 0) & (bits[:, 1:] == 1)
    return not bad.any()


def verify_network_fast(comparators, sample_count=None):
    """
    Faster verification: test a subset of inputs first, then full if needed.
    For GA fitness, we can use partial verification for speed.
    """
    if len(comparators) == 0:
        return N <= 1

    # Quick check with random samples
    if sample_count is not None and sample_count < (1 << N):
        rng = np.random.default_rng(42)
        # Generate random bit patterns
        vals = rng.integers(0, 1 << N, size=sample_count, dtype=np.int64)
        powers = np.arange(N, dtype=np.int64)
        bits = (vals[:, None] >> powers[None, :]) & 1

        for ci in comparators:
            i = COMP_I[ci]
            j = COMP_J[ci]
            new_i = bits[:, i] & bits[:, j]
            new_j = bits[:, i] | bits[:, j]
            bits[:, i] = new_i
            bits[:, j] = new_j

        bad = (bits[:, :-1] == 0) & (bits[:, 1:] == 1)
        if bad.any():
            return False

    # Full verification
    return verify_network(comparators)


def compute_depth(comparators):
    """Compute the depth (critical path) of a comparator network."""
    if not comparators:
        return 0
    # depth[i] = earliest time wire i can be considered "settled"
    # Actually, depth is the length of the longest chain of dependent comparators
    # Two comparators conflict if they share a wire
    # Build conflict graph and find longest path
    # Simpler: simulate levels
    # A comparator can be placed at level d if neither of its wires was used at level d-1
    wire_last_level = [-1] * N
    max_level = 0
    for ci in comparators:
        i = COMP_I[ci]
        j = COMP_J[ci]
        # Earliest level this comparator can go
        level = max(wire_last_level[i], wire_last_level[j]) + 1
        wire_last_level[i] = level
        wire_last_level[j] = level
        max_level = max(max_level, level)
    return max_level


def odd_even_transposition_network():
    """Generate the odd-even transposition sorting network."""
    comparators = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def batcher_odd_even_network(n):
    """Generate Batcher's odd-even merge sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]

    # Recursive: split into halves, sort each, merge
    half = n // 2
    # Sort first half and second half
    # This is complex; let's use a simpler approach
    # Actually, let's just use odd-even transposition for simplicity
    return odd_even_transposition_network()


def pair_to_index(i, j):
    """Convert (i,j) pair to index in ALL_PAIRS."""
    # ALL_PAIRS is ordered by i then j
    # Index = i*(N-1) + (j - i - 1)
    # Because for fixed i, j ranges from i+1 to N-1, that's N-1-i values
    # Cumulative: sum_{k=0}^{i-1} (N-1-k) + (j-i-1)
    # = i*(N-1) - i*(i-1)/2 + j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)


def index_to_pair(idx):
    """Convert index back to (i,j) pair."""
    return ALL_PAIRS[idx]


def mutate(comparators, rng, mutation_rate=0.1):
    """Apply mutations to a comparator list."""
    new_comp = list(comparators)
    n = len(new_comp)

    # Random mutations
    for _ in range(max(1, int(n * mutation_rate))):
        op = rng.integers(0, 4)
        if op == 0 and n > 0:
            # Remove a random comparator
            idx = rng.integers(0, n)
            new_comp.pop(idx)
            n -= 1
        elif op == 1:
            # Add a random comparator
            new_comp.append(rng.integers(0, NPAIRS))
            n += 1
        elif op == 2 and n > 0:
            # Change a random comparator
            idx = rng.integers(0, n)
            new_comp[idx] = rng.integers(0, NPAIRS)
        elif op == 3 and n > 1:
            # Swap two comparators
            i1 = rng.integers(0, n)
            i2 = rng.integers(0, n)
            new_comp[i1], new_comp[i2] = new_comp[i2], new_comp[i1]

    # Limit size
    if len(new_comp) > MAX_COMPARATORS:
        new_comp = new_comp[:MAX_COMPARATORS]

    return new_comp


def crossover(comp1, comp2, rng):
    """Crossover two comparator lists."""
    if len(comp1) == 0:
        return list(comp2)
    if len(comp2) == 0:
        return list(comp1)

    # Uniform crossover
    child = []
    for c1, c2 in zip(comp1, comp2):
        if rng.random() < 0.5:
            child.append(c1)
        else:
            child.append(c2)
    # Add remaining
    for c in comp1[len(child):]:
        child.append(c)
    for c in comp2[len(child):]:
        child.append(c)

    if len(child) > MAX_COMPARATORS:
        child = child[:MAX_COMPARATORS]

    return child


def fitness(comparators, time_budget=None):
    """
    Compute fitness: lower is better.
    Uses comparator count as primary, with correctness check.
    For GA, we use a proxy fitness that's fast to compute.
    """
    if not verify_network_fast(comparators, sample_count=5000):
        return float('inf')

    # For the GA, we want to minimize comparator count
    # But we also need to ensure eventual correctness
    # Use count as fitness, but only if it passes the quick check
    return len(comparators)


def evolve(pop_size=50, generations=200, mutation_rate=0.15, crossover_rate=0.7):
    """
    Genetic algorithm to find a small sorting network.
    Returns the best network found.
    """
    start_time = time.time()
    rng = np.random.default_rng(12345)

    # Initialize population
    # Start with odd-even transposition and variations
    base_network = odd_even_transposition_network()
    base_indices = [pair_to_index(i, j) for i, j in base_network]

    population = []
    # First individual: the base network
    population.append(list(base_indices))

    # Add mutated versions
    for _ in range(pop_size - 1):
        mutated = mutate(base_indices, rng, mutation_rate=0.3)
        population.append(mutated)

    best_network = None
    best_fitness = float('inf')

    for gen in range(generations):
        if time.time() - start_time > DEADLINE - 5:
            break

        # Evaluate fitness
        fitnesses = []
        for comp in population:
            f = fitness(comp)
            fitnesses.append(f)
            if f < best_fitness:
                best_fitness = f
                best_network = list(comp)

        # Selection: tournament
        new_population = []
        for _ in range(pop_size):
            # Tournament selection
            tournament_size = 5
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            winner = min(candidates, key=lambda idx: fitnesses[idx])
            parent1 = population[winner]

            if rng.random() < crossover_rate and len(population) > 1:
                # Pick second parent
                candidates2 = rng.choice(pop_size, size=tournament_size, replace=False)
                winner2 = min(candidates2, key=lambda idx: fitnesses[idx])
                parent2 = population[winner2]
                child = crossover(parent1, parent2, rng)
            else:
                child = list(parent1)

            child = mutate(child, rng, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_network, best_fitness


def try_heuristic_optimization():
    """
    Try to optimize the odd-even transposition network by removing
    redundant comparators.
    """
    start_time = time.time()
    base_network = odd_even_transposition_network()
    base_indices = [pair_to_index(i, j) for i, j in base_network]

    current = list(base_indices)
    current_count = len(current)

    # Try removing each comparator and checking if still valid
    # This is O(n * verification) which is expensive
    # Use greedy removal
    changed = True
    while changed and time.time() - start_time < DEADLINE - 10:
        changed = False
        for idx in range(len(current)):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Try removing this comparator
            trial = current[:idx] + current[idx+1:]
            if verify_network_fast(trial, sample_count=10000):
                # Do full verification
                if verify_network(trial):
                    current = trial
                    current_count -= 1
                    changed = True
                    break  # Restart the loop since indices changed

    return current, current_count


def construct():
    """
    Construct a sorting network for 17 wires.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()

    # Start with the odd-even transposition network
    base_network = odd_even_transposition_network()
    best_network = base_network
    best_count = len(base_network)

    # Try heuristic optimization (greedy removal)
    if time.time() - start_time < DEADLINE - 30:
        optimized, opt_count = try_heuristic_optimization()
        if opt_count < best_count:
            best_network = optimized
            best_count = opt_count

    # Try genetic algorithm if time permits
    if time.time() - start_time < DEADLINE - 20:
        ga_network, ga_fitness = evolve(
            pop_size=30,
            generations=100,
            mutation_rate=0.1,
            crossover_rate=0.7
        )
        if ga_network is not None and ga_fitness < best_count:
            # Verify the GA result fully
            if verify_network(ga_network):
                best_network = ga_network
                best_count = ga_fitness

    # Final safety check
    if not verify_network(best_network):
        # Fall back to the known-correct odd-even transposition
        best_network = base_network

    # Check deadline
    if time.time() - start_time > DEADLINE - 2:
        # Return immediately
        pass

    # Convert indices back to pairs
    result = [index_to_pair(ci) for ci in best_network]

    # Final verification
    if not verify_network(result):
        result = base_network

    return result


# Pre-verify at module load time to catch issues early
# (but don't spend too much time)
if __name__ == "__main__":
    network = construct()
    print(f"Network size: {len(network)} comparators")
    print(f"Depth: {compute_depth(network)}")
    print(f"Verified: {verify_network(network)}")
