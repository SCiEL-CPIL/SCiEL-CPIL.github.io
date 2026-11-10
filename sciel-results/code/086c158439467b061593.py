import time
import random
from itertools import combinations
from math import comb

START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # safety margin before 150s kill

def count_regular_subgraphs_size6(n, adj):
    """Count the number of 6-vertex subsets that induce a regular subgraph."""
    count = 0
    vertices = range(n)
    for subset in combinations(vertices, 6):
        # Compute degrees within the subset
        degs = []
        for u in subset:
            deg = 0
            for v in subset:
                if u != v and (adj[u] >> v) & 1:
                    deg += 1
            degs.append(deg)
        if len(set(degs)) == 1:
            count += 1
    return count

def is_valid_no_regular_6plus(n, adj):
    """Check if no subset of size >= 6 induces a regular subgraph."""
    # Check size 6 first (most common violation)
    for subset in combinations(range(n), 6):
        degs = []
        for u in subset:
            deg = 0
            for v in subset:
                if u != v and (adj[u] >> v) & 1:
                    deg += 1
            degs.append(deg)
        if len(set(degs)) == 1:
            return False
    # Check larger sizes
    for size in range(7, n + 1):
        for subset in combinations(range(n), size):
            degs = []
            for u in subset:
                deg = 0
                for v in subset:
                    if u != v and (adj[u] >> v) & 1:
                        deg += 1
                degs.append(deg)
            if len(set(degs)) == 1:
                return False
    return True

def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 143.0

    n = 21
    num_pairs = n * (n - 1) // 2  # 210 edges possible

    def random_graph():
        """Generate a random graph as a list of 0/1 for each pair."""
        return [random.random() < 0.5 for _ in range(num_pairs)]

    def pair_index(u, v):
        if u > v:
            u, v = v, u
        # Map (u,v) to index in the list
        # Pairs: (0,1),(0,2),...,(0,20),(1,2),(1,3),...
        # Index = u*(n-1) - u*(u+1)//2 + (v - u - 1)
        return u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)

    def adj_from_genome(genome):
        """Convert genome (list of 0/1) to adjacency list (bitmasks)."""
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                idx = pair_index(u, v)
                if genome[idx]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def fitness(genome):
        """Negative count of regular 6-subsets. Higher is better (fewer violations)."""
        adj = adj_from_genome(genome)
        return -count_regular_subgraphs_size6(n, adj)

    def crossover(g1, g2):
        """Uniform crossover."""
        return [g1[i] if random.random() < 0.5 else g2[i] for i in range(num_pairs)]

    def mutate(genome, rate=0.05):
        """Bit-flip mutation."""
        new_g = genome[:]
        for i in range(num_pairs):
            if random.random() < rate:
                new_g[i] = not new_g[i]
        return new_g

    def validate_full(genome):
        """Full validation: check all subsets of size >= 6."""
        adj = adj_from_genome(genome)
        return is_valid_no_regular_6plus(n, adj)

    # Initialize population
    pop_size = 30
    population = [random_graph() for _ in range(pop_size)]

    # Evaluate initial population
    best_genome = None
    best_fit = float('-inf')
    for g in population:
        f = fitness(g)
        if f > best_fit:
            best_fit = f
            best_genome = g[:]

    # Check if any is already valid
    if best_fit == 0:
        if validate_full(best_genome):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if best_genome[pair_index(u, v)]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}

    # Genetic algorithm
    generation = 0
    max_generations = 500

    while generation < max_generations:
        if time.time() > deadline - 15:
            break

        generation += 1

        # Evaluate all
        fits = [fitness(g) for g in population]

        # Track best
        max_idx = max(range(pop_size), key=lambda i: fits[i])
        if fits[max_idx] > best_fit:
            best_fit = fits[max_idx]
            best_genome = population[max_idx][:]

        if best_fit == 0:
            if validate_full(best_genome):
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if best_genome[pair_index(u, v)]:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}

        # Selection: tournament
        def select():
            a, b = random.sample(range(pop_size), 2)
            return population[a] if fits[a] >= fits[b] else population[b]

        # Create next generation
        new_pop = []
        # Elitism: keep best
        new_pop.append(best_genome[:])

        while len(new_pop) < pop_size:
            if time.time() > deadline - 10:
                break
            p1 = select()
            p2 = select()
            child = crossover(p1, p2)
            child = mutate(child, rate=0.03)
            new_pop.append(child)

        population = new_pop

    # Final check on best
    if best_genome is not None and best_fit == 0:
        if validate_full(best_genome):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if best_genome[pair_index(u, v)]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}

    # Fallback: try to fix best_genome with local search
    if best_genome is not None:
        adj = adj_from_genome(best_genome)
        # Local search: flip edges to reduce violations
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            # Find a violating 6-set
            violated = False
            for subset in combinations(range(n), 6):
                degs = []
                for u in subset:
                    deg = 0
                    for v in subset:
                        if u != v and (adj[u] >> v) & 1:
                            deg += 1
                    degs.append(deg)
                if len(set(degs)) == 1:
                    violated = True
                    # Try flipping an edge in or out of this subset
                    flipped = False
                    for u in subset:
                        for v in range(n):
                            if u == v:
                                continue
                            idx = pair_index(u, v)
                            # Flip
                            best_genome[idx] = not best_genome[idx]
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                            # Quick check: does this help?
                            new_fit = fitness(best_genome)
                            if new_fit > best_fit:
                                best_fit = new_fit
                                flipped = True
                                break
                            else:
                                # Revert
                                best_genome[idx] = not best_genome[idx]
                                adj[u] ^= (1 << v)
                                adj[v] ^= (1 << u)
                        if flipped:
                            break
                    if not flipped:
                        break
                    if best_fit == 0:
                        break
            if best_fit == 0:
                break

        if best_fit == 0 and validate_full(best_genome):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if best_genome[pair_index(u, v)]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}

    # Fallback: try n=20 with random search
    n2 = 20
    num_pairs2 = n2 * (n2 - 1) // 2

    def pair_index2(u, v):
        if u > v:
            u, v = v, u
        return u * (n2 - 1) - u * (u + 1) // 2 + (v - u - 1)

    for seed in range(500):
        if time.time() > deadline - 3:
            break
        random.seed(seed)
        genome2 = [random.random() < 0.5 for _ in range(num_pairs2)]
        adj2 = [0] * n2
        for u in range(n2):
            for v in range(u + 1, n2):
                idx = pair_index2(u, v)
                if genome2[idx]:
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)
        valid = True
        for subset in combinations(range(n2), 6):
            degs = []
            for u in subset:
                deg = 0
                for v in subset:
                    if u != v and (adj2[u] >> v) & 1:
                        deg += 1
                degs.append(deg)
            if len(set(degs)) == 1:
                valid = False
                break
        if not valid:
            continue
        # Check larger
        for size in range(7, n2 + 1):
            for subset in combinations(range(n2), size):
                degs = []
                for u in subset:
                    deg = 0
                    for v in subset:
                        if u != v and (adj2[u] >> v) & 1:
                            deg += 1
                    degs.append(deg)
                if len(set(degs)) == 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            edges = []
            for u in range(n2):
                for v in range(u + 1, n2):
                    if genome2[pair_index2(u, v)]:
                        edges.append([u, v])
            return {'n': n2, 'edges': edges}

    # Ultimate fallback: empty graph on 5 vertices
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
