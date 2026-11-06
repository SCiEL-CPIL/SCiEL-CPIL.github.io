"""Rewritten program: depth-aware genetic algorithm for sorting networks on 17 wires."""
import numpy as np
import random
import sys
import time

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55  # seconds, leaving buffer for the 60s limit

def _verify_network(comparators):
    """
    Verify a sorting network using the Zero-One Principle with bit-vector simulation.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    num_inputs = 1 << N
    state = np.zeros((num_inputs, N), dtype=np.uint8)
    for j in range(N):
        state[:, j] = ((np.arange(num_inputs, dtype=np.uint32) >> j) & 1).astype(np.uint8)
    
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    
    diffs = state[:, :-1].astype(np.int8) - state[:, 1:].astype(np.int8)
    return np.all(diffs >= 0)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _generate_layer(n_wires, max_comparators=None):
    """Generate a valid layer (antichain) of comparators."""
    if max_comparators is None:
        max_comparators = n_wires // 2
    used = set()
    layer = []
    candidates = []
    for i in range(n_wires - 1):
        for j in range(i + 1, n_wires):
            candidates.append((i, j))
    random.shuffle(candidates)
    for (i, j) in candidates:
        if len(layer) >= max_comparators:
            break
        if i not in used and j not in used:
            layer.append((i, j))
            used.add(i)
            used.add(j)
    return layer

def _generate_network_by_depth(num_layers, avg_per_layer=4):
    """Generate a network by creating layers of comparators."""
    comparators = []
    for _ in range(num_layers):
        layer_size = max(1, avg_per_layer + random.randint(-2, 2))
        layer = _generate_layer(N, layer_size)
        comparators.extend(layer)
    return comparators

def _crossover(parent1, parent2):
    """Crossover: take a random prefix from parent1 and suffix from parent2."""
    if not parent1 or not parent2:
        return list(parent1) if parent1 else list(parent2)
    point = random.randint(0, min(len(parent1), len(parent2)))
    child = list(parent1[:point]) + list(parent2[point:])
    return child

def _mutate(comparators, mutation_rate=0.1):
    """Mutate a network: add, remove, or swap comparators."""
    comparators = list(comparators)
    n_mutations = max(1, int(len(comparators) * mutation_rate)) + random.randint(0, 2)
    for _ in range(n_mutations):
        op = random.choice(['add', 'remove', 'swap', 'reorder', 'move'])
        if op == 'add' and len(comparators) < MAX_COMPARATORS:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(comparators))
            comparators.insert(pos, (i, j))
        elif op == 'remove' and len(comparators) > 1:
            pos = random.randint(0, len(comparators) - 1)
            comparators.pop(pos)
        elif op == 'swap' and len(comparators) >= 2:
            i, j = random.sample(range(len(comparators)), 2)
            comparators[i], comparators[j] = comparators[j], comparators[i]
        elif op == 'reorder' and len(comparators) >= 2:
            i, j = random.sample(range(len(comparators)), 2)
            comparators[i], comparators[j] = comparators[j], comparators[i]
        elif op == 'move' and len(comparators) >= 2:
            i = random.randint(0, len(comparators) - 1)
            j = random.randint(0, len(comparators) - 1)
            comp = comparators.pop(i)
            comparators.insert(j, comp)
    return comparators

def _remove_redundant(comparators, max_iterations=50):
    """Try to remove redundant comparators one at a time."""
    comparators = list(comparators)
    for _ in range(max_iterations):
        if len(comparators) <= 1:
            break
        found = False
        for i in range(len(comparators)):
            candidate = comparators[:i] + comparators[i+1:]
            if _verify_network(candidate):
                comparators = candidate
                found = True
                break
        if not found:
            break
    return comparators

def _evaluate_fitness(comparators):
    """
    Fitness: lower is better.
    If the network is valid, fitness = number of comparators.
    If invalid, fitness = a large penalty.
    """
    if len(comparators) > MAX_COMPARATORS:
        return MAX_COMPARATORS + 1000
    if _verify_network(comparators):
        return len(comparators)
    else:
        return MAX_COMPARATORS + 1000

def _genetic_algorithm():
    """
    Run a depth-aware genetic algorithm to find a small sorting network.
    """
    start_time = time.time()
    
    pop_size = 25
    population = []
    
    seed = _odd_even_transposition()
    population.append(seed)
    
    for i in range(pop_size - 1):
        num_layers = random.randint(8, 15)
        avg_per = random.randint(3, 8)
        network = _generate_network_by_depth(num_layers, avg_per)
        population.append(network)
    
    fitnesses = [_evaluate_fitness(ind) for ind in population]
    
    best_network = min(zip(fitnesses, population), key=lambda x: x[0])
    best_fitness = best_network[0]
    best_comparators = best_network[1]
    
    print(f"Initial best: {best_fitness} comparators", file=sys.stderr)
    
    generation = 0
    while time.time() - start_time < TIME_LIMIT:
        generation += 1
        
        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        elite_size = max(2, pop_size // 4)
        new_population = [ind for _, ind in sorted_pop[:elite_size]]
        
        while len(new_population) < pop_size:
            r = random.random()
            if r < 0.4 and len(population) >= 2:
                p1 = random.choice(sorted_pop[:elite_size])[1]
                p2 = random.choice(sorted_pop[:elite_size])[1]
                child = _crossover(p1, p2)
            elif r < 0.7:
                parent = random.choice(sorted_pop[:elite_size])[1]
                child = _mutate(parent, mutation_rate=0.05)
            else:
                num_layers = random.randint(8, 15)
                avg_per = random.randint(3, 8)
                child = _generate_network_by_depth(num_layers, avg_per)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [_evaluate_fitness(ind) for ind in population]
        
        gen_best = min(zip(fitnesses, population), key=lambda x: x[0])
        if gen_best[0] < best_fitness:
            best_fitness = gen_best[0]
            best_comparators = gen_best[1]
            print(f"Gen {generation}: new best {best_fitness} comparators", file=sys.stderr)
        
        if best_fitness <= 100 and generation % 5 == 0:
            pruned = _remove_redundant(best_comparators, max_iterations=30)
            if len(pruned) < best_fitness:
                best_fitness = len(pruned)
                best_comparators = pruned
                print(f"Pruned: new best {best_fitness} comparators", file=sys.stderr)
        
        if best_fitness <= 80:
            for _ in range(3):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                mutated = _mutate(best_comparators, mutation_rate=0.02)
                fit = _evaluate_fitness(mutated)
                if fit < best_fitness:
                    best_fitness = fit
                    best_comparators = mutated
                    print(f"Local search: new best {best_fitness} comparators", file=sys.stderr)
    
    return best_comparators

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    try:
        network = _genetic_algorithm()
        if _verify_network(network):
            return network
    except Exception as e:
        print(f"GA failed: {e}", file=sys.stderr)
    
    return _odd_even_transposition()
