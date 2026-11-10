import time
import random
import itertools
import sys

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6+ vertices.
    Maximizes n. Uses a genetic algorithm for 21-vertex search,
    with fallback to 20 and then to known small valid graphs.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    def count_violations_fast(adj, n, subsets_6, subsets_7):
        """
        Returns number of regular induced subgraphs of size 6 and 7.
        Uses bit operations for speed.
        """
        count = 0
        for subset in subsets_6:
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = []
            for v in subset:
                degs.append((adj[v] & mask).bit_count())
            if len(set(degs)) == 1:
                count += 1
        for subset in subsets_7:
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = []
            for v in subset:
                degs.append((adj[v] & mask).bit_count())
            if len(set(degs)) == 1:
                count += 1
        return count

    def is_fully_valid(adj, n):
        """
        Checks if graph has no regular induced subgraph of size >= 6.
        Checks sizes 8..n (sizes 6 and 7 are assumed to be zero by caller).
        """
        for size in range(8, n + 1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = []
                for v in subset:
                    degs.append((adj[v] & mask).bit_count())
                if len(set(degs)) == 1:
                    return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def random_adj(n):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def flip_edge(adj, i, j):
        """Flip the edge between i and j in place."""
        if adj[i] & (1 << j):
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

    # Precompute subsets for n=21 and n=20
    subsets_6_21 = list(itertools.combinations(range(21), 6))
    subsets_7_21 = list(itertools.combinations(range(21), 7))
    subsets_6_20 = list(itertools.combinations(range(20), 6))
    subsets_7_20 = list(itertools.combinations(range(20), 7))

    # Genetic Algorithm for n=21
    def genetic_search(n, subsets_6, subsets_7, pop_size=100, generations=50):
        if not time_ok():
            return None

        # Initialize population
        population = []
        for _ in range(pop_size):
            adj = random_adj(n)
            viol = count_violations_fast(adj, n, subsets_6, subsets_7)
            population.append((adj, viol))
        
        # Sort by violations
        population.sort(key=lambda x: x[1])
        
        best_adj, best_viol = population[0]
        
        # Check if best is fully valid
        if best_viol == 0 and is_fully_valid(best_adj, n):
            return best_adj

        for gen in range(generations):
            if not time_ok():
                break
            
            new_population = []
            
            # Elitism: keep top 10
            for adj, viol in population[:10]:
                new_population.append((adj[:], viol))
            
            # Fill rest with crossover and mutation
            while len(new_population) < pop_size:
                if not time_ok():
                    break
                
                # Tournament selection
                def select():
                    candidates = random.sample(population, 3)
                    return min(candidates, key=lambda x: x[1])[0]
                
                parent1 = select()
                parent2 = select()
                
                # Crossover: uniform crossover
                child = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            if parent1[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                        else:
                            if parent2[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                
                # Mutation: flip a few random edges
                for _ in range(random.randint(1, 5)):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        flip_edge(child, i, j)
                
                viol = count_violations_fast(child, n, subsets_6, subsets_7)
                new_population.append((child, viol))
            
            population = new_population
            population.sort(key=lambda x: x[1])
            
            # Check best
            best_adj, best_viol = population[0]
            if best_viol == 0 and is_fully_valid(best_adj, n):
                return best_adj
            
            # Local search on best individual
            if best_viol < 100:  # Only if close
                adj = best_adj[:]
                for step in range(200):
                    if not time_ok():
                        break
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    flip_edge(adj, i, j)
                    new_viol = count_violations_fast(adj, n, subsets_6, subsets_7)
                    if new_viol <= best_viol:
                        best_viol = new_viol
                        if new_viol == 0 and is_fully_valid(adj, n):
                            return adj
                    else:
                        flip_edge(adj, i, j)  # revert
                        best_viol = count_violations_fast(adj, n, subsets_6, subsets_7)
        
        return None

    # Try n=21 with genetic algorithm
    result = genetic_search(21, subsets_6_21, subsets_7_21)
    if result is not None:
        return {'n': 21, 'edges': adj_to_edges(result, 21)}

    # Try n=20 with genetic algorithm
    if time_ok():
        result = genetic_search(20, subsets_6_20, subsets_7_20)
        if result is not None:
            return {'n': 20, 'edges': adj_to_edges(result, 20)}

    # Fallback: try local search for n=20
    if time_ok():
        for restart in range(20):
            if not time_ok():
                break
            adj = random_adj(20)
            viol = count_violations_fast(adj, 20, subsets_6_20, subsets_7_20)
            for step in range(500):
                if not time_ok():
                    break
                i = random.randint(0, 19)
                j = random.randint(0, 19)
                if i == j:
                    continue
                flip_edge(adj, i, j)
                new_viol = count_violations_fast(adj, 20, subsets_6_20, subsets_7_20)
                if new_viol <= viol:
                    viol = new_viol
                    if viol == 0 and is_fully_valid(adj, 20):
                        return {'n': 20, 'edges': adj_to_edges(adj, 20)}
                else:
                    flip_edge(adj, i, j)

    # Known valid small graphs
    # n=5 empty graph
    return {'n': 5, 'edges': []}

if __name__ == '__main__':
    result = construct()
    print(result)
