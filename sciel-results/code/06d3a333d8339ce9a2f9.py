import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = graph['n']
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    def build_matrix(signs):
        A = np.zeros((n, n))
        for i in range(n_edges):
            s = signs[i]
            A[u_arr[i], v_arr[i]] = s
            A[v_arr[i], u_arr[i]] = s
        return A
    
    def spectral_radius(signs):
        A = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid(signs):
        try:
            check_lift(graph, signs)
            return True
        except Exception:
            return False
            
    def evaluate(signs):
        if not is_valid(signs):
            return float('inf')
        return spectral_radius(signs)

    start_time = time.time()
    time_limit = 14
    
    # Initialize Population
    pop_size = 50
    population = []
    
    # Add some structured patterns
    population.append(np.ones(n_edges, dtype=int))
    population.append(-np.ones(n_edges, dtype=int))
    
    # Alternating on cycles
    edge_type = np.zeros(n_edges, dtype=int)
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            edge_type[i] = 0
        elif u >= m and v >= m:
            edge_type[i] = 1
        else:
            edge_type[i] = 2
            
    alt1 = np.ones(n_edges, dtype=int)
    # Alternate signs on top cycle
    top_edges = np.where(edge_type == 0)[0]
    alt1[top_edges] = np.where(np.arange(len(top_edges)) % 2 == 0, 1, -1)
    population.append(alt1)
    
    alt2 = np.ones(n_edges, dtype=int)
    bot_edges = np.where(edge_type == 1)[0]
    alt2[bot_edges] = np.where(np.arange(len(bot_edges)) % 2 == 0, 1, -1)
    population.append(alt2)
    
    # Random initializations
    rng = np.random.default_rng(123)
    for _ in range(pop_size - 4):
        population.append(rng.choice([-1, 1], size=n_edges))
        
    best_signs = None
    best_radius = float('inf')
    
    # Evaluate initial population
    for signs in population:
        r = evaluate(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            
    # Genetic Algorithm Loop
    max_generations = 1000
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
            
        # Sort population by fitness
        fitnesses = [evaluate(s) for s in population]
        indices = np.argsort(fitnesses)
        population = [population[i] for i in indices]
        fitnesses = [fitnesses[i] for i in indices]
        
        # Update global best
        if fitnesses[0] < best_radius:
            best_radius = fitnesses[0]
            best_signs = population[0].copy()
            
        # Selection: Keep top half
        elite_size = pop_size // 2
        new_population = [population[i].copy() for i in range(elite_size)]
        
        # Crossover and Mutation to fill the rest
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = rng.integers(0, elite_size)
            p2_idx = rng.integers(0, elite_size)
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Uniform Crossover
            mask = rng.random(n_edges) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            mutation_rate = 0.1
            mut_mask = rng.random(n_edges) < mutation_rate
            child[mut_mask] = -child[mut_mask]
            
            new_population.append(child)
            
        population = new_population
        
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=int)
        
    return best_signs
