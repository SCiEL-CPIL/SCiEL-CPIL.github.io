# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_routes(n=10):
    """Precompute which edges each demand uses for CW and CCW routing."""
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        
        cw_set = set(cw)
        ccw = [e for e in range(n) if e not in cw_set]
        ccw_edges.append(ccw)
    
    return cw_edges, ccw_edges, demands


def _compute_loads(routing, cw_edges, ccw_edges, demands):
    """Compute edge loads given a routing vector."""
    n = 10
    D = len(demands)
    loads = np.zeros(n)
    
    for d in range(D):
        s, t, w = demands[d]
        if routing[d] >= 0.5:
            for e in ccw_edges[d]:
                loads[e] += w
        else:
            for e in cw_edges[d]:
                loads[e] += w
    
    return loads


def _compute_max_load(routing, cw_edges, ccw_edges, demands):
    """Compute the maximum edge load for a given routing."""
    loads = _compute_loads(routing, cw_edges, ccw_edges, demands)
    return np.max(loads), loads


def _local_search(routing, cw_edges, ccw_edges, demands, deadline):
    """Single-flip local search."""
    D = len(demands)
    current_max_load, current_loads = _compute_max_load(routing, cw_edges, ccw_edges, demands)
    
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for d in range(D):
            trial_routing = routing.copy()
            trial_routing[d] = 1.0 - trial_routing[d]
            
            trial_max_load, _ = _compute_max_load(trial_routing, cw_edges, ccw_edges, demands)
            
            if trial_max_load < current_max_load:
                routing = trial_routing
                current_max_load = trial_max_load
                improved = True
    
    return routing, current_max_load


def _pair_local_search(routing, cw_edges, ccw_edges, demands, deadline):
    """Pair-flip local search."""
    D = len(demands)
    current_max_load, _ = _compute_max_load(routing, cw_edges, ccw_edges, demands)
    
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for d1 in range(D):
            if time.time() > deadline - 5:
                break
            for d2 in range(d1 + 1, D):
                trial_routing = routing.copy()
                trial_routing[d1] = 1.0 - trial_routing[d1]
                trial_routing[d2] = 1.0 - trial_routing[d2]
                
                trial_max_load, _ = _compute_max_load(trial_routing, cw_edges, ccw_edges, demands)
                
                if trial_max_load < current_max_load:
                    routing = trial_routing
                    current_max_load = trial_max_load
                    improved = True
                    break
            if time.time() > deadline - 5:
                break
    
    return routing, current_max_load


def _simulated_annealing(routing, cw_edges, ccw_edges, demands, deadline):
    """Simulated annealing for escaping local optima."""
    D = len(demands)
    current_max_load, _ = _compute_max_load(routing, cw_edges, ccw_edges, demands)
    
    best_routing = routing.copy()
    best_max_load = current_max_load
    
    T = 5.0  # initial temperature
    T_min = 0.01
    cooling = 0.95
    max_iter = 10000
    
    for i in range(max_iter):
        if time.time() > deadline - 5:
            break
        if T < T_min:
            break
        
        # Pick a random demand to flip
        d = np.random.randint(D)
        trial_routing = routing.copy()
        trial_routing[d] = 1.0 - trial_routing[d]
        
        trial_max_load, _ = _compute_max_load(trial_routing, cw_edges, ccw_edges, demands)
        delta = trial_max_load - current_max_load
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            routing = trial_routing
            current_max_load = trial_max_load
            
            if current_max_load < best_max_load:
                best_routing = routing.copy()
                best_max_load = current_max_load
        
        T *= cooling
    
    return best_routing, best_max_load


def _genetic_algorithm(cw_edges, ccw_edges, demands, deadline, pop_size=50, generations=500):
    """Genetic algorithm for global search."""
    D = len(demands)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        individual = np.random.choice([0.0, 1.0], size=D)
        population.append(individual)
    
    best_routing = None
    best_max_load = float('inf')
    
    for gen in range(generations):
        if time.time() > deadline - 5:
            break
        
        # Evaluate fitness
        fitness = []
        for individual in population:
            max_load, _ = _compute_max_load(individual, cw_edges, ccw_edges, demands)
            fitness.append(max_load)
        
        # Update best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_max_load:
            best_max_load = fitness[best_idx]
            best_routing = population[best_idx].copy()
        
        # Selection (tournament)
        new_population = []
        for _ in range(pop_size):
            # Tournament selection
            contestants = np.random.choice(pop_size, size=3, replace=False)
            winner = contestants[np.argmin([fitness[c] for c in contestants])]
            new_population.append(population[winner].copy())
        
        # Crossover
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < 0.8:  # crossover rate
                point = np.random.randint(1, D)
                child1 = np.concatenate([new_population[i][:point], new_population[i+1][point:]])
                child2 = np.concatenate([new_population[i+1][:point], new_population[i][point:]])
                new_population[i] = child1
                new_population[i+1] = child2
        
        # Mutation
        for i in range(pop_size):
            if np.random.rand() < 0.1:  # mutation rate
                d = np.random.randint(D)
                new_population[i][d] = 1.0 - new_population[i][d]
        
        population = new_population
    
    return best_routing, best_max_load


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using multiple search strategies.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    cw_edges, ccw_edges, demands = _build_routes()
    D = len(demands)
    n = 10
    
    # Best solution found so far
    best_routing = np.zeros(D)
    best_max_load = float('inf')
    
    # Strategy 1: Start with all CW, then local search
    routing1 = np.zeros(D)
    routing1, max_load1 = _local_search(routing1, cw_edges, ccw_edges, demands, deadline)
    if max_load1 < best_max_load:
        best_max_load = max_load1
        best_routing = routing1.copy()
    
    # Strategy 2: Start with all CCW, then local search
    routing2 = np.ones(D)
    routing2, max_load2 = _local_search(routing2, cw_edges, ccw_edges, demands, deadline)
    if max_load2 < best_max_load:
        best_max_load = max_load2
        best_routing = routing2.copy()
    
    # Strategy 3: Random initial solutions with local search
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        routing_rand = np.random.choice([0.0, 1.0], size=D)
        routing_rand, max_load_rand = _local_search(routing_rand, cw_edges, ccw_edges, demands, deadline)
        if max_load_rand < best_max_load:
            best_max_load = max_load_rand
            best_routing = routing_rand.copy()
    
    # Strategy 4: Pair local search on best solution
    if time.time() < deadline - 10:
        routing_pair, max_load_pair = _pair_local_search(best_routing.copy(), cw_edges, ccw_edges, demands, deadline)
        if max_load_pair < best_max_load:
            best_max_load = max_load_pair
            best_routing = routing_pair.copy()
    
    # Strategy 5: Simulated annealing from best solution
    if time.time() < deadline - 10:
        routing_sa, max_load_sa = _simulated_annealing(best_routing.copy(), cw_edges, ccw_edges, demands, deadline)
        if max_load_sa < best_max_load:
            best_max_load = max_load_sa
            best_routing = routing_sa.copy()
        # Polish with local search
        routing_sa, max_load_sa = _local_search(routing_sa, cw_edges, ccw_edges, demands, deadline)
        if max_load_sa < best_max_load:
            best_max_load = max_load_sa
            best_routing = routing_sa.copy()
    
    # Strategy 6: Genetic algorithm
    if time.time() < deadline - 10:
        routing_ga, max_load_ga = _genetic_algorithm(cw_edges, ccw_edges, demands, deadline, pop_size=30, generations=200)
        if max_load_ga < best_max_load:
            best_max_load = max_load_ga
            best_routing = routing_ga.copy()
        # Polish GA result with local search
        routing_ga, max_load_ga = _local_search(routing_ga, cw_edges, ccw_edges, demands, deadline)
        if max_load_ga < best_max_load:
            best_max_load = max_load_ga
            best_routing = routing_ga.copy()
    
    # Final polish: pair local search on the best
    if time.time() < deadline - 5:
        routing_final, max_load_final = _pair_local_search(best_routing.copy(), cw_edges, ccw_edges, demands, deadline)
        if max_load_final < best_max_load:
            best_max_load = max_load_final
            best_routing = routing_final.copy()
    
    return best_routing


# EVOLVE-BLOCK-END
