import time
import random

def construct():
    n = 20
    adj = [0] * n
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
        
    def can_add(u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
            
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if popcount(adj[w] & adj[v]) >= 2:
                return False
            mask_u ^= lsb
            
        # Check neighbors of v
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            if popcount(adj[u] & adj[x]) >= 2:
                return False
            mask_v ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges():
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    def get_edges_list():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def find_violation():
        for u in range(n):
            for v in range(u+1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return u, v
        return None

    def repair_graph():
        """
        Repair the current graph (adj) by removing edges until it is K2,3-free.
        """
        while True:
            viol = find_violation()
            if not viol:
                break
            u, v = viol
            common = adj[u] & adj[v]
            ws = []
            mask = common
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                ws.append(w)
                mask ^= lsb
            
            if not ws:
                break # Should not happen
            
            w = random.choice(ws)
            if random.random() < 0.5:
                remove_edge(u, w)
            else:
                remove_edge(v, w)

    def greedy_fill():
        """
        Try to add all possible missing edges to the current graph (adj).
        """
        added_any = True
        while added_any:
            added_any = False
            for i in range(n):
                for j in range(i+1, n):
                    if not ((adj[i] >> j) & 1):
                        if can_add(i, j):
                            add_edge(i, j)
                            added_any = True

    start_time = time.time()
    deadline = start_time + 115.0 # 5s buffer
    
    best_adj = [0] * n
    best_count = 0
    
    # Genetic Algorithm Parameters
    POP_SIZE = 30
    GENERATIONS = 100 # Will be limited by time
    
    # Initialize Population
    population = []
    for _ in range(POP_SIZE):
        for i in range(n):
            adj[i] = 0
        # Random dense start
        p = random.choice([0.5, 0.6, 0.7, 0.8])
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    add_edge(i, j)
        repair_graph()
        greedy_fill()
        population.append(adj[:])
        
        # Update best
        c = count_edges()
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Evolution Loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        if time.time() > deadline - 5.0:
            break # Reserve time for final polish
        
        # Evaluate fitness
        fitnesses = [count_edges(a) for a in population]
        
        # Update best
        max_fit = max(fitnesses)
        if max_fit > best_count:
            best_count = max_fit
            best_idx = fitnesses.index(max_fit)
            best_adj = population[best_idx][:]
        
        # Selection and Crossover
        new_population = []
        
        # Elitism: Keep the best individual
        if fitnesses:
            best_idx = fitnesses.index(max(fitnesses))
            new_population.append(population[best_idx][:])
        
        while len(new_population) < POP_SIZE:
            # Tournament Selection
            p1 = random.choice(range(POP_SIZE))
            p2 = random.choice(range(POP_SIZE))
            if fitnesses[p1] > fitnesses[p2]:
                parent1 = population[p1]
            else:
                parent1 = population[p2]
                
            p3 = random.choice(range(POP_SIZE))
            p4 = random.choice(range(POP_SIZE))
            if fitnesses[p3] > fitnesses[p4]:
                parent2 = population[p3]
            else:
                parent2 = population[p4]
            
            # Crossover: Uniform (Union)
            for i in range(n):
                adj[i] = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    e1 = (parent1[i] >> j) & 1
                    e2 = (parent2[i] >> j) & 1
                    if e1 or e2:
                        add_edge(i, j)
            
            # Mutation: Randomly flip some edges
            for _ in range(5):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    if (adj[u] >> v) & 1:
                        remove_edge(u, v)
                    else:
                        add_edge(u, v)
            
            # Repair
            repair_graph()
            greedy_fill()
            
            new_population.append(adj[:])
            
        population = new_population
        
        # Occasionally inject a random individual to maintain diversity
        if gen % 10 == 0:
            for i in range(n):
                adj[i] = 0
            p = random.choice([0.5, 0.6, 0.7, 0.8])
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        add_edge(i, j)
            repair_graph()
            greedy_fill()
            population[random.randint(0, POP_SIZE-1)] = adj[:]

    # Final Polish on Best
    for i in range(n):
        adj[i] = best_adj[i]
    
    greedy_fill()
    current_count = count_edges()
    
    # Try a few local swaps
    for _ in range(20):
        if time.time() > deadline:
            break
        existing_edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    existing_edges.append((i, j))
                mask ^= lsb
                
        if not existing_edges:
            break
            
        u, v = random.choice(existing_edges)
        remove_edge(u, v)
        greedy_fill()
        new_count = count_edges()
        
        if new_count > current_count:
            current_count = new_count
            # Keep the new state
        else:
            # Try to restore u-v
            if can_add(u, v):
                add_edge(u, v)
            # If we can't add it back, we are in a new state with same or lower count.
            # If lower, we lost an edge. But we can't easily revert.
            # However, if new_count == current_count, it's neutral.
            # If new_count < current_count, it's worse.
            # Let's assume the risk is low or accept it.
            pass

    # Update best one last time
    final_count = count_edges()
    if final_count > best_count:
        best_count = final_count
        best_adj = adj[:]

    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges
