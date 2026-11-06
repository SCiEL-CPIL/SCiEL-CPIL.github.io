import random
import time
import math
import itertools
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s limit (150s hard kill)
    
    n = 16
    
    # Precompute: all 8-vertex subsets and their edge bit positions
    # We need to check Q3-freeness for each 8-vertex induced subgraph
    q3_edges_local = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i,j)] = idx
            idx += 1
    
    # Generate all distinct Q3 masks (as 28-bit integers)
    q3_mask_set = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            pu, pv = perm[u], perm[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << bit_index_map[(pu, pv)])
        q3_mask_set.add(mask)
    q3_masks = list(q3_mask_set)
    
    # Build lookup table: for each 28-bit mask, does it contain a Q3?
    # Use subset enumeration: for each Q3 mask, mark all supersets
    full_mask = (1 << 28) - 1
    lookup = bytearray(1 << 28)
    for qm in q3_masks:
        Z = full_mask ^ qm
        s = 0
        while True:
            lookup[qm | s] = 1
            if s == Z:
                break
            s = (s - Z) & Z
    
    if time.time() - start_time > time_limit:
        return []
    
    # Precompute all C(16,8) = 12870 subsets and their edge-to-bit mappings
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)
    
    # For each subset, store mapping from (u,v) -> bit index
    subset_edge_bits = []  # list of dicts
    # For each edge (u,v), store which subsets contain it and what bit it maps to
    edge_to_subsets = [[[] for _ in range(n)] for _ in range(n)]
    
    for si, combo in enumerate(subsets):
        info = {}
        for i in range(8):
            for j in range(i+1, 8):
                u, v = combo[i], combo[j]
                b = bit_index_map[(i, j)]
                info[(u,v)] = b
                info[(v,u)] = b
                edge_to_subsets[u][v].append((si, b))
                edge_to_subsets[v][u].append((si, b))
        subset_edge_bits.append(info)
    
    if time.time() - start_time > time_limit:
        return []
    
    # Helper: compute induced masks and violation count for a given adjacency
    def compute_masks_and_violations(adj):
        masks = [0] * num_subsets
        for si, combo in enumerate(subsets):
            mask = 0
            info = subset_edge_bits[si]
            for i in range(8):
                u = combo[i]
                for j in range(i+1, 8):
                    v = combo[j]
                    if (adj[u] >> v) & 1:
                        mask |= (1 << info[(u,v)])
            masks[si] = mask
        violations = sum(1 for m in masks if lookup[m])
        return masks, violations
    
    # Helper: count edges
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    # === LABEL-BASED GENETIC ALGORITHM ===
    # Each vertex gets a 4-bit label. Edge (u,v) exists iff Hamming distance
    # of labels is in the allowed set.
    # We optimize: (1) the labeling (permutation of 0..15), (2) the allowed distances.
    
    # There are 4 possible Hamming distances: 1, 2, 3, 4
    # Allowed distance sets: 16 possible subsets of {1,2,3,4}
    # But we want dense graphs, so we'll focus on larger sets
    
    # For a given labeling and allowed distances, build the graph
    def build_graph_from_labels(labels, allowed_dists):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                d = bin(labels[i] ^ labels[j]).count('1')
                if d in allowed_dists:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    # Precompute Hamming distances between all pairs of 4-bit labels
    hamming = [[0]*16 for _ in range(16)]
    for a in range(16):
        for b in range(16):
            hamming[a][b] = bin(a ^ b).count('1')
    
    # Quick check: does a graph defined by labels and allowed_dists contain Q3?
    def is_q3_free(labels, allowed_dists):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                d = hamming[labels[i]][labels[j]]
                if d in allowed_dists:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        masks, viol = compute_masks_and_violations(adj)
        return viol == 0, count_edges(adj)
    
    # Population of (labeling, allowed_dists)
    # Labeling is a permutation of 0..15
    # allowed_dists is a frozenset of distances from {1,2,3,4}
    
    # Initial population
    pop_size = 40
    population = []
    
    # Seed with some structured labelings
    # 1. Identity: labels[i] = i
    # 2. Random permutations
    # 3. Grouped labelings
    
    # Try all 16 allowed distance sets with a few labelings
    all_dist_sets = [frozenset(d) for d in itertools.combinations([1,2,3,4], 1)]
    all_dist_sets += [frozenset(d) for d in itertools.combinations([1,2,3,4], 2)]
    all_dist_sets += [frozenset(d) for d in itertools.combinations([1,2,3,4], 3)]
    all_dist_sets += [frozenset([1,2,3,4])]
    
    # Start with identity labeling and various distance sets
    identity_labels = list(range(16))
    for ds in all_dist_sets:
        population.append((identity_labels[:], ds))
    
    # Add random permutations with dense distance sets
    for _ in range(pop_size - len(population)):
        labels = list(range(16))
        random.shuffle(labels)
        # Prefer larger distance sets
        k = random.choice([2, 3, 3, 4])
        ds = frozenset(random.sample([1,2,3,4], k))
        population.append((labels, ds))
    
    # Evaluate initial population
    def evaluate(individual):
        labels, ds = individual
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                d = hamming[labels[i]][labels[j]]
                if d in ds:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        masks, viol = compute_masks_and_violations(adj)
        ec = count_edges(adj)
        # Fitness: minimize violations, maximize edges
        # If violations > 0, heavy penalty
        fitness = ec - viol * 10000
        return fitness, viol, ec, adj
    
    pop_fitness = []
    for ind in population:
        f, v, e, adj = evaluate(ind)
        pop_fitness.append((f, v, e, adj))
    
    best_f = max(x[0] for x in pop_fitness)
    best_v = min(x[1] for x in pop_fitness if x[0] == best_f)
    best_e = max(x[2] for x in pop_fitness if x[0] == best_f)
    best_adj = [x[3] for x in pop_fitness if x[0] == best_f][0]
    
    if time.time() - start_time > time_limit:
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (best_adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Genetic algorithm loop
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    elite_count = 5
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T
        
        # Select parents (tournament selection)
        def tournament_select(k=3):
            indices = random.sample(range(len(population)), k)
            best_idx = max(indices, key=lambda i: pop_fitness[i][0])
            return population[best_idx][:]
        
        # Crossover and mutation to produce offspring
        new_population = []
        
        # Elitism: keep best individuals
        sorted_indices = sorted(range(len(population)), key=lambda i: pop_fitness[i][0], reverse=True)
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]][:])
        
        # Generate rest via crossover/mutation
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit - 5:
                break
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            labels1, ds1 = parent1
            labels2, ds2 = parent2
            
            # Crossover labels (uniform crossover)
            child_labels = []
            for i in range(n):
                if random.random() < 0.5:
                    child_labels.append(labels1[i])
                else:
                    child_labels.append(labels2[i])
            
            # Ensure it's a permutation: fix duplicates
            # If not a permutation, repair by swapping
            seen = set()
            for i in range(n):
                if child_labels[i] in seen:
                    # Find a missing value and swap
                    missing = [v for v in range(16) if v not in seen]
                    if missing:
                        child_labels[i] = missing[0]
                        seen.add(child_labels[i])
                    else:
                        child_labels[i] = i
                        seen.add(child_labels[i])
                else:
                    seen.add(child_labels[i])
            
            # Crossover distance sets
            if random.random() < 0.5:
                child_ds = ds1
            elif random.random() < 0.5:
                child_ds = ds2
            else:
                # Union or intersection
                if random.random() < 0.5:
                    child_ds = frozenset(ds1 | ds2)
                else:
                    child_ds = frozenset(ds1 & ds2)
                    if not child_ds:
                        child_ds = frozenset([random.choice([1,2,3,4])])
            
            # Mutation
            if random.random() < 0.3:
                # Mutate labels: swap two positions
                i, j = random.sample(range(n), 2)
                child_labels[i], child_labels[j] = child_labels[j], child_labels[i]
            
            if random.random() < 0.2:
                # Mutate distance set: add or remove a distance
                d = random.choice([1,2,3,4])
                if d in child_ds and len(child_ds) > 1:
                    child_ds = frozenset(child_ds - {d})
                elif d not in child_ds:
                    child_ds = frozenset(child_ds | {d})
            
            new_population.append((child_labels, child_ds))
        
        # Evaluate new population
        new_pop_fitness = []
        for ind in new_population:
            if time.time() - start_time > time_limit - 3:
                f, v, e, adj = evaluate(ind)
                new_pop_fitness.append((f, v, e, adj))
                continue
            f, v, e, adj = evaluate(ind)
            new_pop_fitness.append((f, v, e, adj))
        
        # Replace population
        population = new_population
        pop_fitness = new_pop_fitness
        
        # Update best
        for f, v, e, adj in pop_fitness:
            if f > best_f or (f == best_f and e > best_e):
                best_f = f
                best_v = v
                best_e = e
                best_adj = adj[:]
        
        T *= cooling
    
    # Final: try to improve best_adj with local edge search
    # Quick local search on the best solution
    masks, violations = compute_masks_and_violations(best_adj)
    ec = count_edges(best_adj)
    
    # Try adding edges that don't create violations
    improved = False
    for _ in range(200):  # limited passes
        if time.time() - start_time > time_limit - 2:
            break
        improved_this_pass = False
        for u in range(n):
            for v in range(u+1, n):
                if (best_adj[u] >> v) & 1:
                    continue
                # Check if adding edge (u,v) creates violations
                delta_v = 0
                for si, b in edge_to_subsets[u][v]:
                    old_m = masks[si]
                    new_m = old_m | (1 << b)
                    if lookup[new_m] and not lookup[old_m]:
                        delta_v += 1
                if delta_v == 0:
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    for si, b in edge_to_subsets[u][v]:
                        masks[si] |= (1 << b)
                    ec += 1
                    improved_this_pass = True
        if not improved_this_pass:
            break
    
    # Build final edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
