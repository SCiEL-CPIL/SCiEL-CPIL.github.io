"""K3,3-free graph on 20 vertices: maximize edges via genetic algorithm + local search."""
import time
import random
import numpy as np

N = 20
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
NUM_PAIRS = len(ALL_PAIRS)  # 190
PAIR_INDEX = {}
for idx, (i, j) in enumerate(ALL_PAIRS):
    PAIR_INDEX[(i, j)] = idx
    PAIR_INDEX[(j, i)] = idx

# Precompute for each pair of vertices (a,b), the list of other vertices
# This is used for common neighbor counting
# Actually, let's use a faster approach with bitmasks

def _check_k33(adj):
    """Check if graph with adjacency matrix adj (N x N bool) contains K3,3.
    A K3,3 exists if there are 3 vertices on one side and 3 on the other
    such that all 9 edges exist. Equivalently, no 3 vertices have 3 common neighbors
    on the opposite side... wait, the task says 'no three vertices have three common neighbours'.
    
    Actually re-reading: "no three vertices have three common neighbours (no K3,3)"
    
    K3,3 has two parts of 3 vertices each, with all 9 cross edges. 
    If vertices a,b,c form one part and d,e,f form the other, then d,e,f are common neighbors of a,b,c.
    So the condition is: for any 3 vertices, they should NOT have 3 or more common neighbors.
    
    Wait, that's not quite right either. In K3,3, the 3 vertices on one side have exactly the 3 vertices 
    on the other side as common neighbors (and possibly others). The key is that there exist 3 vertices 
    that share at least 3 common neighbors, AND those 3 common neighbors are connected to all 3.
    
    Actually, the standard characterization: a graph contains K3,3 as a subgraph iff there exist 
    two disjoint sets of 3 vertices {a,b,c} and {d,e,f} such that every vertex in {a,b,c} is 
    adjacent to every vertex in {d,e,f}.
    
    This is equivalent to: there exist 3 vertices with at least 3 common neighbors, where those 
    common neighbors form an independent set... no, that's not right either.
    
    Let me think again. K3,3 is a subgraph if we can find 6 distinct vertices a,b,c,d,e,f such that 
    a-d, a-e, a-f, b-d, b-e, b-f, c-d, c-e, c-f all exist. The vertices within {a,b,c} and within 
    {d,e,f} don't need to be connected.
    
    So the check is: do there exist two disjoint triples with all 9 cross edges?
    
    Efficient check: For each pair of vertices (u,v), compute common neighbors. If |common(u,v)| >= 3, 
    then for each triple of common neighbors {x,y,z}, check if x,y,z are all connected to a third vertex w 
    that is also a common neighbor of u,v... this gets complicated.
    
    Simpler: iterate over all pairs of disjoint triples. But that's C(20,3)*C(17,3) which is large.
    
    Better: For each vertex v, consider its neighborhood. If the neighborhood contains K3,3... no.
    
    Actually, the most practical check for n=20: 
    For each triple of vertices (a,b,c), count common neighbors. If count >= 3, then we need to check 
    if any 3 of those common neighbors form a valid K3,3 with (a,b,c). But wait, if a,b,c have 3+ common 
    neighbors, that doesn't automatically mean K3,3 exists. We need the 3 common neighbors to be 
    distinct from a,b,c AND we need... actually no. If a,b,c are all adjacent to d,e,f (3 distinct 
    vertices different from a,b,c), then we have K3,3. The edges within {a,b,c} and within {d,e,f} 
    don't matter.
    
    So: K3,3 exists iff there exist 3 distinct vertices a,b,c and 3 distinct vertices d,e,f (all 6 distinct) 
    such that all 9 edges between {a,b,c} and {d,e,f} exist.
    
    Equivalently: there exist 3 vertices a,b,c such that |N(a) ∩ N(b) ∩ N(c)| >= 3, where N(v) is the 
    neighbor set of v, and we need the intersection to contain at least 3 vertices that are not in {a,b,c}.
    
    Wait, if d is a common neighbor of a,b,c, d must be adjacent to a, b, and c. d could be one of a,b,c? 
    No, because a graph is simple (no self-loops), so a is not adjacent to a. So common neighbors of a,b,c 
    are automatically distinct from a,b,c.
    
    So the check is: for all triples (a,b,c), |N(a) ∩ N(b) ∩ N(c)| < 3.
    
    This is O(n^3 * n) = O(n^4) which for n=20 is 20^4 = 160,000 operations. Very fast.
    """
    # adj is NxN boolean matrix
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                # Count common neighbors of a, b, c
                common = adj[a] & adj[b] & adj[c]
                if np.sum(common) >= 3:
                    return False  # K3,3 found
    return True  # No K3,3


def _edge_count(adj):
    return int(np.sum(adj) // 2)


def _fitness(adj):
    """Return edge count if K3,3-free, else negative penalty."""
    if _check_k33(adj):
        return _edge_count(adj)
    else:
        return -1000 + _edge_count(adj)


def _random_graph(seed_rng, density=0.5):
    adj = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            if seed_rng.random() < density:
                adj[i, j] = adj[j, i] = True
    return adj


def _local_search(adj, max_iters=500, rng=None):
    """Try to add edges to improve while maintaining K3,3-free."""
    if rng is None:
        rng = random
    adj = adj.copy()
    current_fitness = _fitness(adj)
    if current_fitness < 0:
        return adj  # Already invalid, skip
    
    for _ in range(max_iters):
        improved = False
        # Try adding each missing edge
        missing = []
        for i in range(N):
            for j in range(i + 1, N):
                if not adj[i, j]:
                    missing.append((i, j))
        rng.shuffle(missing)
        for (i, j) in missing:
            adj[i, j] = adj[j, i] = True
            if _check_k33(adj):
                current_fitness += 1
                improved = True
            else:
                adj[i, j] = adj[j, i] = False
        if not improved:
            break
    return adj


def construct():
    deadline = time.time() + 140  # Safety margin
    
    best_adj = None
    best_fitness = -1
    
    # Start with the seed: K2,18
    seed_adj = np.zeros((N, N), dtype=bool)
    for a in range(2):
        for b in range(2, N):
            seed_adj[a, b] = seed_adj[b, a] = True
    best_adj = seed_adj.copy()
    best_fitness = _edge_count(seed_adj)  # 36 edges
    
    # Try local search from seed
    ls_result = _local_search(seed_adj, max_iters=200)
    ls_fit = _fitness(ls_result)
    if ls_fit > best_fitness:
        best_adj = ls_result.copy()
        best_fitness = ls_fit
    
    if time.time() > deadline:
        return [[int(i), int(j)] for i in range(N) for j in range(i + 1, N) if best_adj[i, j]]
    
    # Genetic algorithm
    pop_size = 40
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    # Initialize population
    population = [best_adj.copy()]
    for _ in range(pop_size - 1):
        density = rng.uniform(0.3, 0.6)
        g = _random_graph(rng, density)
        g = _local_search(g, max_iters=100, rng=rng)
        population.append(g)
    
    for gen in range(200):
        if time.time() > deadline - 10:
            break
        
        # Evaluate fitness
        fitnesses = [_fitness(adj) for adj in population]
        
        # Update best
        max_fit = max(fitnesses)
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_adj = population[fitnesses.index(max_fit)].copy()
        
        # Selection: tournament
        new_population = []
        for _ in range(pop_size):
            # Tournament of 3
            candidates = rng.sample(range(pop_size), 3)
            winner = max(candidates, key=lambda i: fitnesses[i])
            parent1 = population[winner]
            
            candidates2 = rng.sample(range(pop_size), 3)
            winner2 = max(candidates2, key=lambda i: fitnesses[i])
            parent2 = population[winner2]
            
            # Crossover: mix edges
            child = np.zeros((N, N), dtype=bool)
            for i in range(N):
                for j in range(i + 1, N):
                    if parent1[i, j] and parent2[i, j]:
                        child[i, j] = child[j, i] = True
                    elif parent1[i, j] and rng.random() < 0.5:
                        child[i, j] = child[j, i] = True
                    elif parent2[i, j] and rng.random() < 0.5:
                        child[i, j] = child[j, i] = True
                    else:
                        # Try to add if possible
                        if rng.random() < 0.1:
                            child[i, j] = child[j, i] = True
            
            # Mutation: flip random edges
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < 0.05:
                        child[i, j] = child[j, i] = not child[i, j]
            
            # Local search
            child = _local_search(child, max_iters=50, rng=rng)
            new_population.append(child)
        
        population = new_population
    
    # Final local search on best
    final_adj = _local_search(best_adj, max_iters=300)
    final_fit = _fitness(final_adj)
    if final_fit > best_fitness:
        best_adj = final_adj
        best_fitness = final_fit
    
    edges = [[int(i), int(j)] for i in range(N) for j in range(i + 1, N) if best_adj[i, j]]
    return edges
