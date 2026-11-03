import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    
    # Build incompatibility graph
    # incompat_sup[i, j] = 1 if |sup_i ∩ sup_j| >= 3
    sup_sets = [set(s) for s in supports]
    incompat_sup = np.zeros((N_sup, N_sup), dtype=np.int8)
    
    for i in range(N_sup):
        if time.time() > deadline - 10.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat_sup[i, j] = 1
                incompat_sup[j, i] = 1
                
    # Precompute neighbors for faster access
    # neighbors[i] is a set of j such that incompat_sup[i, j] == 1
    neighbors = [set(np.where(incompat_sup[i] == 1)[0]) for i in range(N_sup)]
    
    def greedy_min_degree(available, rng):
        """
        Greedy MIS using minimum degree heuristic.
        available: set of node indices
        rng: random generator for tie-breaking
        """
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 5.0:
                break
            
            # Find node in avail with minimum degree in the induced subgraph on avail
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(neighbors[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            # Pick random from candidates to break symmetry
            u = rng.choice(candidates)
            selected.add(u)
            
            # Remove u and its neighbors from avail
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
            
        return selected

    def local_search(selected, rng):
        """
        Try to improve the independent set using 1-2 swaps.
        """
        selected = set(selected)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try 1-2 swap: remove v, add u1, u2
            # To speed up, we can iterate over v in selected
            for v in list(selected):
                if time.time() > deadline - 5.0:
                    break
                # Nodes compatible with selected \ {v}
                # A node x is compatible with selected \ {v} if x is not in neighbors[y] for any y in selected \ {v}
                # And x != v
                # And x not in selected
                
                # Efficient check:
                # blocked_by_others = union of neighbors[y] for y in selected \ {v}
                # But we can just check candidates.
                
                # Candidates are nodes NOT in selected and NOT v
                # That are not incompatible with any y in selected \ {v}
                
                # Let's find all nodes x such that x is compatible with selected \ {v}
                # This is (All \ selected) \ (Union_{y in selected \ {v}} neighbors[y])
                
                # Since N is small, we can just filter
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    # Check if x is compatible with all y in selected \ {v}
                    # i.e., incompat_sup[x, y] == 0 for all y in selected \ {v}
                    # Equivalent: x is not in neighbors[y] for any y in selected \ {v}
                    # Or: (neighbors[x] & (selected - {v})) is empty
                    
                    if not (neighbors[x] & (selected - {v})):
                        potential_adds.append(x)
                        
                # Now we need to find two nodes u1, u2 in potential_adds
                # such that they are compatible with each other (incompat_sup[u1, u2] == 0)
                # If found, we can swap v for u1, u2
                
                # Sort potential_adds by something? Or just check pairs?
                # If len(potential_adds) is large, checking all pairs is O(K^2).
                # K can be up to 330. 330^2 = 100k. 330 * 100k = 33M ops. Might be too slow for many iterations.
                
                # Optimization:
                # We only need ONE pair.
                # We can iterate u1 in potential_adds, and check if there is any u2 in potential_adds compatible with u1.
                
                found = False
                for i in range(len(potential_adds)):
                    u1 = potential_adds[i]
                    # Check u2 in potential_adds[i+1:]
                    for j in range(i + 1, len(potential_adds)):
                        u2 = potential_adds[j]
                        if incompat_sup[u1, u2] == 0:
                            # Swap
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
            if improved:
                # Restart local search loop to catch new opportunities
                continue
        return selected

    best_iset = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Random subset to start? Or full set?
        # Full set with min-degree greedy is good.
        # Add some randomness: remove a random subset of nodes initially?
        # Or just use the min-degree greedy on the full set.
        
        avail = set(range(N_sup))
        
        # Add slight randomness to the min-degree choice
        # The rng.choice in greedy_min_degree handles this.
        
        iset = greedy_min_degree(avail, rng)
        
        # Local search
        iset = local_search(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = iset.copy()
            
        # Optional: Try to perturb and search again?
        # If we have time, we can do simulated annealing-like swaps.
        
    if best_iset is None:
        return np.array([[]], dtype=np.int8)
        
    # Generate vectors
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
