import numpy as np
import itertools
import time
import random
from collections import deque

D = 11

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Convert subsets to bitmasks for fast intersection checks
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)
        
    # Precompute adjacency list and degree
    # i and j conflict if they share exactly 3 elements
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            inter = (masks[i] & masks[j])
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute degrees for weighted greedy
    degrees = [len(adj[i]) for i in range(n)]
    
    def greedy_mis_weighted(seed_val):
        """Weighted greedy: pick node with fewest conflicts among remaining."""
        rng = random.Random(seed_val)
        available = set(range(n))
        is_set = set()
        while available:
            # Pick the node with minimum degree in the remaining graph
            # Add small random tie-breaking
            best_node = None
            best_score = float('inf')
            for v in available:
                conflicts = len(adj[v] & is_set)
                # We want nodes with no conflicts; among those, prefer low degree
                if conflicts == 0:
                    score = degrees[v] + rng.random() * 0.01
                else:
                    score = float('inf')
                if score < best_score:
                    best_score = score
                    best_node = v
            
            if best_node is None or best_score == float('inf'):
                break
            
            is_set.add(best_node)
            available.discard(best_node)
            # Remove neighbors of best_node from available
            for nb in adj[best_node]:
                available.discard(nb)
        
        return is_set

    def greedy_mis_random(seed_val):
        """Random order greedy."""
        rng = random.Random(seed_val)
        order = list(range(n))
        rng.shuffle(order)
        is_set = set()
        for v in order:
            if not (adj[v] & is_set):
                is_set.add(v)
        return is_set

    def local_search_add(is_set, rng):
        """Try to add any node that doesn't conflict."""
        is_set = set(is_set)
        while True:
            improved = False
            for v in range(n):
                if v not in is_set and not (adj[v] & is_set):
                    is_set.add(v)
                    improved = True
            if not improved:
                break
        return is_set

    def local_search_swap(is_set, rng, max_iter=50):
        """Try to swap: remove one node to allow adding two or more."""
        is_set = set(is_set)
        for _ in range(max_iter):
            improved = False
            for v in list(is_set):
                if time.time() > deadline:
                    return is_set
                trial = is_set - {v}
                added = 0
                for u in range(n):
                    if u not in trial and not (adj[u] & trial):
                        trial.add(u)
                        added += 1
                if added >= 2:
                    is_set = trial
                    improved = True
                    break
            if not improved:
                break
        return is_set

    def local_search_2swap(is_set, rng, max_iter=30):
        """Try to remove two nodes and add three or more."""
        is_set = set(is_set)
        for _ in range(max_iter):
            improved = False
            members = list(is_set)
            for i in range(len(members)):
                if time.time() > deadline:
                    return is_set
                for j in range(i+1, len(members)):
                    v1, v2 = members[i], members[j]
                    trial = is_set - {v1, v2}
                    added = 0
                    for u in range(n):
                        if u not in trial and not (adj[u] & trial):
                            trial.add(u)
                            added += 1
                    if added >= 3:
                        is_set = trial
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return is_set

    def branch_and_bound_mis(adj, n, time_limit=30):
        """Branch and bound for Maximum Independent Set with greedy coloring upper bound."""
        start_time = time.time()
        
        # Precompute neighbor lists as sorted arrays for faster access
        adj_lists = [sorted(adj[i]) for i in range(n)]
        
        # State: current independent set, candidates (nodes not in IS and not adjacent to any node in IS)
        # We use a recursive approach with pruning
        
        best_set = set()
        best_size = 0
        node_count = 0
        
        # Greedy coloring for upper bound
        def greedy_color_bound(candidates, adj):
            """Return upper bound on MIS size in the subgraph induced by candidates using greedy coloring."""
            if not candidates:
                return 0
            candidates = sorted(candidates)
            colors = {}
            color_count = 0
            for v in candidates:
                # Find the smallest color not used by neighbors
                neighbor_colors = set()
                for u in adj[v]:
                    if u in colors:
                        neighbor_colors.add(colors[u])
                c = 0
                while c in neighbor_colors:
                    c += 1
                colors[v] = c
                color_count = max(color_count, c + 1)
            return color_count
        
        # Recursive branch and bound
        def bbsolve(is_set, candidates):
            nonlocal best_set, best_size, node_count
            
            if time.time() - start_time > time_limit:
                return
            
            node_count += 1
            if node_count % 10000 == 0 and time.time() - start_time > time_limit:
                return
            
            if not candidates:
                if len(is_set) > best_size:
                    best_size = len(is_set)
                    best_set = set(is_set)
                return
            
            # Upper bound: current IS size + greedy coloring bound on candidates
            ub = len(is_set) + greedy_color_bound(candidates, adj)
            if ub <= best_size:
                return
            
            # Choose a pivot vertex (heuristic: vertex with max degree in candidate subgraph)
            pivot = max(candidates, key=lambda v: len([u for u in adj[v] if u in candidates]))
            
            # Branch 1: exclude pivot
            bbsolve(is_set, candidates - {pivot})
            
            # Branch 2: include pivot
            new_candidates = candidates - {pivot} - set(adj[pivot])
            bbsolve(is_set | {pivot}, new_candidates)
        
        # Initial upper bound from greedy coloring on all nodes
        all_candidates = set(range(n))
        initial_ub = greedy_color_bound(all_candidates, adj)
        
        bbsolve(set(), all_candidates)
        
        return best_set, best_size

    def simulated_annealing(is_set, seed_val, max_iter=5000):
        """Simulated annealing with swap moves for the independent set problem."""
        rng = random.Random(seed_val)
        is_set = set(is_set)
        current_size = len(is_set)
        best_set = set(is_set)
        best_size = current_size
        
        T = 1.0
        cooling = 0.9995
        start_time = time.time()
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Generate a move: remove one node from IS and add one node not in IS
            if not is_set:
                break
                
            removed = rng.choice(list(is_set))
            # Find candidates that can be added
            candidates = [v for v in range(n) if v not in is_set and v != removed]
            if not candidates:
                continue
            
            added = rng.choice(candidates)
            
            # Check if the move is valid (added doesn't conflict with IS - {removed})
            trial = is_set - {removed}
            if added in trial or (adj[added] & trial):
                continue
            
            # Calculate delta
            delta = 0  # size stays the same for a swap
            # Actually, we might be able to add more after the swap
            # Let's just do the swap and see if we can add more
            
            new_is = trial | {added}
            # Try to greedily add more
            for v in range(n):
                if v not in new_is and not (adj[v] & new_is):
                    new_is.add(v)
            
            new_size = len(new_is)
            
            if new_size >= current_size:
                is_set = new_is
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(is_set)
            else:
                # Accept with probability exp(-delta/T) where delta = current_size - new_size
                if rng.random() < np.exp(-(current_size - new_size) / T):
                    is_set = new_is
                    current_size = new_size
            
            T *= cooling
        
        return best_set

    # First, get a good initial solution using greedy + local search
    best_is = set()
    best_size = 0
    
    # Try multiple greedy strategies
    for seed in range(20):
        if time.time() > deadline:
            break
        if seed % 3 == 0:
            is_set = greedy_mis_weighted(seed)
        else:
            is_set = greedy_mis_random(seed)
        
        is_set = local_search_add(is_set, random.Random(seed))
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
    
    # Apply local search to the best
    if time.time() < deadline - 30:
        best_is = local_search_swap(best_is, random.Random(0), max_iter=30)
        best_size = len(best_is)
    
    if time.time() < deadline - 20:
        best_is = local_search_2swap(best_is, random.Random(0), max_iter=20)
        best_size = len(best_is)
    
    # Try branch and bound to find optimal or near-optimal solution
    if time.time() < deadline - 15:
        bnb_set, bnb_size = branch_and_bound_mis(adj, n, time_limit=10)
        if bnb_size > best_size:
            best_size = bnb_size
            best_is = bnb_set
    
    # Try simulated annealing from the best solution
    if time.time() < deadline - 10:
        sa_set = simulated_annealing(best_is, 42, max_iter=2000)
        if len(sa_set) > best_size:
            best_size = len(sa_set)
            best_is = sa_set
    
    # Final local search on the best
    best_is = local_search_add(best_is, random.Random(0))
    if time.time() < deadline - 5:
        best_is = local_search_swap(best_is, random.Random(0), max_iter=20)
    
    best_size = len(best_is)
    
    # Build vectors
    vectors = []
    for idx in best_is:
        s = subsets[idx]
        # For each selected subset, add all 16 sign variations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
