import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(all_pairs)
    
    # Map edge (u, v) to index
    edge_index = {}
    for idx, (u, v) in enumerate(all_pairs):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        
    # Pre-compute all potential K3,3 violations
    # A violation is a set of 6 distinct vertices {a,b,c,d,e,f} such that 
    # {a,b,c} are common neighbors of {d,e,f}.
    # This is equivalent to: there exist 3 vertices d,e,f and 3 vertices a,b,c 
    # such that a,b,c are neighbors of d,e,f.
    # In terms of edges: 
    # Edges (d,a), (d,b), (d,c), (e,a), (e,b), (e,c), (f,a), (f,b), (f,c) must exist.
    # Wait, the definition is "no three vertices have three common neighbours".
    # Let S = {d, e, f}. N(S) = N(d) ∩ N(e) ∩ N(f).
    # If |N(S)| >= 3, then there exist a,b,c in N(S).
    # This implies edges (d,a), (d,b), (d,c), (e,a), (e,b), (e,c), (f,a), (f,b), (f,c) exist.
    # So a K3,3 is formed by partitioning vertices into {d,e,f} and {a,b,c} such that all 9 cross edges exist.
    
    violations = []
    # Iterate over all sets of 3 vertices for the first part {d,e,f}
    for d in range(n):
        for e in range(d+1, n):
            for f in range(e+1, n):
                # Find common neighbors of d, e, f
                # We don't have the graph yet, so we can't filter by existing graph.
                # But the *potential* violation is any set of 3 vertices {a,b,c} distinct from {d,e,f}.
                # If edges (d,a)...(f,c) are all present, it's a violation.
                # There are C(17, 3) choices for {a,b,c} for each {d,e,f}.
                # Total violations: C(20,3) * C(17,3) = 1140 * 680 = 775,200.
                # This is too many to store and check individually in a tight loop.
                
                # Alternative: 
                # A move flips one edge. 
                # We can maintain a count of how many "K3,3 cores" are active.
                # But updating counts for 775k violations per move is too slow.
                
                # Let's refine the violation definition.
                # The constraint is: For any triple {u,v,w}, |N(u) ∩ N(v) ∩ N(w)| < 3.
                # This is equivalent to: No 3 vertices have 3 common neighbors.
                
                # Let's use a different representation for speed.
                # We can pre-compute for each edge (u,v) the list of pairs (w,x) such that 
                # adding (u,v) might create a violation? No, that's complex.
                
                # Let's stick to the bitset approach for validity checking but optimize the search.
                # The current program uses bitsets which is fast.
                # The QUBO idea was to use penalties. 
                # But checking validity is O(n^3) or O(n^2 * bitops).
                # Bitset check is very fast.
                
                pass

    # Actually, the "QUBO" framing with simulated annealing is good, but the bottleneck is the validity check.
    # The current program's `is_valid` is O(n^3) bit operations.
    # n=20, so 1140 triples. Each involves bit AND and bit count.
    # 1140 * (2 ANDs + 1 count) is extremely fast in Python if optimized, but pure Python loops are slow.
    
    # Let's try to speed up the validity check using NumPy or pre-computation.
    # Or, better: Simulated Annealing with a fast "delta" evaluation.
    
    # When we flip an edge (u,v), how does the validity change?
    # A violation is a triple {a,b,c} with 3 common neighbors.
    # Adding edge (u,v) can only create violations where {u,v} is part of the neighborhood intersection.
    # Specifically, if we add (u,v), it affects:
    # 1. Triples containing u or v.
    # 2. The common neighbors of triples containing u or v.
    
    # This delta calculation is complex. 
    # Given the time limit, a robust Simulated Annealing with a *fast* validity check is the key.
    
    # Let's use a NumPy-based validity check.
    # Adjacency matrix A (20x20).
    # Common neighbors of u,v,w is A[u] & A[v] & A[w].
    # We need max over all triples of popcount(A[u] & A[v] & A[w]) < 3.
    
    import numpy as np
    
    def count_edges_np(adj):
        return np.sum(adj) // 2

    def is_valid_np(adj):
        # adj is 20x20 binary matrix
        # We need to check all triples i<j<k
        # This is still a loop in Python if done naively.
        # Vectorization:
        # Let M be a tensor? No, memory.
        # We can iterate over pairs (i,j) and compute common neighbors, then check if any k shares 3.
        
        # Optimized Python loop with bitsets is often faster than NumPy for n=20 due to overhead.
        # Let's stick to bitsets but optimize the loop.
        
        n = 20
        adj_bits = [0] * n
        for i in range(n):
            b = 0
            for j in range(n):
                if adj[i][j]:
                    b |= (1 << j)
            adj_bits[i] = b
            
        for i in range(n):
            ai = adj_bits[i]
            for j in range(i + 1, n):
                cij = ai & adj_bits[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (cij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    # To make SA fast, we need to evaluate validity quickly.
    # Let's try a hybrid: Use bitsets for state, and a pre-computed list of "critical triples" 
    # that are close to being violations.
    
    # Actually, let's just run a tight SA loop with the bitset validity check.
    # The check is ~1000 operations. 100k iterations * 1000 ops = 100M ops. 
    # Python might be too slow for 100k full checks in 120s.
    # 120s / 100k = 1.2ms per check. 
    # A Python loop of 1000 iterations takes ~0.1ms. So it's borderline.
    
    # Let's optimize the validity check further.
    # We can pre-compute the list of triples.
    
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def is_valid_fast(adj_bits):
        for i, j, k in triples:
            # Check if (adj[i] & adj[j] & adj[k]).bit_count() >= 3
            # This is 2 ANDs and 1 bit_count.
            if (adj_bits[i] & adj_bits[j] & adj_bits[k]).bit_count() >= 3:
                return False
        return True

    # Let's test speed of is_valid_fast.
    # 1140 iterations.
    # Should be fast enough for ~10k-50k iterations in 120s.
    
    def adj_to_edges(adj_bits):
        edges = []
        for u in range(n):
            v = adj_bits[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    # Initialize with a random graph or Paley graph
    def generate_paley_19():
        adj_bits = [0] * 20
        residues = set((x * x) % 19 for x in range(1, 10))
        for u in range(19):
            for v in range(u + 1, 19):
                if (v - u) % 19 in residues or (u - v) % 19 in residues:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        return adj_bits

    def generate_random(p):
        adj_bits = [0] * 20
        for i, j in all_pairs:
            if random.random() < p:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
        return adj_bits

    best_adj = None
    best_count = 0
    best_edges = []

    def update_best(adj_bits):
        nonlocal best_adj, best_count, best_edges
        if not is_valid_fast(adj_bits):
            return
        c = sum(adj_bits[i].bit_count() for i in range(n)) // 2
        if c > best_count:
            best_count = c
            best_adj = adj_bits[:]
            best_edges = adj_to_edges(adj_bits)

    random.seed(42)
    
    # Simulated Annealing
    # Start with a dense graph, then repair? 
    # Or start with valid graph and add?
    # SA needs to move between states. If we start with invalid, we need to handle penalties.
    # Let's start with a valid graph (e.g. Paley or repaired random) and try to add edges via SA.
    
    # Strategy:
    # 1. Generate a valid starting graph.
    # 2. Run SA to maximize edges.
    #    State: adj_bits
    #    Move: Flip a random edge.
    #    Energy: -num_edges + penalty * num_violations
    #    But checking num_violations is expensive.
    #    
    # Alternative: 
    #    Only accept moves that maintain validity (greedy hill climbing) but use random restarts and swaps to escape local optima.
    #    This is what the current program does.
    #    
    # To improve: 
    #    Use a "repair" phase. If a move creates a violation, try to fix it by removing another edge.
    #    Or, use a penalty-based SA where we allow invalid states but penalize them heavily.
    
    def count_violations(adj_bits):
        count = 0
        for i, j, k in triples:
            if (adj_bits[i] & adj_bits[j] & adj_bits[k]).bit_count() >= 3:
                count += 1
        return count

    # Penalty SA
    # E = -edges + P * violations
    # If P is large enough, we prefer valid graphs.
    
    def get_energy(adj_bits, P):
        edges = sum(adj_bits[i].bit_count() for i in range(n)) // 2
        viols = count_violations(adj_bits)
        return -edges + P * viols

    # Let's run SA with penalty.
    # This allows the search to jump over barriers.
    
    P = 100  # Penalty weight
    
    def simulated_annealing(initial_adj, time_limit):
        current_adj = initial_adj[:]
        current_energy = get_energy(current_adj, P)
        
        best_adj_local = current_adj[:]
        best_energy_local = current_energy
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        # Pre-compute all possible moves (flips)
        # There are 190 moves.
        
        while time.time() < time_limit and T > T_min:
            # Pick a random edge to flip
            u, v = random.choice(all_pairs)
            
            old_u = current_adj[u]
            old_v = current_adj[v]
            
            # Flip
            if current_adj[u] & (1 << v):
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
            new_energy = get_energy(current_adj, P)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy_local:
                    best_energy_local = new_energy
                    best_adj_local = current_adj[:]
            else:
                # Revert
                current_adj[u] = old_u
                current_adj[v] = old_v
                
            T *= cooling
            
        return best_adj_local

    # Run multiple restarts
    num_restarts = 10
    for s in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if s == 0:
            adj = generate_paley_19()
        else:
            adj = generate_random(random.uniform(0.4, 0.8))
            
        # Make it valid first to start SA in a good basin
        # Simple repair: remove edges causing violations
        # This is expensive. 
        # Instead, start SA from this state. The penalty will drive it to validity.
        
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            continue
            
        sa_adj = simulated_annealing(adj, time.time() + remaining * 0.8)
        
        # Post-process: Greedy add edges to the best SA result
        # This ensures we are in a local optimum of the valid subspace
        
        # Check if SA result is valid
        if is_valid_fast(sa_adj):
            # Greedy add
            improved = True
            while improved:
                improved = False
                for u, v in all_pairs:
                    if sa_adj[u] & (1 << v):
                        continue
                    old_u, old_v = sa_adj[u], sa_adj[v]
                    sa_adj[u] |= (1 << v)
                    sa_adj[v] |= (1 << u)
                    if is_valid_fast(sa_adj):
                        improved = True
                        break
                    else:
                        sa_adj[u] = old_u
                        sa_adj[v] = old_v
            
            update_best(sa_adj)
        else:
            # If SA didn't find a valid graph, try to repair it
            # Or just skip
            pass
            
    # Final fallback
    if not best_edges:
        # Return a simple valid graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 2 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Check validity, if not, return path
        if is_valid_fast(adj):
            return adj_to_edges(adj)
        return [[i, i + 1] for i in range(n - 1)]

    return best_edges
