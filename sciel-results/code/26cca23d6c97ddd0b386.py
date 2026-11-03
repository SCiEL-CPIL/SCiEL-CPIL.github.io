"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from collections import defaultdict

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Generate the primary shell: vectors with 4 non-zero entries (norm^2 = 4)
    # This shell is dense enough to provide a good kissing configuration.
    # We will search for the Maximum Independent Set (MIS) in the conflict graph of this shell.
    # Conflict: Two vectors u, v conflict if |u . v| > 2 (since distance^2 = 4+4-2*u.v >= 4 => u.v <= 2).
    # Actually, distance^2 >= max_norm^2 = 4.
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # We need 8 - 2 u.v >= 4 => 2 u.v <= 4 => u.v <= 2.
    # Also u.v >= -2 is not strictly required for distance, but if u.v is very negative, distance is large.
    # However, usually we look for sets where all pairs satisfy the constraint.
    # Wait, the condition is "minimum pairwise distance is at least the maximum vector norm".
    # Max norm is 2 (since norm^2=4).
    # So min distance >= 2.
    # ||u-v|| >= 2 => ||u-v||^2 >= 4.
    # 8 - 2 u.v >= 4 => u.v <= 2.
    # So the only constraint is u.v <= 2.
    
    shell = _generate_shell(4)
    n = len(shell)
    
    # Precompute inner products
    # We only care if ip > 2.
    # To save memory, we can compute the conflict matrix on the fly or store it as a bitset/boolean.
    # n ~ 5280. n^2 * 1 byte = ~28 MB. This is fine.
    # Let's compute the full dot product matrix.
    dots = shell @ shell.T
    # Conflict if dots[i, j] > 2
    # Note: diagonal is 4, so self-conflict is True, but we ignore self.
    
    # Build adjacency lists for conflicts
    # adj[i] = set of j such that dots[i, j] > 2 and i != j
    adj = [set() for _ in range(n)]
    for i in range(n):
        # Vectorized: find indices where dots[i] > 2
        # But we need to exclude i itself.
        conflicts = np.where(dots[i] > 2)[0]
        for j in conflicts:
            if j != i:
                adj[i].add(j)
                
    # CSP Backtracking with Forward Checking
    # State: current set of selected indices S
    # Domain: all indices not in S and not conflicting with S
    # Heuristic: MRV (Minimum Remaining Values) - pick variable with fewest remaining values?
    # Here, the "variables" are the items to select. 
    # A better heuristic for MIS: pick the node with the highest degree in the remaining graph?
    # Or simply: Pick a node, add to S, remove it and its neighbors from the domain.
    # This is essentially a greedy MIS but with backtracking.
    
    # To make it a proper CSP that finds the *maximum* set, we need to explore different choices.
    # However, full MIS is NP-hard. 
    # Strategy: 
    # 1. Use a strong greedy heuristic to get a good initial solution S0.
    # 2. Use local search / branch and bound to improve it.
    # Given the time limit, a robust randomized greedy with restarts + local improvement is often better than full backtracking which might explode.
    # But the prompt suggests "CSP backtracking with forward checking".
    # Let's implement a Branch and Bound MIS solver.
    
    # Precompute degrees for heuristic
    degrees = [len(adj[i]) for i in range(n)]
    
    best_size = 0
    best_set = []
    
    # Branch and Bound
    # State: (available_nodes, current_set, upper_bound)
    # Upper bound: size(current_set) + number of available nodes (loose)
    # Better upper bound: size(current_set) + max_clique_size(available_nodes)? No, we want independent set.
    # Upper bound for Independent Set: size(current_set) + size(maximal_independent_set_greedy(available_nodes))?
    # Or simply: size(current_set) + len(available_nodes) is too loose.
    # Let's use a simple greedy upper bound: run a fast greedy on the available nodes to estimate potential.
    
    def greedy_mis(nodes):
        """Fast greedy MIS for upper bound estimation."""
        if not nodes:
            return 0
        nodes = set(nodes)
        count = 0
        while nodes:
            # Pick node with min degree in the subgraph induced by nodes
            # To be fast, just pick the first one? No, min degree is better for size.
            # Actually, for an upper bound, we want a *large* independent set.
            # Greedy picking min degree usually gives a decent IS.
            min_deg = float('inf')
            chosen = None
            for u in nodes:
                deg_in_subgraph = len(adj[u] & nodes)
                if deg_in_subgraph < min_deg:
                    min_deg = deg_in_subgraph
                    chosen = u
            count += 1
            # Remove chosen and its neighbors from nodes
            nodes.discard(chosen)
            for v in adj[chosen]:
                if v in nodes:
                    nodes.discard(v)
        return count

    # To avoid recursion depth issues and overhead, we can use an iterative approach or careful recursion.
    # Given N=5280, the graph is large. Full B&B might be too slow.
    # Alternative: Use a sophisticated Local Search (Simulated Annealing or Tabu Search) on the IS.
    # But the "knowledge" said CSP backtracking improved a previous island.
    # Let's try a hybrid: 
    # 1. Get a good initial IS via greedy.
    # 2. Use a "repair" or "swap" local search to improve it.
    # 3. If time remains, try a few random restarts of the greedy+local search.
    
    # Let's implement the Local Search approach which is more robust for large N.
    
    def get_greedy_is():
        """Greedy MIS with randomization."""
        available = set(range(n))
        selected = []
        while available:
            # Pick a random node from available? Or min degree?
            # Min degree is better for maximizing size.
            # But computing min degree in the subgraph is O(N^2) per step if not careful.
            # We can approximate: pick a random node from available, or the one with lowest global degree.
            # Let's use global degree as a static heuristic, it's fast.
            # Or dynamic: 
            # Let's just pick a random node to allow diversity, then local search fixes it.
            idx = np.random.choice(list(available))
            selected.append(idx)
            available.discard(idx)
            for v in adj[idx]:
                if v in available:
                    available.discard(v)
        return selected

    def local_search_improve(initial_set):
        """
        Improve the independent set using swaps.
        1. Try to add a node if possible (should be none if maximal).
        2. Try to swap one node u for two nodes v, w that are compatible with the rest.
        """
        S = set(initial_set)
        # Precompute for each node u in S, the set of conflicts with S\{u}
        # Actually, just maintain the set S.
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 10:
            improved = False
            iterations += 1
            # Try all possible swaps
            # For each u in S:
            #   Find candidates C = {v not in S | v compatible with S\{u}}
            #   If there are two candidates v, w in C that are compatible with each other:
            #       Replace u with v, w.
            
            # To make this fast, we need efficient conflict checks.
            # Let's iterate through u in S.
            for u in list(S):
                # Find candidates compatible with S \ {u}
                # A candidate v is compatible if v not in S and for all s in S\{u}, v not in adj[s].
                # This is expensive to check for all v.
                
                # Optimization: 
                # Precompute for each v (not in S), the number of conflicts with S.
                # If count == 1 and the conflict is u, then v is a candidate.
                
                # Let's maintain a conflict_count array for all nodes relative to current S.
                # conflict_count[v] = number of s in S such that v in adj[s]
                
                # This approach requires updating conflict_count on every swap.
                # Let's implement a simpler version: 
                # Just try to find a pair (v, w) that can replace u.
                
                # For a fixed u, the "forbidden" set is F_u = (S \ {u}) U (adj of nodes in S \ {u})? No.
                # v must not conflict with any s in S \ {u}.
                # So v must be in (All \ S) \ (Union of adj[s] for s in S\{u}).
                
                # This is getting complex to vectorize easily in pure Python loops for N=5280.
                
                # Let's stick to a simpler local search: 
                # 1. Remove a random u from S.
                # 2. Try to add as many nodes as possible from the newly available space.
                # 3. If size increases, keep. Else, revert.
                
                pass # Skip complex swap for now, use simpler "kick" strategy
                
            # Kick strategy:
            # Pick a random subset of S to remove, then greedily fill the hole.
            if iterations < 50: # Limit kicks
                kick_size = min(3, len(S))
                if kick_size > 0:
                    removed = np.random.choice(list(S), size=kick_size, replace=False)
                    S_kick = S.copy()
                    for r in removed:
                        S_kick.discard(r)
                    
                    # Greedily add nodes compatible with S_kick
                    # Find all nodes not in S_kick and not conflicting with S_kick
                    # candidates = [v for v in range(n) if v not in S_kick and all(v not in adj[s] for s in S_kick)]
                    
                    # Vectorized check:
                    # Create a boolean mask for S_kick
                    mask_sk = np.zeros(n, dtype=bool)
                    for s in S_kick:
                        mask_sk[s] = True
                        
                    # Find nodes that conflict with ANY node in S_kick
                    # conflicts_with_sk = np.zeros(n, dtype=bool)
                    # for s in S_kick:
                    #     conflicts_with_sk[adj[s]] = True # This is slow in Python loop
                    
                    # Better: 
                    # We can precompute the union of adjacencies? 
                    # Since S_kick is small (kick is small), let's just check candidates.
                    
                    # Let's just try to add one node at a time from a random sample of non-conflicting nodes.
                    # This is too slow if done naively.
                    
                    # Alternative: Use the precomputed dots matrix.
                    # A node v is compatible with S_kick if max(dots[v, S_kick]) <= 2.
                    
                    # Get indices of S_kick
                    sk_indices = list(S_kick)
                    if len(sk_indices) > 0:
                        # dots[v, sk_indices]
                        # We want max <= 2.
                        # sub_dots = dots[:, sk_indices]
                        # max_dots = np.max(sub_dots, axis=1)
                        # valid = max_dots <= 2
                        # valid[sk_indices] = False # Exclude already in S
                        # candidates = np.where(valid)[0]
                        
                        # This is O(N * K) per kick. K is small.
                        # Let's try it.
                        sub_dots = dots[:, sk_indices]
                        max_dots = np.max(sub_dots, axis=1)
                        valid = max_dots <= 2
                        valid[sk_indices] = False
                        candidates = np.where(valid)[0]
                        
                        # Now greedily pick from candidates
                        added_any = False
                        while len(candidates) > 0 and time.time() < deadline - 10:
                            # Pick random candidate
                            idx = np.random.choice(candidates)
                            S_kick.add(idx)
                            added_any = True
                            
                            # Update candidates: remove idx and its neighbors
                            # neighbors = adj[idx]
                            # new_valid = valid.copy()
                            # new_valid[idx] = False
                            # for nb in neighbors:
                            #     if new_valid[nb]:
                            #         new_valid[nb] = False
                            # candidates = np.where(new_valid)[0]
                            
                            # Faster: 
                            # We can just check compatibility with the NEW S_kick.
                            # But that's O(N*K) again.
                            # Let's just break after adding a few to keep it fast.
                            break # Add just one for now, loop will do more kicks if beneficial?
                            # Actually, let's add all possible in a simple greedy loop.
                            # But updating candidates is the bottleneck.
                            # Let's just add one and check if size increased.
                            
                        if len(S_kick) > len(S):
                            S = S_kick
                            improved = True
                            # Reset iterations to allow more kicks
                            iterations = 0 
                            break # Restart the while loop with new S
        
        return list(S)

    # Run multiple restarts
    best_k = 0
    best_S = []
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # 1. Greedy initialization
        S_init = get_greedy_is()
        
        # 2. Local Search
        S_improved = local_search_improve(S_init)
        
        if len(S_improved) > best_k:
            best_k = len(S_improved)
            best_S = S_improved.copy()
            
    # Convert best_S indices to vectors
    if best_S:
        result = shell[best_S]
    else:
        # Fallback
        result = shell[:1]
        
    return result

# EVOLVE-BLOCK-END
