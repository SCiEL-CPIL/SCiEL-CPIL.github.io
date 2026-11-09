import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 148  # 2 seconds margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def build_conflict_graph(S):
        """Build adjacency list for conflict graph. Edge (u,v) means u+v is in S or u-v is in S or v-u is in S."""
        S_set = set(S)
        n = len(S)
        idx = {x: i for i, x in enumerate(S)}
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                s = a + b
                if s in S_set:
                    # a + b = s, so (a,b) and (b,s) and (a,s) are all conflicts
                    si = idx[s]
                    adj[i].add(j)
                    adj[j].add(i)
                    adj[i].add(si)
                    adj[si].add(i)
                    adj[j].add(si)
                    adj[si].add(j)
                # Check a - b = c or b - a = c
                d = a - b
                if d in S_set:
                    di = idx[d]
                    adj[i].add(di)
                    adj[di].add(i)
                    adj[j].add(di)
                    adj[di].add(j)
                d2 = b - a
                if d2 in S_set:
                    d2i = idx[d2]
                    adj[i].add(d2i)
                    adj[d2i].add(i)
                    adj[j].add(d2i)
                    adj[d2i].add(j)
        
        return adj, idx, S_set

    def is_sum_free(C, S_set):
        """Check if C is sum-free."""
        C_set = set(C)
        for a in C:
            for b in C:
                if a + b in C_set:
                    return False
        return True

    def greedy_mis(adj, n, deadline, start_time, seed_offset=0):
        """Greedy Maximum Independent Set: repeatedly pick vertex with min degree among remaining."""
        remaining = set(range(n))
        in_set = set()
        
        while remaining and time.time() < deadline:
            # Find vertex with minimum degree in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
            
            in_set.add(best_v)
            remaining.discard(best_v)
            # Remove neighbors of best_v from remaining
            for u in adj[best_v]:
                remaining.discard(u)
        
        return in_set

    def greedy_mis_weighted(adj, n, weights, deadline, start_time):
        """Greedy MIS with weighted priority: pick vertex with min (degree, -weight)."""
        remaining = set(range(n))
        in_set = set()
        
        while remaining and time.time() < deadline:
            min_key = None
            best_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                key = (deg, -weights[v])
                if min_key is None or key < min_key:
                    min_key = key
                    best_v = v
            
            if best_v is None:
                break
            
            in_set.add(best_v)
            remaining.discard(best_v)
            for u in adj[best_v]:
                remaining.discard(u)
        
        return in_set

    def conflict_count(v, current_set, adj, n):
        """Count how many vertices in current_set are adjacent to v."""
        count = 0
        for u in adj[v]:
            if u in current_set:
                count += 1
        return count

    def local_search_swap(S, adj, n, idx, initial_set, deadline, start_time):
        """Local search: try single swaps to improve independence set size."""
        C = set(initial_set)
        outside = set(range(n)) - C
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            best_swap = None
            best_gain = 0
            
            for remove_x in list(C):
                if time.time() > deadline:
                    break
                
                # Count how many outside vertices become addable if we remove remove_x
                # A vertex y outside can be added if it has no neighbors in C - {remove_x}
                addable = 0
                for add_y in outside:
                    # Check if add_y is compatible with C - {remove_x}
                    compatible = True
                    for c in adj[add_y]:
                        if c in C and c != remove_x:
                            compatible = False
                            break
                    if compatible:
                        addable += 1
                
                # We gain 1 (adding) - 1 (removing) = 0 net, but we want to find a swap
                # Actually, let's just try each swap and see if it's valid
                for add_y in outside:
                    if add_y == remove_x:
                        continue
                    
                    # Check if add_y is compatible with C - {remove_x}
                    compatible = True
                    for c in adj[add_y]:
                        if c in C and c != remove_x:
                            compatible = False
                            break
                    
                    if compatible:
                        # This swap is valid
                        best_swap = (remove_x, add_y)
                        improved = True
                        break
                
                if improved:
                    break
            
            if best_swap is not None:
                remove_x, add_y = best_swap
                C.discard(remove_x)
                C.add(add_y)
                outside.discard(add_y)
                outside.add(remove_x)
        
        return C

    def simulated_annealing(S, adj, n, idx, initial_set, deadline, start_time, rng):
        """Simulated annealing on the conflict graph."""
        C = set(initial_set)
        current_size = len(C)
        best_set = set(C)
        best_size = current_size
        
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        while time.time() < deadline and T > min_T:
            # Pick a random move: add, remove, or swap
            move_type = rng.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                candidates = [v for v in range(n) if v not in C]
                if not candidates:
                    T *= cooling
                    continue
                v = rng.choice(candidates)
                # Check if v can be added
                can_add = True
                for u in adj[v]:
                    if u in C:
                        can_add = False
                        break
                if can_add:
                    C.add(v)
                    new_size = len(C)
                    if new_size > best_size:
                        best_size = new_size
                        best_set = set(C)
                else:
                    # Reject or accept with probability
                    delta = -1  # size decreases
                    if rng.random() < np.exp(-delta / T):
                        C.add(v)  # This would create a conflict, so we don't actually do this
                        # Actually, we can't add if it creates a conflict. Let's just skip.
                        C.discard(v)  # undo
                    # In practice, we should only add valid moves or use a different representation
                    # For now, let's just do valid moves
            elif move_type == 'remove':
                if not C:
                    T *= cooling
                    continue
                v = rng.choice(list(C))
                C.discard(v)
                new_size = len(C)
                if new_size > best_size:
                    best_size = new_size
                    best_set = set(C)
            else:  # swap
                if not C or len(C) == n:
                    T *= cooling
                    continue
                remove_candidates = list(C)
                add_candidates = [v for v in range(n) if v not in C]
                if not remove_candidates or not add_candidates:
                    T *= cooling
                    continue
                remove_v = rng.choice(remove_candidates)
                add_v = rng.choice(add_candidates)
                
                # Check if add_v is compatible with C - {remove_v}
                compatible = True
                for u in adj[add_v]:
                    if u in C and u != remove_v:
                        compatible = False
                        break
                
                if compatible:
                    C.discard(remove_v)
                    C.add(add_v)
                    new_size = len(C)
                    if new_size > best_size:
                        best_size = new_size
                        best_set = set(C)
                else:
                    # Try to accept with probability (this would be a "bad" move)
                    # But since it's not valid, we just reject
                    pass
            
            T *= cooling
        
        return best_set

    def solve_instance(S, deadline, start_time):
        """Solve a single instance using multiple strategies."""
        n = len(S)
        adj, idx, S_set = build_conflict_graph(S)
        
        best_set = set()
        best_size = 0
        
        # Strategy 1: Greedy MIS with different weightings
        weights_options = [
            [1] * n,
            [abs(S[i]) for i in range(n)],
            [S[i] for i in range(n)],
            [-S[i] for i in range(n)],
        ]
        
        for weights in weights_options:
            if time.time() > deadline:
                break
            mis = greedy_mis_weighted(adj, n, weights, deadline, start_time)
            # Convert back to values
            mis_values = [S[i] for i in mis]
            if len(mis_values) > best_size:
                best_size = len(mis_values)
                best_set = set(mis)
        
        # Strategy 2: Local search from best greedy result
        if time.time() <= deadline:
            improved = local_search_swap(S, adj, n, idx, best_set, deadline, start_time)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
        
        # Strategy 3: Simulated annealing from best set
        if time.time() <= deadline:
            rng = random.Random(12345)
            sa_result = simulated_annealing(S, adj, n, idx, best_set, deadline, start_time, rng)
            if len(sa_result) > best_size:
                best_size = len(sa_result)
                best_set = sa_result
        
        # Strategy 4: Multiple random restarts of greedy + local search
        if time.time() <= deadline:
            rng = random.Random(67890)
            for restart in range(5):
                if time.time() > deadline:
                    break
                # Randomly perturb the initial set
                perturbed = set(best_set)
                num_perturb = min(3, len(perturbed) // 4)
                for _ in range(num_perturb):
                    if perturbed:
                        v = rng.choice(list(perturbed))
                        perturbed.discard(v)
                        # Try to add a random vertex
                        candidates = [u for u in range(n) if u not in perturbed]
                        if candidates:
                            u = rng.choice(candidates)
                            # Check if u is compatible
                            compatible = True
                            for w in adj[u]:
                                if w in perturbed:
                                    compatible = False
                                    break
                            if compatible:
                                perturbed.add(u)
                
                # Run local search from perturbed set
                improved = local_search_swap(S, adj, n, idx, perturbed, deadline, start_time)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved
        
        return [S[i] for i in best_set]

    results = []
    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    for S in instances:
        if time.time() > deadline:
            # Fallback: return odds if we're out of time
            odds = [x for x in S if x % 2 != 0]
            results.append(odds)
            continue
        
        best = solve_instance(S, deadline, start_time)
        results.append(best)
    
    return results
