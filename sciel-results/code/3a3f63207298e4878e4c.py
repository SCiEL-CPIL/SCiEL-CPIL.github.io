import time
import random
import sys
import math
from itertools import combinations

# Increase recursion limit
sys.setrecursionlimit(10000)

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple indices
_triple_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# --- Precomputation: K(2,2,2) Configurations ---
# A K(2,2,2) is defined by 3 disjoint pairs of vertices.
# We need:
# 1. For each triple, the list of K222 indices it belongs to.
# 2. For each K222, the list of triple indices it contains.

def _precompute_k222s():
    triple_to_k222s = [[] for _ in range(N_TRIPLES)]
    k222_triples = []
    
    # Iterate over all sets of 6 vertices
    for verts in combinations(range(V), 6):
        # Generate all 15 pairings of these 6 vertices
        def get_matchings(verts):
            matchings = []
            v0 = verts[0]
            for i in range(1, 6):
                p1 = (v0, verts[i])
                remaining = [v for j, v in enumerate(verts) if j != 0 and j != i]
                r0, r1, r2, r3 = remaining
                matchings.append((p1, (r0, r1), (r2, r3)))
                matchings.append((p1, (r0, r2), (r1, r3)))
                matchings.append((p1, (r0, r3), (r1, r2)))
            return matchings

        for matching in get_matchings(verts):
            p1, p2, p3 = matching
            t_indices = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        t_indices.append(_triple_idx[t])
            
            k_idx = len(k222_triples)
            k222_triples.append(t_indices)
            
            for t_idx in t_indices:
                triple_to_k222s[t_idx].append(k_idx)
                
    return triple_to_k222s, k222_triples

# Run precomputation once at module load
triple_to_k222s, k222_triples = _precompute_k222s()
N_K222 = len(k222_triples)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_size = 0
    best_active = [False] * N_TRIPLES
    best_counts = [0] * N_K222
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
    
    # Greedy Initialization
    def greedy_init(seed):
        random.seed(seed)
        active = [False] * N_TRIPLES
        counts = [0] * N_K222
        current_size = 0
        
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        for t_idx in order:
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        
        return active, counts, current_size

    # Simulated Annealing Local Search
    def simulated_annealing(active, counts, current_size, time_budget=30.0):
        """SA with add, swap, and 2-swap moves"""
        start = time.time()
        deadline_sa = start + time_budget
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        best_size = current_size
        best_active = active[:]
        best_counts = counts[:]
        
        while time.time() < deadline_sa and T > T_min:
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random inactive triple
                inactive = [i for i in range(N_TRIPLES) if not active[i]]
                if not inactive:
                    T *= cooling
                    continue
                i = random.choice(inactive)
                addable = True
                for k_idx in triple_to_k222s[i]:
                    if counts[k_idx] == 7:
                        addable = False
                        break
                if addable:
                    active[i] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[i]:
                        counts[k_idx] += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_active = active[:]
                        best_counts = counts[:]
                else:
                    # Penalty for blocked add
                    T *= cooling
            elif move_type < 0.7:
                # Swap: remove one active, add one inactive
                active_list = [i for i in range(N_TRIPLES) if active[i]]
                if not active_list:
                    T *= cooling
                    continue
                i = random.choice(active_list)
                inactive = [i for i in range(N_TRIPLES) if not active[i]]
                if not inactive:
                    T *= cooling
                    continue
                j = random.choice(inactive)
                
                # Check if j is addable after removing i
                affected = set(triple_to_k222s[i])
                addable = True
                for k_idx in triple_to_k222s[j]:
                    c = counts[k_idx]
                    if k_idx in affected:
                        c -= 1
                    if c == 7:
                        addable = False
                        break
                
                if addable:
                    # Perform swap
                    active[i] = False
                    for k_idx in triple_to_k222s[i]:
                        counts[k_idx] -= 1
                    active[j] = True
                    for k_idx in triple_to_k222s[j]:
                        counts[k_idx] += 1
                    # Size unchanged, but might enable future adds
                else:
                    T *= cooling
            else:
                # 2-swap: remove two active, add two inactive
                active_list = [i for i in range(N_TRIPLES) if active[i]]
                if len(active_list) < 2:
                    T *= cooling
                    continue
                i1, i2 = random.sample(active_list, 2)
                inactive = [i for i in range(N_TRIPLES) if not active[i]]
                if len(inactive) < 2:
                    T *= cooling
                    continue
                j1, j2 = random.sample(inactive, 2)
                
                # Check if both j1, j2 addable after removing i1, i2
                affected = set(triple_to_k222s[i1]) | set(triple_to_k222s[i2])
                addable = True
                for j in [j1, j2]:
                    for k_idx in triple_to_k222s[j]:
                        c = counts[k_idx]
                        if k_idx in affected:
                            c -= 1
                        if c == 7:
                            addable = False
                            break
                    if not addable:
                        break
                
                if addable:
                    # Perform 2-swap
                    for i in [i1, i2]:
                        active[i] = False
                        for k_idx in triple_to_k222s[i]:
                            counts[k_idx] -= 1
                    for j in [j1, j2]:
                        active[j] = True
                        for k_idx in triple_to_k222s[j]:
                            counts[k_idx] += 1
                else:
                    T *= cooling
            
            # Greedy fill: add any possible triples
            for i in range(N_TRIPLES):
                if not active[i]:
                    addable = True
                    for k_idx in triple_to_k222s[i]:
                        if counts[k_idx] == 7:
                            addable = False
                            break
                    if addable:
                        active[i] = True
                        current_size += 1
                        for k_idx in triple_to_k222s[i]:
                            counts[k_idx] += 1
            
            if current_size > best_size:
                best_size = current_size
                best_active = active[:]
                best_counts = counts[:]
            
            T *= cooling
        
        # Restore best
        return best_active, best_counts, best_size
    
    # Perturb and improve
    def perturb_and_improve(active, counts, current_size, time_budget=15.0):
        """Perturb current solution, then run SA to recover"""
        # Remove k random active triples
        k = random.randint(2, min(5, current_size // 3))
        active_list = [i for i in range(N_TRIPLES) if active[i]]
        to_remove = random.sample(active_list, k)
        for i in to_remove:
            active[i] = False
            current_size -= 1
            for k_idx in triple_to_k222s[i]:
                counts[k_idx] -= 1
        
        # Run SA to recover and improve
        return simulated_annealing(active, counts, current_size, time_budget)
    
    # Local Search with Swaps and Perturbations
    def local_search(active, counts, current_size, max_iters=5000):
        improved = True
        iters = 0
        no_improve_count = 0
        max_no_improve = 30
        
        while improved and time_left() > 1.0 and iters < max_iters:
            improved = False
            iters += 1
            
            # Phase 1: Add any possible triple
            added_any = False
            for i in range(N_TRIPLES):
                if time_left() < 0.5:
                    return active, counts, current_size
                if not active[i]:
                    addable = True
                    for k_idx in triple_to_k222s[i]:
                        if counts[k_idx] == 7:
                            addable = False
                            break
                    if addable:
                        active[i] = True
                        current_size += 1
                        for k_idx in triple_to_k222s[i]:
                            counts[k_idx] += 1
                        improved = True
                        added_any = True
            
            if added_any:
                no_improve_count = 0
                continue
            
            # Phase 2: Try swaps (remove one, add one or more)
            if current_size == 0:
                break
                
            active_indices = [i for i in range(N_TRIPLES) if active[i]]
            if not active_indices:
                break
            
            # Try multiple random active triples to find a good swap
            found_improvement = False
            for _ in range(15):
                if time_left() < 0.5:
                    break
                i = random.choice(active_indices)
                
                affected_ks = set(triple_to_k222s[i])
                addable_candidates = []
                
                for j in range(N_TRIPLES):
                    if active[j]:
                        continue
                    addable = True
                    for k_idx in triple_to_k222s[j]:
                        c = counts[k_idx]
                        if k_idx in affected_ks:
                            c -= 1
                        if c == 7:
                            addable = False
                            break
                    if addable:
                        addable_candidates.append(j)
                
                if len(addable_candidates) > 1:
                    random.shuffle(addable_candidates)
                    
                    # Simulate removal
                    old_counts = {k: counts[k] for k in affected_ks}
                    active[i] = False
                    for k_idx in triple_to_k222s[i]:
                        counts[k_idx] -= 1
                    
                    added_j = []
                    for j in addable_candidates:
                        if time_left() < 0.5:
                            break
                        addable = True
                        for k_idx in triple_to_k222s[j]:
                            if counts[k_idx] == 7:
                                addable = False
                                break
                        if addable:
                            active[j] = True
                            added_j.append(j)
                            for k_idx in triple_to_k222s[j]:
                                counts[k_idx] += 1
                    
                    if len(added_j) > 1:
                        current_size += len(added_j) - 1
                        improved = True
                        found_improvement = True
                        no_improve_count = 0
                        break
                    else:
                        # Revert
                        for j in added_j:
                            active[j] = False
                            for k_idx in triple_to_k222s[j]:
                                counts[k_idx] -= 1
                        active[i] = True
                        for k_idx in triple_to_k222s[i]:
                            counts[k_idx] = old_counts[k_idx]
            
            if not found_improvement:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturb: remove a random subset of active triples
                    if current_size > 5:
                        k = min(4, current_size // 2)
                        to_remove = random.sample(active_indices, k)
                        for i in to_remove:
                            active[i] = False
                            current_size -= 1
                            for k_idx in triple_to_k222s[i]:
                                counts[k_idx] -= 1
                        no_improve_count = 0
                        improved = True
                    else:
                        break
        
        return active, counts, current_size

    # Main Loop
    num_restarts = 300
    for r in range(num_restarts):
        if time_left() < 15.0:
            break
            
        seed = random.randint(0, 1000000)
        active, counts, current_size = greedy_init(seed)
        
        # Local search
        active, counts, current_size = local_search(active, counts, current_size, max_iters=3000)
        
        # Simulated annealing
        sa_budget = min(20.0, max(5.0, time_left() * 0.3))
        active, counts, current_size = simulated_annealing(active, counts, current_size, sa_budget)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
    
    # Final intensive phase: perturb and improve
    if time_left() > 10.0:
        # Run several perturb-and-improve cycles
        remaining = time_left()
        num_perturb = int(remaining / 15)
        for p in range(num_perturb):
            if time_left() < 10.0:
                break
            active, counts, current_size = perturb_and_improve(
                best_active[:], best_counts[:], best_size, 
                time_budget=min(15.0, time_left() * 0.5)
            )
            if current_size > best_size:
                best_size = current_size
                best_active = active[:]
                best_counts = counts[:]
    
    # Final local search on best solution
    if time_left() > 5.0:
        active, counts, current_size = local_search(best_active[:], best_counts[:], best_size, max_iters=10000)
        if current_size > best_size:
            best_size = current_size
            best_active = active
            best_counts = counts

    # Construct result
    result = [list(ALL_TRIPLES[i]) for i in range(N_TRIPLES) if best_active[i]]
    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
