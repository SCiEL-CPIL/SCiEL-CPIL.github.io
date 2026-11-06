import random
import time
# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P union X to reduce recursion
        U = P | X
        if not U:
            return
        # Pick pivot with max degree in P (heuristic)
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _validate_solution(adj, selected, cliques):
    """Check if selected vertices meet every maximal clique of size >= 2."""
    selected_set = set(selected)
    for clique in cliques:
        if not (selected_set & clique):
            return False
    return True

def _greedy_set_cover(cliques):
    """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
    if not cliques:
        return []
    
    n = max(max(c) for c in cliques) + 1
    # Build vertex -> cliques mapping
    vertex_to_cliques = {}
    for i, clique in enumerate(cliques):
        for v in clique:
            if v not in vertex_to_cliques:
                vertex_to_cliques[v] = []
            vertex_to_cliques[v].append(i)
    
    covered = [False] * len(cliques)
    selected = []
    
    while any(not c for c in covered):
        best_v = None
        best_count = -1
        for v, clist in vertex_to_cliques.items():
            count = sum(1 for idx in clist if not covered[idx])
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count == 0:
            break
        selected.append(best_v)
        for idx in vertex_to_cliques[best_v]:
            covered[idx] = True
    
    return sorted(selected)

def _local_search(adj, initial_solution, cliques, max_iters=5000, time_budget=5.0):
    """Optimized Local Search: Removals, 1-1 Swaps, and 2-1 Swaps with coverage counts."""
    start = time.time()
    current = list(initial_solution)
    current_set = set(current)
    n = len(adj)
    
    if not cliques:
        return sorted(current)
    
    # Precompute clique membership for fast checks
    # clique_counts[i] = number of vertices in current_set that are in cliques[i]
    clique_counts = [0] * len(cliques)
    for v in current_set:
        for i, clique in enumerate(cliques):
            if v in clique:
                clique_counts[i] += 1
    
    def is_valid():
        return all(c > 0 for c in clique_counts)
    
    best_size = len(current)
    best_solution = current[:]
    
    if not is_valid():
        # If initial is invalid, we need to repair first, but _random_local_search handles that.
        # For local search, we assume valid start or try to fix.
        # Actually, let's just return if invalid to avoid breaking invariants, 
        # but the caller ensures validity.
        pass

    sideways_moves = 0
    max_sideways = 200
    
    for _ in range(max_iters):
        if time.time() - start > time_budget:
            break
            
        improved = False
        
        # 1. Try removals (0-1 swap: remove 1, add 0)
        for v in list(current):
            # Check if removing v leaves any clique uncovered
            # A clique is uncovered if clique_counts[i] == 1 and v in clique
            # We can just simulate decrement
            valid = True
            for i, clique in enumerate(cliques):
                if v in clique:
                    if clique_counts[i] == 1:
                        valid = False
                        break
            
            if valid:
                # Apply removal
                current.remove(v)
                current_set.remove(v)
                for i, clique in enumerate(cliques):
                    if v in clique:
                        clique_counts[i] -= 1
                
                improved = True
                sideways_moves = 0
                if len(current) < best_size:
                    best_size = len(current)
                    best_solution = current[:]
                break
        
        if improved:
            continue
        
        # 2. Try 1-1 swaps (remove 1, add 1)
        found_swap = False
        if sideways_moves < max_sideways:
            # Sort current vertices by "criticality" (number of cliques they uniquely cover) to prioritize removing less critical ones?
            # Or just iterate.
            for v in list(current):
                # Identify cliques that become uncovered if v is removed
                critical_cliques = []
                for i, clique in enumerate(cliques):
                    if v in clique and clique_counts[i] == 1:
                        critical_cliques.append(i)
                
                if not critical_cliques:
                    # Should have been caught by removal
                    continue
                
                # Find u not in current that covers ALL critical_cliques
                # Intersection of all critical cliques
                candidates = set(cliques[critical_cliques[0]])
                for i in critical_cliques[1:]:
                    candidates &= set(cliques[i])
                
                candidates -= current_set
                
                if candidates:
                    # Pick a candidate. Heuristic: pick one that covers most other uncovered-by-v cliques?
                    # Or just min index.
                    # Let's try to pick one that maximizes coverage of other cliques where count > 1?
                    # Actually, just picking any valid one works for validity.
                    # To improve quality, maybe pick one that allows future removals?
                    # Simple heuristic: min index for determinism, or random.
                    u = min(candidates)
                    
                    # Check if swap is beneficial? 
                    # Size remains same. 
                    # We do this to escape local minima for removals.
                    
                    # Apply swap
                    current.remove(v)
                    current_set.remove(v)
                    current.append(u)
                    current_set.add(u)
                    
                    # Update counts
                    for i, clique in enumerate(cliques):
                        if v in clique:
                            clique_counts[i] -= 1
                        if u in clique:
                            clique_counts[i] += 1
                    
                    sideways_moves += 1
                    found_swap = True
                    break
            
            if found_swap:
                continue
        
        # 3. Try 2-1 swaps (remove 2, add 1)
        # This can reduce size by 1 if the 2 vertices were covering disjoint sets of critical cliques
        # or if their union of critical cliques can be covered by 1 new vertex.
        if sideways_moves < max_sideways // 2: # Do this less frequently
            found_2swap = False
            # Sample pairs to avoid O(N^2) full scan if N is large
            # N is small (<=40), so full scan is okay for a few iterations
            current_list = list(current)
            for i in range(len(current_list)):
                for j in range(i + 1, len(current_list)):
                    v1 = current_list[i]
                    v2 = current_list[j]
                    
                    # Simulate removal of v1 and v2
                    # Check if valid
                    valid = True
                    for k, clique in enumerate(cliques):
                        count = clique_counts[k]
                        if v1 in clique: count -= 1
                        if v2 in clique: count -= 1
                        if count < 0: # Should not happen if counts are accurate
                            valid = False
                            break
                        if count == 0:
                            # This clique is uncovered after removing v1, v2
                            # We need a candidate u that is in this clique
                            # We will check candidates later
                            pass
                    
                    # Actually, we need to find u such that for ALL cliques k:
                    # clique_counts[k] - (v1 in k) - (v2 in k) + (u in k) > 0
                    # Which means:
                    # If clique_counts[k] - (v1 in k) - (v2 in k) == 0, then u MUST be in clique k.
                    # If clique_counts[k] - (v1 in k) - (v2 in k) > 0, u can be anywhere.
                    
                    required_cliques = []
                    for k, clique in enumerate(cliques):
                        count = clique_counts[k]
                        if v1 in clique: count -= 1
                        if v2 in clique: count -= 1
                        if count == 0:
                            required_cliques.append(k)
                    
                    if not required_cliques:
                         # This implies removal was valid, which should have been caught in step 1?
                         # Unless v1 and v2 together cover a clique uniquely?
                         # No, if count > 0 for all, then removal is valid.
                         # Step 1 only removes 1 vertex.
                         # So if removing v1 is invalid, and removing v2 is invalid,
                         # but removing both is valid? 
                         # Example: C1={v1, x}, C2={v2, x}. 
                         # Remove v1: C1 uncovered (if x not in current). 
                         # Remove v2: C2 uncovered.
                         # Remove v1, v2: C1 uncovered, C2 uncovered.
                         # So 2-1 swap is for when removing 2 vertices leaves some cliques uncovered,
                         # but 1 new vertex can cover ALL of them.
                         # In the example above, if u=x, then x covers C1 and C2.
                         # So required_cliques would be C1, C2.
                         # Candidates must be in C1 AND C2.
                         # So required_cliques is the set of cliques that become uncovered.
                         # We need u in intersection of all required_cliques.
                         pass
                         
                    if required_cliques:
                        candidates = set(cliques[required_cliques[0]])
                        for k in required_cliques[1:]:
                            candidates &= set(cliques[k])
                        
                        candidates -= current_set
                        
                        if candidates:
                            u = min(candidates)
                            
                            # Apply 2-1 swap
                            current.remove(v1)
                            current_set.remove(v1)
                            current.remove(v2)
                            current_set.remove(v2)
                            current.append(u)
                            current_set.add(u)
                            
                            # Update counts
                            for k, clique in enumerate(cliques):
                                if v1 in clique:
                                    clique_counts[k] -= 1
                                if v2 in clique:
                                    clique_counts[k] -= 1
                                if u in clique:
                                    clique_counts[k] += 1
                            
                            if len(current) < best_size:
                                best_size = len(current)
                                best_solution = current[:]
                            
                            improved = True
                            sideways_moves = 0
                            found_2swap = True
                            break
                    else:
                        # If no required cliques, removal was valid. 
                        # This is a 2-0 swap (remove 2, add 0).
                        # We should have caught this? 
                        # If removing v1 is invalid, and removing v2 is invalid, 
                        # can removing both be valid?
                        # Yes, if v1 covers C1 uniquely and v2 covers C2 uniquely, 
                        # and C1, C2 are distinct.
                        # Then removing v1 leaves C1 uncovered.
                        # Removing v2 leaves C2 uncovered.
                        # Removing both leaves C1 and C2 uncovered.
                        # So it's not valid.
                        # The only case where removing both is valid is if 
                        # the "criticality" of v1 and v2 overlaps in a way that 
                        # one vertex covers the critical cliques of both?
                        # No, if v1 is critical for C1, and v2 is critical for C2,
                        # and C1 != C2, then removing both leaves both uncovered.
                        # So required_cliques will never be empty if removal is valid?
                        # Wait.
                        # If v1 is in C1, C2. v2 is in C2, C3.
                        # Current: {v1, v2, x}. x in C1, C3.
                        # C1: {v1, x} -> count 2.
                        # C2: {v1, v2} -> count 2.
                        # C3: {v2, x} -> count 2.
                        # Remove v1: C1 count 1 (x), C2 count 1 (v2). Valid?
                        # My step 1 logic: `if v in clique and clique_counts[i] == 1`.
                        # If count is 2, it's not 1, so it's valid to remove?
                        # NO! If count is 2, and I remove v, count becomes 1.
                        # It is still covered.
                        # So step 1 removal is valid if NO clique has count 1 containing v.
                        # So in this example, removing v1 is valid.
                        # So step 1 would have caught it.
                        
                        # So if we reach here, it means removing v1 alone is invalid, 
                        # and removing v2 alone is invalid.
                        # This implies there is at least one clique C1 with count 1 containing v1,
                        # and at least one clique C2 with count 1 containing v2.
                        # If C1 == C2, then v1 and v2 are both the ONLY vertex in current for C1.
                        # Impossible, count would be 2.
                        # So C1 != C2.
                        # Removing both leaves C1 and C2 uncovered (count 0).
                        # So required_cliques is non-empty.
                        
                        # Thus, the `else` branch (required_cliques empty) should not happen 
                        # if step 1 was thorough.
                        pass
                if found_2swap:
                    break
            if found_2swap:
                continue
        
        sideways_moves += 1
        if sideways_moves > max_sideways:
            break
            
    return sorted(best_solution)

def _random_local_search(adj, cliques, time_budget=5.0):
    """Random restarts with kick strategy and improved local search."""
    start = time.time()
    n = len(adj)
    rng = random.Random(42)
    
    if not cliques:
        return []
    
    best = None
    best_size = float('inf')
    
    # First try greedy
    greedy_sol = _greedy_set_cover(cliques)
    if _validate_solution(adj, greedy_sol, cliques):
        best = greedy_sol
        best_size = len(greedy_sol)
    
    # Random restarts
    for i in range(50):
        if time.time() - start > time_budget:
            break
        
        # Strategy: 
        # 1. Random subset
        # 2. Kick from best (remove k random, repair)
        
        if best and rng.random() < 0.5:
            # Kick from best
            candidate = set(best)
            # Remove k random vertices
            k = rng.randint(1, max(1, len(candidate) // 3))
            if candidate:
                to_remove = rng.sample(list(candidate), min(k, len(candidate)))
                for v in to_remove:
                    candidate.remove(v)
        else:
            # Random start
            subset_size = rng.randint(0, min(n, 10))
            candidate = set(rng.sample(range(n), subset_size))
        
        # Repair: Greedily add vertices to cover all cliques
        covered_cliques = set()
        for i_c, clique in enumerate(cliques):
            if candidate & set(clique):
                covered_cliques.add(i_c)
        
        while len(covered_cliques) < len(cliques):
            uncovered = [cliques[i] for i in range(len(cliques)) if i not in covered_cliques]
            if not uncovered:
                break
            best_v = None
            best_count = -1
            for v in range(n):
                if v in candidate:
                    count = sum(1 for c in uncovered if v in c)
                else:
                    count = sum(1 for c in uncovered if v in c)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            candidate.add(best_v)
            for i_c, clique in enumerate(cliques):
                if i_c not in covered_cliques and best_v in clique:
                    covered_cliques.add(i_c)
        
        if len(covered_cliques) < len(cliques):
            continue
        
        # Local search
        # Allocate remaining time / remaining restarts
        remaining_time = time_budget - (time.time() - start)
        ts = max(0.5, remaining_time / (50 - i))
        improved = _local_search(adj, list(candidate), cliques, max_iters=2000, time_budget=ts)
        
        if _validate_solution(adj, improved, cliques) and len(improved) < best_size:
            best = improved
            best_size = len(improved)
    
    if best is None:
        return greedy_sol
    return best

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline - 10:
            # Fallback
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            sol = _greedy_set_cover(cliques)
            results.append(sol)
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
        
        if not cliques:
            results.append([])
            continue
        
        greedy_sol = _greedy_set_cover(cliques)
        if not _validate_solution(adj, greedy_sol, cliques):
            touched = set()
            for clique in cliques:
                touched.update(clique)
            greedy_sol = sorted(touched)
        
        remaining = deadline - time.time() - 5
        if remaining > 1:
            better = _random_local_search(adj, cliques, time_budget=min(remaining, 40))
            if _validate_solution(adj, better, cliques) and len(better) < len(greedy_sol):
                greedy_sol = better
        
        results.append(greedy_sol)
    
    return results

# EVOLVE-BLOCK-END
