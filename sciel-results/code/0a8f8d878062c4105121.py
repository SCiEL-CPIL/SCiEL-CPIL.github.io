# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero
    quadratic residue modulo 205. Uses local search with simulated annealing
    to maximize the set size.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin for 630s limit

    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = list(forbidden)
    
    # Precompute adjacency
    adj = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden_list:
            j = (i + f) % m
            adj[i].add(j)
    # Convert to lists for faster iteration if needed, or keep sets for membership
    adj_lists = [list(s) for s in adj]

    def is_valid_pair(a, b):
        d = (a - b) % m
        return d not in forbidden

    def local_search(selected, time_limit):
        """
        Optimized local search using blocked counts.
        """
        selected = list(selected)
        in_selected = [False] * m
        for v in selected:
            in_selected[v] = True
            
        # blocked[v] = number of neighbors of v in selected
        blocked = [0] * m
        for v in selected:
            for u in adj[v]:
                blocked[u] += 1
                
        best_size = len(selected)
        best_solution = list(selected)
        
        # Phase 1: Greedy Add
        # We can add any v where blocked[v] == 0 and not in_selected[v]
        # To be aggressive, we loop until no more can be added.
        # Order matters. We can try to add the one that blocks the fewest others?
        # Or just random.
        changed = True
        while changed:
            changed = False
            # Find candidates
            candidates = [v for v in range(m) if not in_selected[v] and blocked[v] == 0]
            if candidates:
                # Pick one (random or specific)
                # Let's pick the one with the smallest degree? Or just random.
                # Actually, if we pick one, it might block others.
                # But if blocked[v]==0, v is independent of all current.
                # Adding v is safe.
                # To maximize future options, maybe pick v with min degree?
                # But degree is constant in this graph? No, it's regular?
                # The graph is vertex transitive, so all degrees are equal.
                # So any candidate is equivalent in terms of "cost".
                v = random.choice(candidates)
                selected.append(v)
                in_selected[v] = True
                for u in adj[v]:
                    blocked[u] += 1
                changed = True
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_solution = list(selected)
            if time.time() > time_limit:
                break

        # Phase 2: Local Search (Swaps)
        # Try to remove one and add one to escape local optimum
        # Or remove one and add two?
        # Let's do: Remove u, then try to add any v that becomes unblocked.
        # If we can add v, size stays same. This explores the manifold.
        # If we can add v AND w, size increases.
        
        # We will perform random swaps.
        # 1. Pick u to remove.
        # 2. Decrement blocked for neighbors of u.
        # 3. Check if any v (not u) has blocked[v] == 0.
        # 4. If yes, pick one (or two) and add.
        # 5. If size increased, keep. If same, keep with prob T (annealing) or just keep to explore.
        # 6. If none, revert.
        
        # Simple strategy: 
        # Iterate through all u. For each u, remove it.
        # Find all v with blocked[v] == 0.
        # If there are >= 2 such v (excluding u which is now removed? No u is removed), 
        # we can add 2? No, adding 1 keeps size. Adding 2 increases size.
        # But adding 1 might block the 2nd one.
        
        # Let's just do random restarts of the greedy phase if we get stuck.
        # But first, let's try some swaps to move around.
        
        for _ in range(1000):
            if time.time() > time_limit:
                break
            if not selected:
                break
            
            # Pick a random element to remove
            idx = random.randint(0, len(selected) - 1)
            u = selected[idx]
            
            # Remove u
            selected.pop(idx)
            in_selected[u] = False
            for v in adj[u]:
                blocked[v] -= 1
                
            # Find candidates to add
            # Candidates are v not in selected with blocked[v] == 0
            # We want to add as many as possible?
            # If we add 1, size is same.
            # If we add 2, size increases.
            
            candidates = [v for v in range(m) if not in_selected[v] and blocked[v] == 0]
            
            if len(candidates) >= 2:
                # We can potentially add 2?
                # Let's try to add 2.
                # Pick 2 random candidates c1, c2.
                # Check if c1 and c2 are compatible (not adjacent).
                # If yes, add both.
                # If no, add just one?
                
                # Greedy add from candidates
                added = []
                # Shuffle candidates to be random
                random.shuffle(candidates)
                for c in candidates:
                    if c in in_selected: continue # Should not happen
                    # Check if c is compatible with all currently added in this step
                    # And compatible with existing selected
                    # Since blocked[c] was 0 before we started adding in this step,
                    # and we only add compatible ones, we just need to check against added.
                    
                    ok = True
                    for a in added:
                        if (c - a) % m in forbidden:
                            ok = False
                            break
                    if ok:
                        # Add c
                        selected.append(c)
                        in_selected[c] = True
                        for v in adj[c]:
                            blocked[v] += 1
                        added.append(c)
                        
                        # If we added 2, we broke even or gained.
                        # Actually, if we removed 1 and added 2, size +1.
                        # If we added 1, size 0.
                        
            elif len(candidates) == 1:
                c = candidates[0]
                selected.append(c)
                in_selected[c] = True
                for v in adj[c]:
                    blocked[v] += 1
                added = [c]
            else:
                # No candidates. Revert.
                selected.append(u)
                in_selected[u] = True
                for v in adj[u]:
                    blocked[v] += 1
                continue
            
            # Update best
            if len(selected) > best_size:
                best_size = len(selected)
                best_solution = list(selected)
                
            # If we didn't increase size, we might want to revert to keep the "best" configuration?
            # Or keep it to explore?
            # Let's keep it. It's a valid independent set.
            
        return best_solution

    # Generate multiple initial solutions and run local search on each
    best_solution = []
    best_size = 0
    
    # Greedy init
    sol = []
    for candidate in range(m):
        ok = True
        for s in sol:
            if not is_valid_pair(candidate, s):
                ok = False
                break
        if ok:
            sol.append(candidate)
            
    sol = local_search(sol, deadline - 10)
    if len(sol) > best_size:
        best_size = len(sol)
        best_solution = list(sol)
    
    # Try different random orderings for greedy
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        order = list(range(m))
        random.shuffle(order)
        sol = []
        for candidate in order:
            ok = True
            for s in sol:
                if not is_valid_pair(candidate, s):
                    ok = False
                    break
            if ok:
                sol.append(candidate)
                
        sol = local_search(sol, deadline - 30)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = list(sol)
            
    # Random restarts
    for trial in range(100):
        if time.time() > deadline - 30:
            break
        size = random.randint(5, 15)
        sol = random.sample(range(m), size)
        # Clean conflicts
        clean = []
        for x in sol:
            ok = True
            for y in clean:
                if not is_valid_pair(x, y):
                    ok = False
                    break
            if ok:
                clean.append(x)
        sol = local_search(clean, deadline - 30)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = list(sol)
            
    return np.array(sorted(best_solution), dtype=np.int64)


# EVOLVE-BLOCK-END
