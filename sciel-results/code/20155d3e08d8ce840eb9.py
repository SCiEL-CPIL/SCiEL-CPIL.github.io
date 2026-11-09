import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        """Returns list of maximal cliques as frozensets"""
        cliques = []
        neighbors = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v and adj[u][v]:
                    neighbors[u].add(v)

        def expand(R, P, X):
            if not P and not X:
                cliques.append(frozenset(R))
                return
            # Pivot selection
            union = P | X
            if union:
                # Choose pivot with maximum |P ∩ N(u)|
                max_count = -1
                pivot = None
                for u in union:
                    count = len(P & neighbors[u])
                    if count > max_count:
                        max_count = count
                        pivot = u
                candidates = P - neighbors[pivot] if pivot is not None else P
            else:
                candidates = P

            for v in list(candidates):
                new_R = R | {v}
                new_P = P & neighbors[v]
                new_X = X & neighbors[v]
                expand(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)

        expand(frozenset(), set(range(n)), set())
        return cliques

    def is_valid(sol, cliques):
        sol_set = set(sol)
        for c in cliques:
            if not sol_set & c:
                return False
        return True

    def greedy_init(cliques, n, rng=None):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques"""
        if rng is None:
            rng = random.Random()
        
        uncovered_indices = list(range(len(cliques)))
        covered = [False] * len(cliques)
        selected = []

        # Precompute vertex -> list of clique indices
        vertex_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(i)

        while True:
            if all(covered):
                break
            
            hits = [0] * n
            for i in uncovered_indices:
                if not covered[i]:
                    for v in cliques[i]:
                        hits[v] += 1
            
            max_hits = max(hits)
            if max_hits == 0:
                break
            
            # Pick vertex with max hits, tie-break randomly
            candidates = [v for v in range(n) if hits[v] == max_hits]
            v = rng.choice(candidates)
            selected.append(v)
            
            newly_covered = []
            for i in vertex_cliques[v]:
                if not covered[i]:
                    covered[i] = True
                    newly_covered.append(i)
            
            # Remove covered cliques from uncovered_indices for efficiency
            # This is O(N) per step, but N is small
            uncovered_indices = [i for i in uncovered_indices if not covered[i]]

        return selected

    def local_search_bitmask(sol, cliques, n, time_limit=10.0):
        """
        Local search using bitmasks for cliques.
        sol: list of vertices
        cliques: list of frozensets
        n: number of vertices
        """
        start_ls = time.time()
        
        # Convert cliques to bitmasks
        clique_masks = []
        for c in cliques:
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks.append(mask)
        
        # Map vertex to list of clique masks containing it
        # Actually, we just need to check coverage.
        # Let's maintain sol_set and a count of uncovered cliques?
        # Or just re-check validity?
        # With bitmasks, checking if a clique is covered is (sol_mask & clique_mask) != 0.
        
        sol_set = set(sol)
        sol_mask = 0
        for v in sol_set:
            sol_mask |= (1 << v)
        
        # Check initial validity
        def is_covered(mask):
            for cm in clique_masks:
                if (mask & cm) == 0:
                    return False
            return True

        if not is_covered(sol_mask):
            # Fix invalid solution
            for cm in clique_masks:
                if (sol_mask & cm) == 0:
                    # Add a vertex from this clique
                    # Find first vertex in clique not in sol
                    for v in range(n):
                        if (cm & (1 << v)) and not (sol_mask & (1 << v)):
                            sol_set.add(v)
                            sol_mask |= (1 << v)
                            break
            # Re-verify
            if not is_covered(sol_mask):
                 # Fallback to simple add
                 for cm in clique_masks:
                     if (sol_mask & cm) == 0:
                         for v in range(n):
                             if (cm & (1 << v)):
                                 sol_set.add(v)
                                 sol_mask |= (1 << v)
                         break

        best_sol_size = len(sol_set)
        best_sol = list(sol_set)
        best_mask = sol_mask

        # Precompute for each vertex, the mask of cliques it covers?
        # No, we need to know if removing v leaves any clique uncovered.
        # A clique c is uncovered by (sol_mask ^ (1<<v)) if (c & (sol_mask ^ (1<<v))) == 0.
        # This implies c & (1<<v) != 0 (v was in c) and c & (sol_mask ^ (1<<v)) == 0 (no other sol vertex in c).
        # So v is removable if for all c containing v, there is another vertex in c \in sol.

        while time.time() - start_ls < time_limit:
            improved = False
            
            # 1. Try removing single vertices
            for v in list(sol_set):
                v_bit = 1 << v
                # Check if v is essential
                essential = False
                for cm in clique_masks:
                    if cm & v_bit:
                        # v is in this clique
                        # Check if any other vertex in sol is in this clique
                        others = cm & (sol_mask ^ v_bit)
                        if others == 0:
                            essential = True
                            break
                if not essential:
                    sol_set.remove(v)
                    sol_mask ^= v_bit
                    improved = True
                    if len(sol_set) < best_sol_size:
                        best_sol_size = len(sol_set)
                        best_sol = list(sol_set)
                        best_mask = sol_mask
                    # Restart loop as structure changed significantly
                    break
            
            if improved:
                continue

            # 2. Try swapping 2 vertices for 1
            # If we remove v1, v2, some cliques become uncovered.
            # If those uncovered cliques are all covered by a single vertex w (not in sol),
            # we can replace {v1, v2} with {w}.
            
            # Optimization: Only try pairs if size > 2
            if len(sol_set) > 2:
                found_swap = False
                # Iterate over pairs
                sol_list = list(sol_set)
                for i in range(len(sol_list)):
                    for j in range(i + 1, len(sol_list)):
                        v1, v2 = sol_list[i], sol_list[j]
                        v1_bit = 1 << v1
                        v2_bit = 1 << v2
                        
                        new_mask = sol_mask ^ v1_bit ^ v2_bit
                        
                        # Find uncovered cliques
                        uncovered_cliques = []
                        for cm in clique_masks:
                            if (new_mask & cm) == 0:
                                # Check if v1 or v2 was covering it
                                if (cm & v1_bit) or (cm & v2_bit):
                                    uncovered_cliques.append(cm)
                        
                        if not uncovered_cliques:
                            # Just remove both
                            sol_set.discard(v1)
                            sol_set.discard(v2)
                            sol_mask = new_mask
                            improved = True
                            if len(sol_set) < best_sol_size:
                                best_sol_size = len(sol_set)
                                best_sol = list(sol_set)
                                best_mask = sol_mask
                            found_swap = True
                            break
                        
                        # Check if a single vertex w covers all uncovered_cliques
                        # Candidate w must be in the intersection of all uncovered cliques
                        # Intersection mask
                        inter_mask = uncovered_cliques[0]
                        for cm in uncovered_cliques[1:]:
                            inter_mask &= cm
                            
                        # If inter_mask is 0, no single vertex covers all
                        if inter_mask == 0:
                            continue
                            
                        # Check if any vertex in inter_mask is NOT in sol_set
                        # We want to add w. If w is already in sol_set, it's already in new_mask?
                        # No, new_mask excludes v1, v2. If w was in sol_set and not v1, v2, it is in new_mask.
                        # But if w is in new_mask, then (new_mask & cm) != 0, so cm would not be in uncovered_cliques.
                        # So w MUST be outside new_mask.
                        
                        # Iterate through bits in inter_mask
                        temp = inter_mask
                        while temp:
                            lsb = temp & -temp
                            w_bit = lsb
                            w = lsb.bit_length() - 1
                            # If w is not in sol_set (and not v1, v2)
                            # Actually, if w was in sol_set, it would be in new_mask (unless it was v1 or v2).
                            # If w was v1 or v2, it's not in new_mask. But w must cover uncovered cliques.
                            # If w == v1, does v1 cover all uncovered?
                            # If we remove v1, v2, and add v1 back, we are just removing v2.
                            # This case is handled by single removal if v2 was redundant given v1.
                            # We want w NOT in sol_set to keep size -1.
                            if not (sol_mask & w_bit): # w not in current sol
                                # Check if w covers all uncovered
                                # We know w is in inter_mask, so it's in all uncovered cliques.
                                # So adding w covers them.
                                # New solution: remove v1, v2, add w
                                sol_set.discard(v1)
                                sol_set.discard(v2)
                                sol_set.add(w)
                                sol_mask = (sol_mask ^ v1_bit ^ v2_bit) | w_bit
                                improved = True
                                if len(sol_set) < best_sol_size:
                                    best_sol_size = len(sol_set)
                                    best_sol = list(sol_set)
                                    best_mask = sol_mask
                                found_swap = True
                                break
                            temp ^= lsb
                            
                        if found_swap:
                            break
                    if found_swap:
                        break
            
            if improved:
                continue

            # 3. Perturbation if stuck
            # Add a random vertex, then try to reduce
            if random.random() < 0.5:
                candidates = [v for v in range(n) if v not in sol_set]
                if candidates:
                    v = random.choice(candidates)
                    sol_set.add(v)
                    sol_mask |= (1 << v)
                    # Immediate reduction pass
                    for v_remove in list(sol_set):
                        v_bit = 1 << v_remove
                        essential = False
                        for cm in clique_masks:
                            if cm & v_bit:
                                others = cm & (sol_mask ^ v_bit)
                                if others == 0:
                                    essential = True
                                    break
                        if not essential:
                            sol_set.remove(v_remove)
                            sol_mask ^= v_bit
                            if len(sol_set) < best_sol_size:
                                best_sol_size = len(sol_set)
                                best_sol = list(sol_set)
                                best_mask = sol_mask
                            break # Restart loop

        return best_sol

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]

        if not relevant_cliques:
            return []

        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 2:
            return greedy_init(relevant_cliques, n)

        best_sol = []
        best_size = float('inf')

        # Multiple restarts
        num_restarts = 5
        time_per_restart = (time_left - 5) / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            rng = random.Random(seed + r)
            sol = greedy_init(relevant_cliques, n, rng)
            
            # Local search
            ls_time = min(time_per_restart, 10.0) # Cap local search time
            sol2 = local_search_bitmask(sol, relevant_cliques, n, time_limit=ls_time)
            
            if len(sol2) < best_size:
                best_size = len(sol2)
                best_sol = sol2

        # Try MILP if time permits and local search didn't find a very small solution
        # Heuristic: if best_size > n/4, try MILP
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 5 and best_size > n // 4:
            try:
                from scipy.optimize import milp, LinearConstraint, Bounds
                m = len(relevant_cliques)
                A = np.zeros((m, n), dtype=np.int8)
                for i, clique in enumerate(relevant_cliques):
                    for v in clique:
                        A[i, v] = 1

                c = np.ones(n, dtype=np.float64)
                constraints = LinearConstraint(A, 1, np.inf)
                bounds_obj = Bounds(0, 1)
                integrality = np.ones(n, dtype=np.int8)

                # Use incumbent to help
                incumbent = np.zeros(n)
                for v in best_sol:
                    incumbent[v] = 1

                instance_time = min(time_left - 2, 10)
                res = milp(c=c, constraints=constraints, integrality=integrality,
                           bounds=bounds_obj,
                           options={"time_limit": instance_time, "presolve": True,
                                    "MIP_start": incumbent})

                if res.x is not None:
                    milp_sol = [i for i in range(n) if res.x[i] > 0.5]
                    if is_valid(milp_sol, relevant_cliques):
                        if len(milp_sol) < best_size:
                            best_sol = milp_sol
                            best_size = len(milp_sol)
            except Exception:
                pass

        return best_sol

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time, use greedy for remaining
            adj = random_graph(n, p, seed)
            all_cliques = bron_kerbosch(adj, n)
            relevant_cliques = [c for c in all_cliques if len(c) >= 2]
            sol = greedy_init(relevant_cliques, n) if relevant_cliques else []
            results.append(sorted(sol))
            continue

        sol = solve_instance(n, p, seed)
        results.append(sorted(sol))

    # Ensure we have 3 results
    while len(results) < 3:
        results.append([])

    return results[:3]
