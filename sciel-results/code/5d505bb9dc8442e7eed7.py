"""Bootstrap direct approach: meet every maximal clique (size>=2) with a minimal-ish transversal."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_DEADLINE = 150.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - time.time()

def construct():
    start = time.time()
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start > _DEADLINE - 5:
            # Fallback: return all touched vertices for remaining instances
            rng = random.Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            touched = sorted({x for e in edges for x in e})
            out.append(touched)
            continue
        sol = _solve_instance(n, p, seed)
        out.append(sol)
    return out

def _solve_instance(n, p, seed):
    # Build adjacency
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    
    # Enumerate all maximal cliques via Bron-Kerbosch with pivot
    cliques = []
    def bronk(R, P, X):
        if time.time() - _START_TIME > _DEADLINE - 10:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot
        best_pivot = None
        max_count = -1
        for v in P:
            c = len(adj[v] & (P | X))
            if c > max_count:
                max_count = c
                best_pivot = v
        if best_pivot is not None:
            candidates = P - adj[best_pivot]
        else:
            candidates = set(P)
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(set(), set(range(n)), set())
    
    # Filter: only cliques of size >= 2 matter
    cliques = [c for c in cliques if len(c) >= 2]
    
    if not cliques:
        return []
    
    m = len(cliques)
    
    # Bitset representation
    # clique_mask[v] is an integer with bits set for each clique containing v
    clique_mask = [0] * n
    for i, c in enumerate(cliques):
        bit = 1 << i
        for v in c:
            clique_mask[v] |= bit
            
    all_cliques_mask = (1 << m) - 1
    
    # Greedy Set Cover
    def greedy_cover(randomize=False):
        selected = set()
        uncovered = all_cliques_mask
        while uncovered:
            if time.time() - _START_TIME > _DEADLINE - 15:
                break
            best_v = -1
            best_count = 0
            if randomize:
                # Randomized greedy
                candidates = []
                for v in range(n):
                    if v in selected:
                        continue
                    count = (clique_mask[v] & uncovered).bit_count()
                    if count > 0:
                        candidates.append((count, v))
                if not candidates:
                    break
                # Pick one of the best or random among top?
                # Let's pick the one with max count, break ties randomly
                max_c = max(c for c, v in candidates)
                best_candidates = [v for c, v in candidates if c == max_c]
                best_v = random.choice(best_candidates)
                best_count = max_c
            else:
                for v in range(n):
                    if v in selected:
                        continue
                    count = (clique_mask[v] & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best_v = v
            if best_v == -1 or best_count == 0:
                # Should not happen if graph is connected and cliques exist, but safe
                break
            selected.add(best_v)
            uncovered &= ~clique_mask[best_v]
            
        # Ensure coverage
        if uncovered:
            # Add arbitrary vertices to cover remaining
            while uncovered:
                v = (uncovered & -uncovered).bit_length() - 1
                # Find a vertex in clique v
                # We need a mapping from clique index to a vertex in it
                # Pre-compute or just pick first
                # For speed, let's just pick the first vertex in the clique list
                # But we don't have direct index access to clique vertices easily here without storing them
                # Let's store clique_vertices
                pass # Handled in outer scope if needed, but greedy should be perfect if graph is valid
                
        return selected

    # Store clique vertices for fallback
    clique_vertices = [list(c) for c in cliques]

    def greedy_cover_safe(randomize=False):
        selected = set()
        uncovered = all_cliques_mask
        while uncovered:
            if time.time() - _START_TIME > _DEADLINE - 15:
                break
            best_v = -1
            best_count = 0
            if randomize:
                candidates = []
                for v in range(n):
                    if v in selected:
                        continue
                    count = (clique_mask[v] & uncovered).bit_count()
                    if count > 0:
                        candidates.append((count, v))
                if not candidates:
                    break
                max_c = max(c for c, v in candidates)
                best_candidates = [v for c, v in candidates if c == max_c]
                best_v = random.choice(best_candidates)
                best_count = max_c
            else:
                for v in range(n):
                    if v in selected:
                        continue
                    count = (clique_mask[v] & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best_v = v
            if best_v == -1 or best_count == 0:
                break
            selected.add(best_v)
            uncovered &= ~clique_mask[best_v]
            
        # Fallback for any remaining uncovered cliques
        while uncovered:
            # Pick first bit
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            uncovered ^= lsb
            # Pick a vertex from clique idx
            v = clique_vertices[idx][0]
            selected.add(v)
            uncovered &= ~clique_mask[v]
            
        return selected

    # Local Search using Bitsets
    def local_search(selected):
        selected = set(selected)
        if not selected:
            return selected
            
        # Compute union of coverage
        union_mask = 0
        for v in selected:
            union_mask |= clique_mask[v]
            
        if union_mask != all_cliques_mask:
            # Fix invalid solution first
            uncovered = all_cliques_mask & ~union_mask
            while uncovered:
                lsb = uncovered & -uncovered
                idx = lsb.bit_length() - 1
                uncovered ^= lsb
                v = clique_vertices[idx][0]
                selected.add(v)
                union_mask |= clique_mask[v]
                
        improved = True
        while improved and time.time() - _START_TIME < _DEADLINE - 10:
            improved = False
            sel_list = list(selected)
            
            # 1. Deletion
            for v in sel_list:
                if v not in selected:
                    continue
                # Check if v is redundant
                # v is redundant if all cliques in clique_mask[v] are covered by other vertices in selected
                # i.e., (clique_mask[v] & ~union_others) == 0
                # union_others = union_mask ^ (clique_mask[v] & ~union_others) ... no
                # Let's compute union_others efficiently?
                # union_others = union_mask
                # We need to know if any bit in clique_mask[v] is NOT in union of others.
                # This is equivalent to: Is there a bit b in clique_mask[v] such that b is not in any clique_mask[u] for u in selected \ {v}?
                # Let's just recompute union without v if |selected| is small?
                # Or use a counting array?
                # Given n <= 40 and m is likely small (maximal cliques in random graphs are few),
                # we can just check each clique in clique_mask[v].
                
                # Optimization: If clique_mask[v] is a subset of (union_mask ^ clique_mask[v] ??? No)
                # Let's just iterate bits in clique_mask[v]
                mask_v = clique_mask[v]
                redundant = True
                while mask_v:
                    lsb = mask_v & -mask_v
                    idx = lsb.bit_length() - 1
                    mask_v ^= lsb
                    # Check if clique idx is covered by any other vertex in selected
                    covered = False
                    for u in selected:
                        if u == v:
                            continue
                        if clique_mask[u] & lsb:
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                if redundant:
                    selected.remove(v)
                    union_mask &= ~clique_mask[v] # Update union mask? 
                    # Wait, if v was redundant, union_mask doesn't change because all bits in v were in others.
                    # So union_mask remains valid.
                    improved = True
                    break # Restart loop to re-evaluate other vertices
            
            # 2. Swap (1-out, 1-in)
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            for v in list(selected):
                if v not in selected:
                    continue
                # Try swapping v with w not in selected
                # For each w, check if (selected - {v} + {w}) covers all cliques
                # This is expensive if done naively for all w.
                # We can look for w that covers cliques "unique" to v?
                # Actually, we just want to find ANY w such that the new set is valid.
                # Then we can try to delete from the new set.
                
                # Heuristic: Only try w that are in cliques that v covers.
                # Or just try a few random w's?
                # Let's try all w in range(n) not in selected, but limit time.
                # If n is 40, this is 40 * 40 * m operations roughly.
                
                for w in range(n):
                    if w in selected:
                        continue
                    # Check validity of swap
                    # New union = (union_mask & ~clique_mask[v]) | clique_mask[w]
                    # We need this to be all_cliques_mask
                    # Equivalent to: all_cliques_mask is subset of new_union
                    # i.e., (all_cliques_mask & ~new_union) == 0
                    # new_union = (union_mask & ~clique_mask[v]) | clique_mask[w]
                    # Let's compute missing bits if we remove v:
                    # missing_if_remove_v = all_cliques_mask & ~(union_mask & ~clique_mask[v])
                    # We need w to cover missing_if_remove_v
                    # i.e., (clique_mask[w] & missing_if_remove_v) == missing_if_remove_v
                    
                    # Compute union_without_v
                    # Instead of full OR, we can check:
                    # Does w cover all cliques that were ONLY covered by v?
                    # Let's find cliques that are in clique_mask[v] but NOT in any other selected vertex.
                    # This is the "unique" part of v.
                    unique_to_v = clique_mask[v]
                    for u in selected:
                        if u == v:
                            continue
                        unique_to_v &= ~clique_mask[u]
                    
                    # If unique_to_v is 0, v is redundant (handled in deletion).
                    # If not, w MUST cover unique_to_v.
                    if (clique_mask[w] & unique_to_v) != unique_to_v:
                        continue
                    
                    # If w covers unique_to_v, is the rest covered?
                    # The rest of the cliques (all_cliques_mask ^ unique_to_v) are covered by selected \ {v} 
                    # because they were covered by someone else (by definition of unique_to_v).
                    # So if w covers unique_to_v, the new set is valid.
                    
                    # Perform swap
                    selected.remove(v)
                    selected.add(w)
                    # union_mask update:
                    # union_mask = (union_mask & ~clique_mask[v]) | clique_mask[w]
                    # But since we know it's valid, we can just set improved and restart deletion.
                    # We don't strictly need to maintain union_mask perfectly if we recompute or check on the fly,
                    # but it's good for speed.
                    # Let's just mark improved and break to re-run deletion loop with new set.
                    improved = True
                    break
                if improved:
                    break
                    
        return selected

    # Run multiple randomized greedy starts and local search
    best_sol = None
    best_size = float('inf')
    
    # First, a deterministic greedy
    sol = greedy_cover_safe(randomize=False)
    sol = local_search(sol)
    if len(sol) < best_size:
        best_size = len(sol)
        best_sol = sol
        
    # Then, randomized restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        sol = greedy_cover_safe(randomize=True)
        sol = local_search(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_sol = sol
            
    return sorted(best_sol) if best_sol else []

# EVOLVE-BLOCK-END
