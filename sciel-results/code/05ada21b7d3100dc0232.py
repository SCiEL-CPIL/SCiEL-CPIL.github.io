"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency as bitmasks for fast intersection
        rem_mask = [0] * n
        for u, v in edges:
            rem_mask[u] |= (1 << v)
            rem_mask[v] |= (1 << u)

        remaining_count = len(edges)
        cliques = []

        # Helper to get list of neighbors from mask
        def mask_to_list(mask):
            res = []
            while mask:
                lsb = mask & (-mask)
                res.append(lsb.bit_length() - 1)
                mask ^= lsb
            return res

        def find_best_clique():
            """
            Finds the best clique by evaluating multiple candidates and picking
            the one that covers the most currently uncovered edges.
            Uses a greedy approach with multiple starting vertices.
            """
            if not any(rem_mask):
                return []
            
            # Get all active vertices sorted by degree
            active_vertices = [v for v in range(n) if rem_mask[v]]
            if not active_vertices:
                return []
            
            active_vertices.sort(key=lambda v: bin(rem_mask[v]).count('1'), reverse=True)
            
            best_clique = []
            best_edges_covered = 0
            
            # Try multiple starting vertices (top candidates by degree)
            num_starts = min(len(active_vertices), 8)
            
            for start_idx in range(num_starts):
                start_v = active_vertices[start_idx]
                # Greedy clique construction from this start vertex
                clique = [start_v]
                candidates = [u for u in active_vertices if u != start_v and (rem_mask[start_v] >> u) & 1]
                candidates.sort(key=lambda u: bin(rem_mask[u] & rem_mask[start_v]).count('1'), reverse=True)
                
                while candidates:
                    # Pick the candidate with highest degree within current clique neighborhood
                    best_cand = None
                    best_score = -1
                    for cand in candidates:
                        # Count how many edges this candidate would add to the clique
                        edges_added = 0
                        for v in clique:
                            if (rem_mask[cand] >> v) & 1:
                                edges_added += 1
                        if edges_added > best_score:
                            best_score = edges_added
                            best_cand = cand
                    
                    if best_cand is None or best_score == 0:
                        break
                    
                    clique.append(best_cand)
                    # Update candidates: only keep those adjacent to all clique members
                    new_candidates = []
                    for cand in candidates:
                        if cand == best_cand:
                            continue
                        is_adj_all = True
                        for v in clique:
                            if not ((rem_mask[cand] >> v) & 1):
                                is_adj_all = False
                                break
                        if is_adj_all:
                            new_candidates.append(cand)
                    candidates = new_candidates
                
                # Count edges covered by this clique
                if len(clique) >= 2:
                    edges_covered = 0
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            u, v = clique[i], clique[j]
                            if (rem_mask[u] >> v) & 1:
                                edges_covered += 1
                    
                    if edges_covered > best_edges_covered:
                        best_edges_covered = edges_covered
                        best_clique = clique[:]
            
            return best_clique

        while remaining_count > 0:
            if time.time() > deadline - 10:
                break

            clique = find_best_clique()

            if not clique or len(clique) < 2:
                # Fallback: take a single edge
                found = False
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            found = True
                            break
                        mask ^= lsb
                    if found:
                        break
                if not found:
                    break
                continue

            # Remove all edges within clique from remaining
            removed = 0
            for i, u in enumerate(clique):
                for v in clique[i + 1:]:
                    if (rem_mask[u] >> v) & 1:
                        rem_mask[u] &= ~(1 << v)
                        rem_mask[v] &= ~(1 << u)
                        removed += 1

            if removed == 0:
                # Shouldn't happen, but fallback
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            break
                        mask ^= lsb
                    if remaining_count < len(edges):
                        break
                continue

            cliques.append(clique)
            remaining_count -= removed

        # Handle any remaining edges as individual edge-cliques
        for u in range(n):
            mask = rem_mask[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    rem_mask[u] &= ~(1 << v)
                    rem_mask[v] &= ~(1 << u)
                mask ^= lsb

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            # Emergency fallback
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
# EVOLVE-BLOCK-END
