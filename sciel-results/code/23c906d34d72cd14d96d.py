"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 142  # 8s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17

        # Precompute powers of 3 for fast index manipulation
        pow3 = [3 ** i for i in range(n + 1)]

        def int_to_word(idx):
            w = []
            for _ in range(n):
                w.append(idx % 3)
                idx //= 3
            return tuple(reversed(w))

        # Build sphere index matrix: shape (6561, 17)
        sphere_idx_2d = np.empty((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
            w = int_to_word(idx)
            sphere_idx_2d[idx, 0] = idx
            k = 1
            for pos in range(n):
                orig = w[pos]
                base = pow3[n - 1 - pos]
                for d in range(q):
                    if d != orig:
                        sphere_idx_2d[idx, k] = idx + (d - orig) * base
                        k += 1

        # Phase 1: Greedy set cover with criticality weighting
        # Use a priority-based greedy that weights uncovered vertices by how few candidates cover them
        uncovered = np.ones(total_space, dtype=bool)
        uncovered_count = total_space
        code_indices = []

        # Precompute for each vertex, the list of codewords whose sphere contains it
        # This is the "dual" of sphere_idx_2d
        vertex_to_candidates = [[] for _ in range(total_space)]
        for idx in range(total_space):
            for pos in sphere_idx_2d[idx]:
                vertex_to_candidates[int(pos)].append(idx)
        
        # Count how many candidates cover each vertex
        vertex_candidate_count = np.array([len(v) for v in vertex_to_candidates], dtype=np.int32)

        greedy_deadline = start_time + 20
        while uncovered_count > 0:
            if time.time() > greedy_deadline:
                break
            
            # Find uncovered vertices and weight by criticality (fewer candidates = more critical)
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Weight uncovered vertices: prefer those with fewer remaining candidates
            # Use inverse of candidate count as weight (higher weight for more critical)
            weights = 1.0 / (vertex_candidate_count[uncovered_indices] + 1)
            
            # Sample some critical uncovered vertices
            num_samples = min(50, len(uncovered_indices))
            sample_weights = weights[:num_samples]
            sample_indices = uncovered_indices[:num_samples]
            
            # Pick the most critical (lowest candidate count) uncovered vertex
            min_weight_idx = int(np.argmin(weights))
            critical_vertex = uncovered_indices[min_weight_idx]
            
            # Get all candidates that cover this critical vertex
            candidates = vertex_to_candidates[critical_vertex]
            
            # For each candidate, count how many uncovered vertices it covers
            best_candidate = -1
            best_gain = 0
            for cand in candidates:
                gain = int(np.sum(uncovered[sphere_idx_2d[cand]]))
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
            
            if best_candidate < 0 or best_gain <= 0:
                break
            
            code_indices.append(best_candidate)
            uncovered[sphere_idx_2d[best_candidate]] = False
            uncovered_count = int(uncovered.sum())
            
            # Update vertex_candidate_count for newly covered vertices
            for v in sphere_idx_2d[best_candidate]:
                v = int(v)
                if not uncovered[v]:
                    # This vertex is now covered, so one less candidate needed
                    # But we don't actually need to update this for the greedy, it's just a heuristic
                    pass

        m = len(code_indices)

        # Phase 2: Vectorized redundancy removal
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[sphere_idx_2d[idx]] += 1

        active = np.ones(m, dtype=bool)
        sphere_idx_matrix = sphere_idx_2d[code_indices]  # shape (m, 17)

        cleanup_deadline = start_time + 35
        for pass_num in range(3000):
            if time.time() > cleanup_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            # For each active codeword, check if all its sphere positions are covered by >= 2 codewords
            active_spheres = sphere_idx_matrix[active_indices]  # (k, 17)
            active_cover_counts = cover_count[active_spheres]    # (k, 17)
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            if not np.any(redundant_mask):
                break
            redundant_local_indices = np.where(redundant_mask)[0]
            redundant_global_indices = active_indices[redundant_local_indices]
            active[redundant_global_indices] = False
            # Vectorized decrement of cover_count
            for ri in redundant_global_indices:
                cover_count[sphere_idx_2d[code_indices[ri]]] -= 1

        # Phase 3: Simulated annealing with dual variable optimization
        best_code_size = int(np.sum(active))
        best_active = active.copy()
        best_cover_count = cover_count.copy()

        T = 50.0
        cooling_rate = 0.9995
        T_min = 0.1
        sa_deadline = deadline - 15

        # Build a lookup: for each codeword index in code_indices, its position
        idx_to_pos = {}
        for i, ci in enumerate(code_indices):
            idx_to_pos[ci] = i

        # Precompute sphere gains for faster candidate evaluation
        # For each potential codeword, precompute which vertices it covers
        # We'll use this to quickly evaluate the gain of adding/removing a codeword

        while time.time() < sa_deadline and T > T_min:
            active_count = int(np.sum(active))
            if active_count < 2:
                break

            active_indices = np.where(active)[0]
            remove_pos = active_indices[random.randint(0, len(active_indices) - 1)]
            remove_idx = code_indices[remove_pos]

            # Remove it
            active[remove_pos] = False
            remove_sphere = sphere_idx_matrix[remove_pos]
            cover_count[remove_sphere] -= 1
            new_uncovered = int(np.sum(cover_count == 0))

            if new_uncovered > 0:
                # Find best replacement candidate
                # Strategy: look at uncovered vertices, find candidates that cover the most uncovered
                uncovered_mask = cover_count == 0
                uncovered_indices = np.where(uncovered_mask)[0]
                
                best_gain = 0
                best_cand = -1
                
                # Sample uncovered vertices and check their candidate spheres
                num_trials = min(30, len(uncovered_indices))
                if num_trials > 0:
                    trial_indices = random.sample(range(len(uncovered_indices)), num_trials)
                    
                    # For each trial, look at the sphere of that uncovered vertex
                    # and evaluate all candidates in that sphere
                    for ti in trial_indices:
                        uw = int(uncovered_indices[trial_indices[ti]])
                        sphere_of_uw = sphere_idx_2d[uw]
                        
                        # For each position in uw's sphere, it's a potential codeword
                        # that would cover uw. Evaluate its gain.
                        for sp in sphere_of_uw:
                            sp = int(sp)
                            # Gain = number of currently uncovered vertices this candidate covers
                            gain = int(np.sum(cover_count[sphere_idx_2d[sp]] == 0))
                            if gain > best_gain:
                                best_gain = gain
                                best_cand = sp

                if best_cand >= 0 and best_gain > 0:
                    # Add the candidate
                    if best_cand in idx_to_pos:
                        found = idx_to_pos[best_cand]
                        if not active[found]:
                            active[found] = True
                            cover_count[sphere_idx_2d[best_cand]] += 1
                    else:
                        code_indices.append(best_cand)
                        idx_to_pos[best_cand] = len(code_indices) - 1
                        m = len(code_indices)
                        sphere_idx_matrix = np.vstack([sphere_idx_matrix, sphere_idx_2d[best_cand]])
                        active = np.append(active, True)
                        cover_count[sphere_idx_2d[best_cand]] += 1

                    new_uncovered_after = int(np.sum(cover_count == 0))
                    new_size = int(np.sum(active))
                    
                    if new_uncovered_after == 0:
                        if new_size < best_code_size:
                            best_code_size = new_size
                            best_active = active.copy()
                            best_cover_count = cover_count.copy()
                    else:
                        # Accept with probability based on temperature
                        delta_cost = new_uncovered_after - new_uncovered
                        if delta_cost <= 0:
                            # No worse, accept
                            pass
                        else:
                            if random.random() < np.exp(-delta_cost / T):
                                pass  # accept
                            else:
                                # Reject: undo the addition
                                if best_cand in idx_to_pos:
                                    found = idx_to_pos[best_cand]
                                    if found != remove_pos:
                                        active[found] = False
                                        cover_count[sphere_idx_2d[best_cand]] -= 1
                else:
                    # No good candidate found, undo removal
                    active[remove_pos] = True
                    cover_count[remove_sphere] += 1
            else:
                # Removal was successful, no replacement needed
                new_size = int(np.sum(active))
                if new_size < best_code_size:
                    best_code_size = new_size
                    best_active = active.copy()
                    best_cover_count = cover_count.copy()

            T *= cooling_rate

        # Restore best state
        active = best_active.copy()
        cover_count = best_cover_count.copy()

        # Phase 4: Final aggressive redundancy cleanup
        final_deadline = deadline - 5
        for pass_num in range(5000):
            if time.time() > final_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            active_spheres = sphere_idx_matrix[active_indices]
            active_cover_counts = cover_count[active_spheres]
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            if not np.any(redundant_mask):
                break
            redundant_local_indices = np.where(redundant_mask)[0]
            redundant_global_indices = active_indices[redundant_local_indices]
            active[redundant_global_indices] = False
            for ri in redundant_global_indices:
                cover_count[sphere_idx_2d[code_indices[ri]]] -= 1

        final_indices = [code_indices[i] for i in range(len(active)) if active[i]]
        final_code = [int_to_word(idx) for idx in final_indices]

        # Verify coverage
        verify_cover = np.zeros(total_space, dtype=bool)
        for idx in final_indices:
            verify_cover[sphere_idx_2d[idx]] = True
        if np.all(verify_cover):
            return [list(w) for w in final_code]

        # Fallback: return all words (valid but large)
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    else:
        # General case: use the tetracode-based construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback: return all words (valid but not minimal)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
