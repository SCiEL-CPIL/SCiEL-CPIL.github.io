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

        # Precompute all spheres: for each word index, the list of indices it covers
        def int_to_word(idx):
            w = []
            for _ in range(n):
                w.append(idx % 3)
                idx //= 3
            return tuple(reversed(w))

        def word_to_int(w):
            result = 0
            for d in w:
                result = result * 3 + d
            return result

        # Build sphere index matrix: shape (6561, 17)
        sphere_idx_2d = np.empty((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
            w = int_to_word(idx)
            sphere_idx_2d[idx, 0] = idx
            k = 1
            for pos in range(n):
                orig = w[pos]
                for d in range(q):
                    if d != orig:
                        new_idx = idx
                        # Recompute index with changed digit
                        # Faster: use precomputed powers of 3
                        new_idx = idx - orig * (3 ** (n - 1 - pos)) + d * (3 ** (n - 1 - pos))
                        sphere_idx_2d[idx, k] = new_idx
                        k += 1

        # Precompute powers of 3 for fast index manipulation
        pow3 = [3 ** i for i in range(n + 1)]

        # Rebuild sphere matrix correctly using digit decomposition
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

        # Phase 1: Fast greedy set cover (vectorized)
        uncovered = np.ones(total_space, dtype=bool)
        uncovered_count = total_space
        code_indices = []

        greedy_deadline = start_time + 25
        while uncovered_count > 0:
            if time.time() > greedy_deadline:
                break
            # Count how many uncovered words each candidate covers
            counts = uncovered[sphere_idx_2d].sum(axis=1)
            best_idx = int(np.argmax(counts))
            best_count = int(counts[best_idx])
            if best_count <= 0:
                break
            code_indices.append(best_idx)
            uncovered[sphere_idx_2d[best_idx]] = False
            uncovered_count = int(uncovered.sum())

        m = len(code_indices)

        # Phase 2: Vectorized redundancy removal
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[sphere_idx_2d[idx]] += 1

        active = np.ones(m, dtype=bool)
        sphere_idx_matrix = sphere_idx_2d[code_indices]  # shape (m, 17)

        cleanup_deadline = start_time + 40
        for pass_num in range(500):
            if time.time() > cleanup_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            active_spheres = sphere_idx_matrix[active_indices]
            active_cover_counts = cover_count[active_spheres]
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            redundant_local_indices = np.where(redundant_mask)[0]
            if len(redundant_local_indices) == 0:
                break
            redundant_global_indices = active_indices[redundant_local_indices]
            active[redundant_global_indices] = False
            for ri in redundant_global_indices:
                cover_count[sphere_idx_2d[code_indices[ri]]] -= 1

        # Phase 3: Swap-based simulated annealing
        # State: active set of codewords
        # Move: pick an active codeword to remove, pick a random inactive (or new) codeword to add
        # Cost: number of uncovered words

        best_code_size = int(np.sum(active))
        best_active = active.copy()
        best_cover_count = cover_count.copy()

        T = 50.0
        cooling_rate = 0.9995
        T_min = 0.1
        sa_deadline = deadline - 10

        # Precompute: for each word index, its sphere (already in sphere_idx_2d)
        # For speed, track active set as a boolean array and cover_count

        while time.time() < sa_deadline and T > T_min:
            active_count = int(np.sum(active))
            if active_count < 2:
                break

            # Choose a random active codeword to potentially replace
            active_indices = np.where(active)[0]
            remove_pos = active_indices[random.randint(0, len(active_indices) - 1)]

            # Remove it
            active[remove_pos] = False
            remove_sphere = sphere_idx_matrix[remove_pos]
            cover_count[remove_sphere] -= 1
            new_uncovered = int(np.sum(cover_count == 0))

            # Try to add a codeword that covers the most newly-uncovered words
            # Sample a few candidates for speed
            if new_uncovered > 0:
                # Find best candidate from a random sample
                num_samples = 50
                candidates = random.sample(range(total_space), num_samples)
                best_gain = 0
                best_cand = -1
                for cand in candidates:
                    gain = int(np.sum(cover_count[sphere_idx_2d[cand]] == 0))
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = cand

                if best_cand >= 0 and best_gain > 0:
                    # Add the candidate (find or create a slot)
                    # We need to track which indices are in the code.
                    # For simplicity, we add to code_indices if not present, else reactivate.
                    # Check if best_cand is already in code_indices
                    # This is O(m) but m is small (~500-700)
                    found = -1
                    for i in range(m):
                        if code_indices[i] == best_cand:
                            found = i
                            break
                    if found >= 0:
                        active[found] = True
                        cover_count[sphere_idx_2d[best_cand]] += 1
                    else:
                        # Append new codeword
                        code_indices.append(best_cand)
                        m = len(code_indices)
                        sphere_idx_matrix = np.vstack([sphere_idx_matrix, sphere_idx_2d[best_cand]])
                        active = np.append(active, True)
                        cover_count[sphere_idx_2d[best_cand]] += 1

                    new_uncovered_after = int(np.sum(cover_count == 0))
                    if new_uncovered_after == 0:
                        # Valid covering
                        new_size = int(np.sum(active))
                        if new_size < best_code_size:
                            best_code_size = new_size
                            best_active = active.copy()
                            best_cover_count = cover_count.copy()
                    else:
                        # Metropolis acceptance
                        delta_cost = new_uncovered_after - new_uncovered  # should be >= 0
                        if random.random() < np.exp(-delta_cost / T):
                            pass  # accept
                        else:
                            # Reject: undo the add
                            if found >= 0:
                                active[found] = False
                                cover_count[sphere_idx_2d[best_cand]] -= 1
                            else:
                                active[-1] = False
                                cover_count[sphere_idx_2d[best_cand]] -= 1
                else:
                    # No good candidate, restore
                    active[remove_pos] = True
                    cover_count[remove_sphere] += 1
            else:
                # Still covered after removal — accept (improvement)
                new_size = int(np.sum(active))
                if new_size < best_code_size:
                    best_code_size = new_size
                    best_active = active.copy()
                    best_cover_count = cover_count.copy()
                # Optionally, don't restore; keep the removal
                # But we might want to try further removals, so keep it removed

            T *= cooling_rate

        # Restore best state
        active = best_active.copy()
        cover_count = best_cover_count.copy()

        # Phase 4: Final redundancy cleanup
        final_deadline = deadline - 3
        for pass_num in range(200):
            if time.time() > final_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            active_spheres = sphere_idx_matrix[active_indices]
            active_cover_counts = cover_count[active_spheres]
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            redundant_local_indices = np.where(redundant_mask)[0]
            if len(redundant_local_indices) == 0:
                break
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
