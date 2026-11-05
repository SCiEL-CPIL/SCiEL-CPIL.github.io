import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Build reverse index: for each tsub, which blocks cover it
    tsub_blocks = [np.array([], dtype=np.int64) for _ in range(num_tsubs)]
    for j in range(num_tsubs):
        col = M[:, j]
        tsub_blocks[j] = np.where(col == 1)[0]

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    # Phase 1: Greedy Construction with randomization
    rng = random.Random(42)
    
    def greedy_construct(randomize=False, seed=None):
        if seed is not None:
            rng_local = random.Random(seed)
        else:
            rng_local = rng
        
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        selected_mask = np.zeros(num_kblocks, dtype=bool)
        selected_indices = []

        while np.any(uncovered):
            if time.monotonic() > deadline - 15.0:
                break
            gains = M @ uncovered
            if randomize:
                # Add some randomness to the gain
                noise = rng_local.random() * 0.1
                gains = gains + noise
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            counts[covered_indices] += 1

        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for b_idx in selected_indices[:]:
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                if np.all(counts[covered_indices] > 1):
                    selected_mask[b_idx] = False
                    selected_indices.remove(b_idx)
                    counts[covered_indices] -= 1
                    changed = True

        return selected_indices, counts

    # Get initial greedy solution
    selected_indices, counts = greedy_construct(randomize=False)
    
    # Try several randomized greedy constructions and keep the best
    best_indices = selected_indices[:]
    best_counts = counts.copy()
    best_len = len(selected_indices)
    
    for seed in range(20):
        if time.monotonic() > deadline - 10.0:
            break
        idx, cnt = greedy_construct(randomize=True, seed=seed)
        if len(idx) < best_len:
            best_len = len(idx)
            best_indices = idx
            best_counts = cnt

    selected_indices = best_indices
    counts = best_counts

    # Phase 2: Local Search / Simulated Annealing
    current_indices = selected_indices[:]
    current_counts = counts.copy()
    current_mask = np.zeros(num_kblocks, dtype=bool)
    for i in current_indices:
        current_mask[i] = True

    def energy(indices, counts):
        """Energy = number of blocks + penalty for uncovered tsubs."""
        uncovered_count = np.sum(counts == 0)
        return len(indices) + 50 * uncovered_count

    def make_swap_move():
        """Try to swap one block for another to reduce size or maintain coverage."""
        if len(current_indices) <= 1:
            return None
        
        # Pick a random block to potentially remove
        b_idx = rng.choice(current_indices)
        row = M[b_idx]
        covered_indices = np.where(row == 1)[0]
        
        # Check which tsubs would become uncovered if we remove this block
        critical_mask = current_counts == 1
        critical_indices = covered_indices[critical_mask[covered_indices]]
        
        if len(critical_indices) == 0:
            # Block is redundant, just remove it
            current_mask[b_idx] = False
            current_indices.remove(b_idx)
            current_counts[covered_indices] -= 1
            return ('remove', b_idx, None)
        
        # Find candidate blocks that cover all critical tsubs
        # For efficiency, find candidates covering the first critical tsub
        crit = critical_indices[0]
        candidates = tsub_blocks[crit]
        candidates = candidates[candidates != b_idx]
        candidates = candidates[~current_mask[candidates]]
        
        if len(candidates) == 0:
            return None
        
        # Score candidates by how many critical tsubs they cover
        best_cand = None
        best_score = -1
        for cand in candidates[:50]:  # Limit to 50 candidates for speed
            cand_row = M[cand]
            cand_covered = np.where(cand_row == 1)[0]
            # Check if all criticals are covered
            all_covered = True
            for c in critical_indices:
                if cand_row[c] == 0:
                    all_covered = False
                    break
            if all_covered:
                # Score: number of new tsubs covered (potential for future reduction)
                score = np.sum(cand_row * (current_counts == 0))
                if score > best_score:
                    best_score = score
                    best_cand = cand
        
        if best_cand is None:
            return None
        
        # Perform swap
        current_mask[b_idx] = False
        current_indices.remove(b_idx)
        current_counts[covered_indices] -= 1
        
        current_mask[best_cand] = True
        current_indices.append(best_cand)
        cand_row = M[best_cand]
        cand_covered = np.where(cand_row == 1)[0]
        current_counts[cand_covered] += 1
        
        return ('swap', b_idx, best_cand)

    def make_remove_move():
        """Try to remove a redundant block."""
        if len(current_indices) == 0:
            return None
        
        b_idx = rng.choice(current_indices)
        row = M[b_idx]
        covered_indices = np.where(row == 1)[0]
        
        if np.all(current_counts[covered_indices] > 1):
            current_mask[b_idx] = False
            current_indices.remove(b_idx)
            current_counts[covered_indices] -= 1
            return ('remove', b_idx, None)
        
        return None

    def undo_move(move):
        """Undo a move."""
        if move is None:
            return
        
        move_type, b_idx, cand = move
        
        if move_type == 'remove':
            if b_idx not in current_indices:
                current_mask[b_idx] = True
                current_indices.append(b_idx)
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                current_counts[covered_indices] += 1
        elif move_type == 'swap':
            # Remove cand, add b_idx
            if cand in current_indices:
                current_mask[cand] = False
                current_indices.remove(cand)
                row = M[cand]
                covered_indices = np.where(row == 1)[0]
                current_counts[covered_indices] -= 1
            if b_idx not in current_indices:
                current_mask[b_idx] = True
                current_indices.append(b_idx)
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                current_counts[covered_indices] += 1

    # Simulated annealing parameters
    T_start = 3.0
    T_end = 0.05
    cooling_rate = 0.997
    T = T_start
    
    best_len = len(current_indices)
    best_solution = current_indices[:]
    best_counts = current_counts.copy()
    best_mask = current_mask.copy()
    best_energy = energy(current_indices, current_counts)
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.monotonic() < deadline - 5.0:
        old_energy = energy(current_indices, current_counts)
        
        # Choose move type
        move_type = rng.random()
        move = None
        
        if move_type < 0.6:
            move = make_swap_move()
        else:
            move = make_remove_move()
        
        if move is None:
            T *= cooling_rate
            if T < T_end:
                T = T_end
            continue
        
        new_energy = energy(current_indices, current_counts)
        delta_E = new_energy - old_energy
        
        # Accept or reject
        if delta_E < 0 or (T > T_end and rng.random() < np.exp(-delta_E / T)):
            # Accept
            if new_energy < best_energy:
                best_energy = new_energy
                best_len = len(current_indices)
                best_solution = current_indices[:]
                best_counts = current_counts.copy()
                best_mask = current_mask.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject - undo
            undo_move(move)
            no_improve_count += 1
        
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        # If stuck for too long, restart with perturbation
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            T = T_start
            # Perturb: remove a random block and re-add via greedy
            if len(current_indices) > 5:
                # Remove 2-3 random blocks
                for _ in range(rng.randint(2, 3)):
                    if current_indices:
                        b_idx = rng.choice(current_indices)
                        current_mask[b_idx] = False
                        current_indices.remove(b_idx)
                        row = M[b_idx]
                        covered_indices = np.where(row == 1)[0]
                        current_counts[covered_indices] -= 1
                
                # Greedily repair
                while np.any(current_counts == 0):
                    if time.monotonic() > deadline - 5.0:
                        break
                    uncovered_mask = (current_counts == 0).astype(np.uint8)
                    gains = M @ uncovered_mask
                    best_idx = np.argmax(gains)
                    if gains[best_idx] == 0:
                        break
                    if not current_mask[best_idx]:
                        current_mask[best_idx] = True
                        current_indices.append(best_idx)
                        row = M[best_idx]
                        covered_indices = np.where(row == 1)[0]
                        current_counts[covered_indices] += 1
                
                # Remove redundant
                changed = True
                while changed:
                    changed = False
                    for b_idx in current_indices[:]:
                        row = M[b_idx]
                        covered_indices = np.where(row == 1)[0]
                        if np.all(current_counts[covered_indices] > 1):
                            current_mask[b_idx] = False
                            current_indices.remove(b_idx)
                            current_counts[covered_indices] -= 1
                            changed = True

    # Use best solution found
    final_indices = best_solution
    final_counts = best_counts.copy()
    
    # Final cleanup: remove redundant blocks
    final_mask = np.zeros(num_kblocks, dtype=bool)
    for i in final_indices:
        final_mask[i] = True
    
    changed = True
    while changed:
        changed = False
        for b_idx in final_indices[:]:
            row = M[b_idx]
            covered_indices = np.where(row == 1)[0]
            if np.all(final_counts[covered_indices] > 1):
                final_mask[b_idx] = False
                final_indices.remove(b_idx)
                final_counts[covered_indices] -= 1
                changed = True

    # Verify coverage
    verify_counts = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in final_indices:
        row = M[b_idx]
        covered_indices = np.where(row == 1)[0]
        verify_counts[covered_indices] += 1

    if not np.all(verify_counts >= 1):
        # Fallback to greedy solution if something went wrong
        # Reconstruct a valid solution
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        counts2 = np.zeros(num_tsubs, dtype=np.int32)
        selected2 = []
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 1.0:
                break
            gains = M @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected2.append(best_idx)
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            counts2[covered_indices] += 1
        
        final_indices = selected2

    return get_blocks(final_indices)
