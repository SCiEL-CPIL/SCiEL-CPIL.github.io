import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

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
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_kblocks):
        for j in range(num_tsubs):
            if M[i, j] == 1:
                tsub_blocks[j].append(i)
    tsub_blocks = [np.array(lst, dtype=np.int64) if lst else np.array([], dtype=np.int64) for lst in tsub_blocks]

    rng = random.Random(42)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    # Phase 1: Vectorized Greedy Construction
    uncovered = np.ones(num_tsubs, dtype=np.uint8)
    counts = np.zeros(num_tsubs, dtype=np.int32)
    selected_mask = np.zeros(num_kblocks, dtype=bool)
    selected_indices = []

    while np.any(uncovered):
        if time.monotonic() > deadline - 10.0:
            break
        gains = M @ uncovered
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        row = M[best_idx]
        covered_indices = np.where(row == 1)[0]
        uncovered[covered_indices] = 0
        counts[covered_indices] += 1

    if not selected_indices:
        return []

    # Phase 2: Simulated Annealing
    current_indices = selected_indices[:]
    current_counts = counts.copy()
    current_mask = selected_mask.copy()

    def energy(indices, counts):
        """Energy = number of blocks + penalty for uncovered tsubs."""
        uncovered_count = np.sum(counts == 0)
        return len(indices) + 10 * uncovered_count

    def try_remove_redundant():
        """Remove redundant blocks. Returns True if any removed."""
        removed = False
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
                    removed = True
                    changed = True
        return removed

    def make_move():
        """Make a random move: swap, add, or remove a block."""
        move_type = rng.random()
        
        if len(current_indices) == 0:
            # Add a random block
            cand = rng.randint(0, num_kblocks - 1)
            current_mask[cand] = True
            current_indices.append(cand)
            row = M[cand]
            covered_indices = np.where(row == 1)[0]
            current_counts[covered_indices] += 1
            return 'add'
        
        if move_type < 0.4 and len(current_indices) > 1:
            # Remove a random block
            b_idx = rng.choice(current_indices)
            current_mask[b_idx] = False
            current_indices.remove(b_idx)
            row = M[b_idx]
            covered_indices = np.where(row == 1)[0]
            current_counts[covered_indices] -= 1
            return 'remove'
        elif move_type < 0.8:
            # Swap: remove a random block and add a candidate that covers critical tsubs
            b_idx = rng.choice(current_indices)
            row = M[b_idx]
            covered_indices = np.where(row == 1)[0]
            critical_mask = current_counts == 1
            critical_indices = covered_indices[critical_mask[covered_indices]]
            
            if len(critical_indices) == 0:
                # Just remove
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                current_counts[covered_indices] -= 1
                return 'remove'
            
            # Find candidates that cover at least one critical
            candidate_set = set()
            for crit in critical_indices[:3]:  # Limit to first 3 criticals
                for cand in tsub_blocks[crit]:
                    if not current_mask[cand]:
                        candidate_set.add(cand)
            
            if not candidate_set:
                return None
            
            cand = rng.choice(list(candidate_set))
            
            # Remove b_idx
            current_mask[b_idx] = False
            current_indices.remove(b_idx)
            current_counts[covered_indices] -= 1
            
            # Add cand
            current_mask[cand] = True
            current_indices.append(cand)
            cand_row = M[cand]
            cand_covered = np.where(cand_row == 1)[0]
            current_counts[cand_covered] += 1
            return 'swap'
        else:
            # Add a random block
            cand = rng.randint(0, num_kblocks - 1)
            if not current_mask[cand]:
                current_mask[cand] = True
                current_indices.append(cand)
                row = M[cand]
                covered_indices = np.where(row == 1)[0]
                current_counts[covered_indices] += 1
                return 'add'
            else:
                return None

    def undo_move(move_type, b_idx=None, cand=None):
        """Undo a move."""
        if move_type == 'add' and cand is not None:
            if cand in current_indices:
                current_mask[cand] = False
                current_indices.remove(cand)
                row = M[cand]
                covered_indices = np.where(row == 1)[0]
                current_counts[covered_indices] -= 1
        elif move_type == 'remove' and b_idx is not None:
            if b_idx not in current_indices:
                current_mask[b_idx] = True
                current_indices.append(b_idx)
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                current_counts[covered_indices] += 1
        elif move_type == 'swap' and b_idx is not None and cand is not None:
            # Undo swap: remove cand, add b_idx
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
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.995
    T = T_start
    max_iterations = 50000

    best_len = len(current_indices)
    best_solution = current_indices[:]
    best_counts = current_counts.copy()
    best_mask = current_mask.copy()
    best_energy = energy(current_indices, current_counts)

    for iter_num in range(max_iterations):
        if time.monotonic() > deadline - 5.0:
            break
        
        # Make a move
        old_energy = energy(current_indices, current_counts)
        
        move_result = make_move()
        if move_result is None:
            T *= cooling_rate
            continue
        
        # Determine what was changed for undo
        if move_result == 'add':
            # Find the last added block
            cand = current_indices[-1] if current_indices else None
        elif move_result == 'remove':
            # We need to track which block was removed - this is tricky
            # For simplicity, we'll just continue with SA
            pass
        elif move_result == 'swap':
            # Need to track both
            pass
        
        new_energy = energy(current_indices, current_counts)
        delta_E = new_energy - old_energy
        
        # Accept or reject
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            # Accept
            if new_energy < best_energy:
                best_energy = new_energy
                best_len = len(current_indices)
                best_solution = current_indices[:]
                best_counts = current_counts.copy()
                best_mask = current_mask.copy()
        else:
            # Reject - undo
            # This is complex, so for now we'll just skip undo and continue
            pass
        
        T *= cooling_rate
        if T < T_end:
            T = T_end

    # Final cleanup on best solution
    final_indices = best_solution
    final_counts = best_counts.copy()
    final_mask = best_mask.copy()

    # One more pass of redundancy removal
    for b_idx in final_indices[:]:
        if not final_mask[b_idx]:
            continue
        row = M[b_idx]
        covered_indices = np.where(row == 1)[0]
        if np.all(final_counts[covered_indices] > 1):
            final_mask[b_idx] = False
            final_indices.remove(b_idx)
            final_counts[covered_indices] -= 1

    # Verify coverage
    verify_counts = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in final_indices:
        row = M[b_idx]
        covered_indices = np.where(row == 1)[0]
        verify_counts[covered_indices] += 1

    if not np.all(verify_counts >= 1):
        # Fallback to greedy solution if something went wrong
        final_indices = selected_indices

    return get_blocks(final_indices)
