import itertools
import random
import time
import numpy as np

V = 19
K = 7
T = 4

# Precompute (one-time cost)
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

# Incidence matrix M: (N_BLOCKS, N_SUBSETS) boolean
M = np.zeros((N_BLOCKS, N_SUBSETS), dtype=bool)
for i, block in enumerate(all_blocks):
    for sub in itertools.combinations(block, T):
        M[i, subset_to_idx[sub]] = True

# Which blocks are currently in the solution (for fast swap filtering)
_in_sol = np.zeros(N_BLOCKS, dtype=bool)


def _rebuild_counts(solution):
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for idx in solution:
        counts += M[idx]
    return counts


def _greedy_init(rng):
    """Greedy set cover, then immediate redundancy cleanup."""
    global _in_sol
    _in_sol[:] = False
    covered = np.zeros(N_SUBSETS, dtype=bool)
    solution = []
    while not covered.all():
        uncovered = (~covered).astype(np.int8)
        gains = M @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        solution.append(best_idx)
        covered |= M[best_idx]
        _in_sol[best_idx] = True
    # immediate redundancy cleanup
    counts = _rebuild_counts(solution)
    changed = True
    while changed:
        changed = False
        for i in range(len(solution) - 1, -1, -1):
            idx = solution[i]
            if np.min(counts[M[idx]]) > 1:
                counts -= M[idx]
                _in_sol[idx] = False
                solution.pop(i)
                changed = True
    return solution, counts


def _try_remove(solution, counts, idx):
    """Try to remove block idx; return True if removed (updates in place)."""
    global _in_sol
    subs = M[idx]
    if np.min(counts[subs]) > 1:
        counts -= subs
        _in_sol[idx] = False
        pos = solution.index(idx)
        solution.pop(pos)
        return True
    return False


def _swap_move(solution, counts, i, rng):
    """Try to swap solution[i] with a better block. Return (changed, new_len_delta)."""
    global _in_sol
    idx_i = solution[i]
    counts -= M[idx_i]
    lost_mask = (counts == 0)
    lost_count = int(np.sum(lost_mask))
    if lost_count == 0:
        # redundant
        _in_sol[idx_i] = False
        solution.pop(i)
        return True, -1
    lost_indices = np.nonzero(lost_mask)[0]
    subset_M = M[:, lost_indices]
    all_covered = subset_M.all(axis=1)
    candidates = np.nonzero(all_covered)[0]
    if candidates.size == 0:
        counts += M[idx_i]
        return False, 0
    valid = [c for c in candidates if not _in_sol[c]]
    if not valid:
        counts += M[idx_i]
        return False, 0
    # Choose candidate minimizing overlap with current solution (more diverse)
    # but weighted toward covering lost well (they all do). Pick random for SA diversity.
    j = int(rng.choice(valid))
    solution[i] = j
    counts += M[j]
    _in_sol[idx_i] = False
    _in_sol[j] = True
    return True, 0


def _add_move(solution, counts, rng):
    """Add a random block (perturbation)."""
    global _in_sol
    c = int(rng.randint(0, N_BLOCKS - 1))
    if _in_sol[c]:
        return False
    solution.append(c)
    counts += M[c]
    _in_sol[c] = True
    return True


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 118.0

    rng = random.Random(12345)

    # Greedy initialization
    solution, counts = _greedy_init(rng)

    best_solution = solution.copy()
    best_len = len(solution)

    # Simulated annealing
    T_start = 2.0
    T_end = 0.05
    # Cooling: geometric
    alpha = (T_end / T_start) ** (1.0 / 2000.0)
    temp = T_start

    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline - 3.0:
        # Pick a move
        r = rng.random()
        if r < 0.55:
            # Remove move
            if len(solution) == 0:
                continue
            i = rng.randint(0, len(solution) - 1)
            idx = solution[i]
            subs = M[idx]
            minc = int(np.min(counts[subs]))
            if minc > 1:
                # safe removal, always accept
                counts -= subs
                _in_sol[idx] = False
                solution.pop(i)
                no_improve = 0
                if len(solution) < best_len:
                    best_len = len(solution)
                    best_solution = solution.copy()
            else:
                # removal loses (minc == 1) -> cost = number of lost subsets
                lost = int(np.sum(counts[subs] == 1))
                # accept with probability exp(-lost/temp)
                if rng.random() < np.exp(-lost / max(temp, 1e-6)):
                    counts -= subs
                    _in_sol[idx] = False
                    solution.pop(i)
                    no_improve = 0
                else:
                    no_improve += 1
        elif r < 0.85:
            # Swap move
            if len(solution) == 0:
                continue
            i = rng.randint(0, len(solution) - 1)
            changed, delta = _swap_move(solution, counts, i, rng)
            if changed:
                no_improve = 0
                if len(solution) < best_len:
                    best_len = len(solution)
                    best_solution = solution.copy()
            else:
                no_improve += 1
        else:
            # Add move (perturbation)
            if _add_move(solution, counts, rng):
                no_improve = 0

        # Cool
        temp *= alpha
        if temp < T_end:
            temp = T_end

        # If stuck long, kick with a few adds
        if no_improve > max_no_improve:
            for _ in range(3):
                _add_move(solution, counts, rng)
            no_improve = 0

    # Final high-precision polish: exhaustive redundancy removal
    final_solution = best_solution.copy()
    final_counts = _rebuild_counts(final_solution)
    changed = True
    while changed:
        changed = False
        for i in range(len(final_solution) - 1, -1, -1):
            idx = final_solution[i]
            if np.min(final_counts[M[idx]]) > 1:
                final_counts -= M[idx]
                final_solution.pop(i)
                changed = True

    return [list(all_blocks[i]) for i in final_solution]
