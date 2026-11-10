"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def is_square(x):
        if x < 0:
            return False
        r = math.isqrt(x)
        return r * r == x

    squares = set(k * k for k in range(1, 23))

    # Diagonal conflicts: 2a is a perfect square
    diag_conflict = set(a for a in range(1, 257) if is_square(2 * a))

    candidates = [a for a in range(1, 257) if a not in diag_conflict]
    n = len(candidates)
    idx_map = {a: i for i, a in enumerate(candidates)}

    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i], candidates[j]
            if a + b in squares:
                adj[i, j] = 1
                adj[j, i] = 1

    # Greedy: sort by degree, add if no conflict
    degrees = adj.sum(axis=1)
    order = np.argsort(degrees)
    current = np.zeros(n, dtype=bool)
    for idx in order:
        if not np.any(adj[idx] & current):
            current[idx] = True

    best = current.copy()
    best_size = int(current.sum())

    # Local search with multiple restarts and swaps
    def local_search(init_state, max_iters=200):
        state = init_state.copy()
        size = int(state.sum())
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            progress = False
            # Try adding excluded nodes
            excluded = np.where(~state)[0]
            np.random.shuffle(excluded)
            for idx in excluded:
                if state[idx]:
                    continue
                conflicts = np.where(adj[idx] & state)[0]
                if len(conflicts) == 0:
                    state[idx] = True
                    size += 1
                    progress = True
                elif len(conflicts) == 1:
                    # Swap: remove one conflict, add this node
                    b_idx = conflicts[0]
                    # Check if removing b_idx and adding idx improves
                    # Only do swap if it doesn't reduce size (it keeps same size)
                    # But we want to find better configurations, so try swap
                    state[idx] = True
                    state[b_idx] = False
                    progress = True
            # Try swapping two nodes in current with two excluded
            if not progress:
                # Try 1-for-1 swaps more aggressively
                in_set = np.where(state)[0]
                out_set = np.where(~state)[0]
                if len(in_set) > 0 and len(out_set) > 0:
                    # Try replacing each in-set node with a non-conflicting out-set node
                    for i_idx in in_set:
                        # Find out-set nodes that conflict with i_idx
                        # We want to remove i_idx and add some j_idx that doesn't conflict with new set
                        # This is complex, so just try simple swaps:
                        # Remove i_idx, add j_idx if j_idx doesn't conflict with (state - {i_idx} + {j_idx})
                        pass
            if not progress:
                break
        return state, size

    # Run local search from greedy solution
    for restart in range(20):
        if time.time() > deadline:
            break
        if restart == 0:
            state, size = local_search(best)
        else:
            # Perturb: randomly flip some bits
            state = best.copy()
            in_set = np.where(state)[0]
            if len(in_set) > 0:
                # Remove 1-3 random nodes
                n_remove = min(3, len(in_set))
                to_remove = np.random.choice(in_set, n_remove, replace=False)
                state[to_remove] = False
            state, size = local_search(state)
        
        if size > best_size:
            best = state.copy()
            best_size = size

    # Final polish: try to add any remaining nodes
    final_state = best.copy()
    for idx in range(n):
        if time.time() > deadline:
            break
        if not final_state[idx]:
            if not np.any(adj[idx] & final_state):
                final_state[idx] = True
    
    if int(final_state.sum()) > best_size:
        best = final_state
        best_size = int(best.sum())

    # Convert back to values
    result = [candidates[i] for i in range(n) if best[i]]

    # Verify
    def is_valid(s):
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                if is_square(sl[i] + sl[j]):
                    return False
        return True

    if is_valid(result):
        return sorted(result)
    return list(range(1, 257, 3))
# EVOLVE-BLOCK-END
