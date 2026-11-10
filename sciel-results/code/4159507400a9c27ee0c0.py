# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum-size subset of Z/205Z with no two elements differing by a nonzero
    quadratic residue modulo 205.

    Strategy:
      1. Build the forbidden-difference set F.
      2. Construct a conflict graph G where edge (u,v) exists iff (u-v) mod 205 in F.
      3. Find a maximum independent set in G via:
         - Multiple randomized greedy initializations
         - Local search with add/remove/swap moves
         - Bitset-accelerated feasibility checks
      4. Return the best feasible set found within the time budget.
    """
    m = 205
    deadline = time.time() + 590  # safety margin before 630s kill

    # --- Build forbidden differences ---
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Precompute conflict masks: for each vertex i, a bitset of vertices j that conflict with i
    # Using Python ints as arbitrary-precision bitsets
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j and ((i - j) % m in forbidden or (j - i) % m in forbidden):
                mask |= (1 << j)
        conflict_mask[i] = mask

    def is_independent(selected_set: set) -> bool:
        """Check if selected_set is an independent set in the conflict graph."""
        for i in selected_set:
            if selected_set & _mask_to_set(conflict_mask[i]):
                return False
        return True

    # Precompute sets for quick intersection
    _mask_to_set_cache = {}
    def _mask_to_set(mask):
        if mask in _mask_to_set_cache:
            return _mask_to_set_cache[mask]
        s = set()
        for j in range(m):
            if mask & (1 << j):
                s.add(j)
        _mask_to_set_cache[mask] = s
        return s

    def greedy_randomize(order: list) -> set:
        """Greedy: iterate in given order, add if compatible with current set."""
        selected = set()
        for v in order:
            # Check if v conflicts with any in selected
            cm = conflict_mask[v]
            compatible = True
            for s in selected:
                if cm & (1 << s):
                    compatible = False
                    break
            if compatible:
                selected.add(v)
        return selected

    def local_search(selected: set, max_iters=50000, seed=0) -> set:
        """Local search: try adding, removing, swapping to grow the set."""
        rng = random.Random(seed)
        selected = set(selected)
        best = len(selected)
        stall = 0

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            improved = False

            # Try adding a random unselected vertex
            unselected = [v for v in range(m) if v not in selected]
            if unselected:
                rng.shuffle(unselected)
                for v in unselected[:20]:  # try up to 20 random candidates
                    cm = conflict_mask[v]
                    ok = True
                    for s in selected:
                        if cm & (1 << s):
                            ok = False
                            break
                    if ok:
                        selected.add(v)
                        best = len(selected)
                        improved = True
                        break

            if not improved:
                # Try swap: remove one, add two
                if len(selected) >= 2:
                    sel_list = list(selected)
                    rng.shuffle(sel_list)
                    for i, s1 in enumerate(sel_list[:10]):
                        # Try removing s1 and adding two others
                        remaining = set(selected) - {s1}
                        # Find vertices that conflict with s1 but not with remaining
                        # Actually: find vertices not in selected that don't conflict with remaining
                        candidates = []
                        for v in range(m):
                            if v in selected:
                                continue
                            cm = conflict_mask[v]
                            ok = True
                            for s in remaining:
                                if cm & (1 << s):
                                    ok = False
                                    break
                            if ok:
                                candidates.append(v)
                        if len(candidates) >= 2:
                            rng.shuffle(candidates)
                            for c1 in candidates[:10]:
                                for c2 in candidates[c1+1:c1+11] if c1+1 < len(candidates) else []:
                                    # Check c1 and c2 don't conflict with each other
                                    if conflict_mask[c1] & (1 << c2):
                                        continue
                                    new_set = remaining | {c1, c2}
                                    # Verify independence
                                    valid = True
                                    for a in new_set:
                                        cm_a = conflict_mask[a]
                                        for b in new_set:
                                            if a != b and (cm_a & (1 << b)):
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if valid:
                                        selected = new_set
                                        best = len(selected)
                                        improved = True
                                        break
                                if improved:
                                    break

            if improved:
                stall = 0
            else:
                stall += 1
                if stall > 200:
                    break

        return selected

    # --- Main search loop ---
    best_set = set()
    best_size = 0

    # Deterministic greedy baseline
    base_order = list(range(m))
    base_set = greedy_randomize(base_order)
    best_set = base_set
    best_size = len(base_set)

    # Multiple randomized restarts
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        order = list(range(m))
        random.shuffle(order)
        s = greedy_randomize(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)

        # Local search from this start
        s2 = local_search(s, max_iters=20000, seed=restart_count)
        if len(s2) > best_size:
            best_set = s2
            best_size = len(s2)

        # Also try local search from the current best
        if time.time() < deadline - 15:
            s3 = local_search(best_set, max_iters=10000, seed=restart_count + 10000)
            if len(s3) > best_size:
                best_set = s3
                best_size = len(s3)

    # Final polish: try to add any remaining vertex
    for v in range(m):
        if v in best_set:
            continue
        cm = conflict_mask[v]
        ok = True
        for s in best_set:
            if cm & (1 << s):
                ok = False
                break
        if ok:
            best_set.add(v)
            best_size = len(best_set)

    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
