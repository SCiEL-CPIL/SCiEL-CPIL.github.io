# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/205Z of maximum
    size that does NOT contain any two elements differing by a nonzero perfect square
    modulo 205.

    Strategy:
      1. Precompute the forbidden difference set F (nonzero quadratic residues mod 205).
      2. Build a conflict graph on vertices 0..204 where an edge exists if the difference
         is in F. We want a maximum independent set.
      3. Use randomized greedy construction with multiple restarts, followed by local
         search (add/remove/swap moves) to escape local optima.
      4. Track the best feasible solution found and return it.

    The evaluator kills after 630 s wall-clock; we use a deadline guard with a safety
    margin of ~10 s.
    """
    m = 205
    deadline = time.time() + 620  # 10 s safety margin

    # --- Precompute forbidden differences ---
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = list(forbidden)
    forbidden_set = forbidden  # alias for clarity

    # Build adjacency: adj[v] = set of u such that (v-u) % m is in forbidden
    adj = [set() for _ in range(m)]
    for v in range(m):
        for f in forbidden_list:
            u1 = (v - f) % m
            u2 = (v + f) % m
            if u1 != v:
                adj[v].add(u1)
            if u2 != v:
                adj[v].add(u2)

    # Convert to lists for faster iteration
    adj_list = [list(s) for s in adj]

    def is_valid_addition(candidate, selected_set):
        """Check if candidate can be added to selected_set (a set of ints)."""
        for s in selected_set:
            d = (candidate - s) % m
            if d in forbidden_set:
                return False
        return True

    def greedy_init(rng):
        """Randomized greedy: shuffle candidates, add if compatible."""
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected = []
        selected_set = set()
        for c in candidates:
            if is_valid_addition(c, selected_set):
                selected.append(c)
                selected_set.add(c)
        return selected, selected_set

    def local_search(selected, selected_set, rng):
        """
        Local search: try to add any compatible element; if none, try swapping
        (remove one, add two). Iterate until no improvement or time check.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Try adding any compatible element
            for c in range(m):
                if c in selected_set:
                    continue
                if is_valid_addition(c, selected_set):
                    selected.append(c)
                    selected_set.add(c)
                    improved = True
                    break

            if improved:
                continue

            # Try swap: remove one element s, then try to add two elements that were
            # previously blocked by s.
            # For efficiency, for each s in selected, find elements blocked only by s.
            best_swap = None
            for i, s in enumerate(selected):
                # Find elements not in selected_set that are compatible with all
                # selected except possibly s.
                others = selected_set - {s}
                candidates_to_add = []
                for c in range(m):
                    if c in selected_set:
                        continue
                    # Check compatibility with all others
                    ok = True
                    for o in others:
                        d = (c - o) % m
                        if d in forbidden_set:
                            ok = False
                            break
                    if ok:
                        candidates_to_add.append(c)

                # Try to add two from candidates_to_add that are compatible with each other
                for j in range(len(candidates_to_add)):
                    for k in range(j + 1, len(candidates_to_add)):
                        c1 = candidates_to_add[j]
                        c2 = candidates_to_add[k]
                        d = (c1 - c2) % m
                        if d in forbidden_set:
                            continue
                        # This swap is valid: remove s, add c1, c2
                        if best_swap is None or 2 > 1:  # net gain of 1
                            best_swap = (i, c1, c2)
                            break
                    if best_swap is not None and best_swap[0] == i:
                        break

                if best_swap is not None:
                    break

            if best_swap is not None:
                i, c1, c2 = best_swap
                s = selected[i]
                selected.pop(i)
                selected_set.discard(s)
                selected.append(c1)
                selected.append(c2)
                selected_set.add(c1)
                selected_set.add(c2)
                improved = True

        return selected, selected_set

    # --- Main search loop ---
    rng = random.Random(42)
    best_selected = None
    best_size = 0

    # First, deterministic greedy to get a baseline
    sel, sel_set = greedy_init(rng)
    sel, sel_set = local_search(sel, sel_set, rng)
    if len(sel) > best_size:
        best_size = len(sel)
        best_selected = list(sel)

    # Multiple randomized restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        sel, sel_set = greedy_init(rng)
        sel, sel_set = local_search(sel, sel_set, rng)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = list(sel)
            # If we've found size >= 13, we can be more aggressive
            if best_size >= 13:
                # Do a few more restarts to try to push higher
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    sel2, sel2_set = greedy_init(rng)
                    sel2, sel2_set = local_search(sel2, sel2_set, rng)
                    if len(sel2) > best_size:
                        best_size = len(sel2)
                        best_selected = list(sel2)

    # Final polish: try one more local search from the best solution
    if best_selected is not None:
        best_set = set(best_selected)
        best_selected, best_set = local_search(best_selected, best_set, rng)
        best_size = len(best_selected)

    if best_selected is None:
        # Fallback: should not happen, but just in case
        best_selected = [0]

    return np.array(sorted(best_selected), dtype=np.int64)


# EVOLVE-BLOCK-END
