"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy: Find large independent set in weight-4 vector conflict graph using
    greedy MIS + simulated annealing polish, then add non-conflicting weight-2 vectors.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Generate all weight-4 vectors: four ±1's, rest 0 (norm^2 = 4)
    w4_supports = list(itertools.combinations(range(D), 4))
    w4_vectors = []  # list of (support_tuple, sign_tuple)
    for support in w4_supports:
        for signs in itertools.product([1, -1], repeat=4):
            w4_vectors.append((support, signs))

    n_w4_total = len(w4_vectors)  # 5280

    # Build conflict graph using algebraic characterization:
    # Two w4 vectors conflict iff <u,v> > 2
    # Same support: never conflict (distinct sign patterns give <u,v> <= 2)
    # |intersection|=3: conflict iff all 3 shared positions have same sign
    # |intersection|<=2: never conflict

    support_to_indices = defaultdict(list)
    for idx, (sup, sig) in enumerate(w4_vectors):
        support_to_indices[sup].append(idx)

    support_list = list(support_to_indices.keys())
    n_sup = len(support_list)

    # Build adjacency as list of lists (for fast iteration in SA)
    conflicts = [set() for _ in range(n_w4_total)]

    # For each pair of supports with |intersection| = 3
    for i in range(n_sup):
        S1 = support_list[i]
        for j in range(i + 1, n_sup):
            S2 = support_list[j]
            inter = set(S1) & set(S2)
            if len(inter) != 3:
                continue

            idx1_list = support_to_indices[S1]
            idx2_list = support_to_indices[S2]

            inter_sorted = sorted(inter)
            pos_in_S1 = [S1.index(p) for p in inter_sorted]
            pos_in_S2 = [S2.index(p) for p in inter_sorted]

            for idx1 in idx1_list:
                sig1 = w4_vectors[idx1][1]
                for idx2 in idx2_list:
                    sig2 = w4_vectors[idx2][1]
                    all_same = all(sig1[p1] == sig2[p2] for p1, p2 in zip(pos_in_S1, pos_in_S2))
                    if all_same:
                        conflicts[idx1].add(idx2)
                        conflicts[idx2].add(idx1)

    # Convert to sorted lists for faster access
    conflicts_list = [sorted(c) for c in conflicts]
    initial_degrees = np.array([len(c) for c in conflicts_list], dtype=np.int32)

    # Greedy MIS with vectorized operations
    def greedy_mis(random_state):
        """Greedy MIS: repeatedly pick min-degree remaining node."""
        remaining = np.ones(n_w4_total, dtype=bool)
        degrees = initial_degrees.copy()
        selected = []

        while remaining.any():
            if time.time() > deadline - 30:
                break
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = random_state.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            selected.append(chosen_global)

            # Remove chosen and neighbors
            to_remove = set(conflicts_list[chosen_global])
            to_remove.add(chosen_global)
            for r in to_remove:
                if remaining[r]:
                    remaining[r] = False
                    for nb in conflicts_list[r]:
                        if remaining[nb]:
                            degrees[nb] -= 1

        return selected

    # Simulated Annealing polish
    def sa_polish(initial_selected, n_iterations, t_start, t_end, random_state):
        """SA: start from initial independent set, try to grow by swapping."""
        selected_set = set(initial_selected)
        remaining = set(range(n_w4_total)) - selected_set
        current_size = len(selected_set)
        best_size = current_size
        best_set = selected_set.copy()

        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max(n_iterations, 1))

        for it in range(n_iterations):
            if time.time() > deadline - 10:
                break
            t *= cooling

            # Try a random move: remove one selected node, try to add a neighbor of removed
            if not selected_set:
                break
            if random_state.random() < 0.3:
                # Add a random remaining node if it doesn't conflict
                if remaining:
                    candidates = list(remaining)
                    cand = candidates[random_state.randint(len(candidates))]
                    # Check if cand conflicts with any selected
                    has_conflict = False
                    for nb in conflicts_list[cand]:
                        if nb in selected_set:
                            has_conflict = True
                            break
                    if not has_conflict:
                        selected_set.add(cand)
                        remaining.discard(cand)
                        if len(selected_set) > best_size:
                            best_size = len(selected_set)
                            best_set = selected_set.copy()
            else:
                # Swap: remove a random selected node, then try to add multiple neighbors
                sel_list = list(selected_set)
                remove_node = sel_list[random_state.randint(len(sel_list))]
                selected_set.discard(remove_node)
                remaining.add(remove_node)

                # Try to add neighbors of remove_node that don't conflict with remaining selected
                for nb in conflicts_list[remove_node]:
                    if nb in remaining:
                        has_conflict = False
                        for nb2 in conflicts_list[nb]:
                            if nb2 in selected_set:
                                has_conflict = True
                                break
                        if not has_conflict:
                            selected_set.add(nb)
                            remaining.discard(nb)
                            if len(selected_set) > best_size:
                                best_size = len(selected_set)
                                best_set = selected_set.copy()

                # Also try to add remove_node back or other random remaining
                if remove_node in remaining:
                    has_conflict = False
                    for nb in conflicts_list[remove_node]:
                        if nb in selected_set:
                            has_conflict = True
                            break
                    if not has_conflict:
                        selected_set.add(remove_node)
                        remaining.discard(remove_node)

                # Try adding a random remaining node
                if remaining and random_state.random() < 0.5:
                    candidates = list(remaining)
                    cand = candidates[random_state.randint(len(candidates))]
                    has_conflict = False
                    for nb in conflicts_list[cand]:
                        if nb in selected_set:
                            has_conflict = True
                            break
                    if not has_conflict:
                        selected_set.add(cand)
                        remaining.discard(cand)
                        if len(selected_set) > best_size:
                            best_size = len(selected_set)
                            best_set = selected_set.copy()

        return list(best_set)

    # Phase 1: Multiple greedy restarts
    np.random.seed(42)
    best_w4_selected = []
    best_w4_size = 0
    n_restarts = 200

    for attempt in range(n_restarts):
        if time.time() > deadline - 40:
            break
        rs = np.random.RandomState(42 + attempt * 7)
        selected = greedy_mis(rs)
        if len(selected) > best_w4_size:
            best_w4_size = len(selected)
            best_w4_selected = selected

    # Phase 2: SA polish on best solution
    if time.time() < deadline - 20 and best_w4_selected:
        sa_iters = 5000
        t_start = 1.0
        t_end = 0.01
        rs = np.random.RandomState(12345)
        polished = sa_polish(best_w4_selected, sa_iters, t_start, t_end, rs)
        if len(polished) > best_w4_size:
            best_w4_size = len(polished)
            best_w4_selected = polished

    # Phase 3: More SA restarts if time allows
    if time.time() < deadline - 15 and best_w4_selected:
        for sa_attempt in range(5):
            if time.time() > deadline - 15:
                break
            # Start SA from a slightly perturbed version
            rs2 = np.random.RandomState(99999 + sa_attempt)
            perturbed = best_w4_selected.copy()
            if perturbed:
                # Remove 2-3 random elements and re-add
                for _ in range(min(3, len(perturbed))):
                    idx_rm = rs2.randint(len(perturbed))
                    perturbed.pop(idx_rm)
            polished2 = sa_polish(perturbed, 3000, 0.8, 0.005, rs2)
            if len(polished2) > best_w4_size:
                best_w4_size = len(polished2)
                best_w4_selected = polished2

    # Convert selected w4 to integer vectors
    w4_int_vectors = np.zeros((best_w4_size, D), dtype=np.int64)
    for i, idx in enumerate(best_w4_selected):
        sup, sig = w4_vectors[idx]
        for pos, sign in zip(sup, sig):
            w4_int_vectors[i, pos] = sign

    # Build w4_blocked_w2 set
    w4_blocked_w2 = set()
    for idx in best_w4_selected:
        sup, sig = w4_vectors[idx]
        for a, b in itertools.combinations(range(4), 2):
            pos_a, pos_b = sup[a], sup[b]
            sig_a, sig_b = sig[a], sig[b]
            if pos_a > pos_b:
                pos_a, pos_b = pos_b, pos_a
                sig_a, sig_b = sig_b, sig_a
            w4_blocked_w2.add((pos_a, pos_b, sig_a, sig_b))

    # Generate all weight-2 candidates and filter non-conflicting ones
    # w2 vectors are mutually non-conflicting (distinct (i,j,si,sj) tuples)
    # w2 conflicts with w4 only if both positions in w4 support with matching signs
    w2_option_c = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                if (i, j, si, sj) not in w4_blocked_w2:
                    w2_option_c.append((i, j, si, sj))

    # w1 doubled vectors: 22 total, never conflict with w4
    # But w2 conflicts with w1 if they share a position with same sign
    # Since we include ALL w1, no w2 can be added
    # So choose: w1 (22 vectors) vs w2_option_c
    w1_count = 2 * D  # 22
    w2_count = len(w2_option_c)

    if w2_count >= w1_count:
        aux_vectors = []
        for (i, j, si, sj) in w2_option_c:
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            aux_vectors.append(v)
        aux_vectors = np.array(aux_vectors, dtype=np.int64) if aux_vectors else np.empty((0, D), dtype=np.int64)
    else:
        aux_vectors = np.zeros((w1_count, D), dtype=np.int64)
        for i in range(D):
            aux_vectors[2*i, i] = 2
            aux_vectors[2*i+1, i] = -2

    all_vectors = np.vstack([w4_int_vectors, aux_vectors])

    return all_vectors
# EVOLVE-BLOCK-END
