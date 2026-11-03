"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import time
import numpy as np
import itertools

D = 11
DEADLINE_S = 145.0  # safety margin before 150s limit


def construct():
    t0 = time.perf_counter()

    def elapsed():
        return time.perf_counter() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    # ------------------------------------------------------------------
    # 1) Generate all candidate vectors with norm^2 in {2, 3, 4, 5}
    # ------------------------------------------------------------------
    def generate_candidates():
        vecs = []
        # norm^2 = 2: pairs of ±1 on two coordinates
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vecs.append(v)
        # norm^2 = 2: single coordinate ±2
        for i in range(D):
            for s in (2, -2):
                v = np.zeros(D, dtype=np.int64)
                v[i] = s
                vecs.append(v)
        # norm^2 = 3: triples of ±1
        for combo in itertools.combinations(range(D), 3):
            for signs in itertools.product([1, -1], repeat=3):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                vecs.append(v)
        # norm^2 = 4: quadruples of ±1
        for combo in itertools.combinations(range(D), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                vecs.append(v)
        # norm^2 = 4: single coordinate ±2
        for i in range(D):
            for s in (2, -2):
                v = np.zeros(D, dtype=np.int64)
                v[i] = s
                vecs.append(v)
        # norm^2 = 5: quintuples of ±1
        for combo in itertools.combinations(range(D), 5):
            for signs in itertools.product([1, -1], repeat=5):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                vecs.append(v)
        # norm^2 = 5: one ±2 and one ±1
        for i, j in itertools.combinations(range(D), 2):
            for si in (2, -2):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return vecs

    # ------------------------------------------------------------------
    # 2) Build compatibility matrix
    #    Two vectors u, v are compatible iff dist^2(u,v) >= max(||u||^2, ||v||^2)
    #    i.e., ||u||^2 + ||v||^2 - 2<u,v> >= max(||u||^2, ||v||^2)
    #    i.e., min(||u||^2, ||v||^2) >= 2<u,v>
    #    i.e., <u,v> <= min(||u||^2, ||v||^2) // 2
    # ------------------------------------------------------------------
    def build_compatibility(V, norms_sq):
        n = len(V)
        V_int = V.astype(np.int64)
        G = V_int @ V_int.T  # inner product matrix
        min_norms = np.minimum(norms_sq[:, None], norms_sq[None, :])
        allowed = min_norms // 2
        # compatible[i,j] = (G[i,j] <= allowed[i,j])
        compat = G <= allowed
        # Self-compatibility is always true (norm^2 >= 2*norm^2/2 = norm^2)
        np.fill_diagonal(compat, True)
        return compat

    # ------------------------------------------------------------------
    # 3) Simulated annealing on the support-selection problem
    #    We work with a bitmask over N candidates. The objective is to
    #    maximize the number of selected candidates such that all pairs
    #    are compatible (independent set in the compatibility graph).
    #    Energy = -|S| + penalty * (# violated pairs in S)
    # ------------------------------------------------------------------
    def simulated_annealing(compat, n_cand, n_restarts=50, time_budget=120.0):
        """
        SA where state is a boolean array of length n_cand.
        Moves: flip one bit (add or remove a candidate).
        Energy: -count(selected) + BIG * count(violated_pairs)
        We want to minimize energy.
        """
        BIG = n_cand + 10  # large enough to dominate count
        best_set = np.zeros(n_cand, dtype=bool)
        best_energy = BIG * n_cand  # worst case

        rng = np.random.RandomState(42)

        for restart in range(n_restarts):
            if is_deadline():
                break

            # Initialize: start with a random valid independent set (greedy)
            state = np.zeros(n_cand, dtype=bool)
            # Greedy initialization: randomly order, add if compatible
            order = rng.permutation(n_cand)
            for idx in order:
                if state[idx]:
                    continue
                # Check if idx is compatible with all currently selected
                if not np.any(state & ~compat[idx]):
                    state[idx] = True

            count = np.sum(state)
            # Compute violated pairs
            S_idx = np.where(state)[0]
            if len(S_idx) > 1:
                sub_compat = compat[np.ix_(S_idx, S_idx)]
                violated = np.sum(~sub_compat) // 2  # upper triangle
            else:
                violated = 0
            energy = -count + BIG * violated

            if energy < best_energy:
                best_energy = energy
                best_set = state.copy()

            # SA parameters
            T0 = 5.0
            Tend = 0.01
            n_steps = 5000
            alpha = (Tend / T0) ** (1.0 / n_steps) if n_steps > 0 else 1.0
            T = T0

            for step in range(n_steps):
                if is_deadline():
                    break
                # Pick a random candidate to flip
                idx = rng.randint(n_cand)
                old_val = state[idx]
                state[idx] = not old_val

                # Compute delta energy
                # If adding (old_val=False, new_val=True):
                #   count increases by 1 -> delta_count = -1
                #   violated pairs: for each j in S (j != idx), if not compat[idx,j], add 1 violation
                # If removing (old_val=True, new_val=False):
                #   count decreases by 1 -> delta_count = +1
                #   violated pairs: for each j in S (j != idx), if not compat[idx,j], remove 1 violation

                if not old_val:
                    # Adding idx
                    others = np.where(state & (np.arange(n_cand) != idx))[0]
                    if len(others) > 0:
                        new_violations = np.sum(~compat[idx, others])
                    else:
                        new_violations = 0
                    delta = -1 + BIG * new_violations
                else:
                    # Removing idx
                    others = np.where(state & (np.arange(n_cand) != idx))[0]
                    if len(others) > 0:
                        old_violations = np.sum(~compat[idx, others])
                    else:
                        old_violations = 0
                    delta = 1 - BIG * old_violations

                # Acceptance
                if delta <= 0:
                    accept = True
                else:
                    if T > 1e-9:
                        accept = rng.rand() < np.exp(-delta / T)
                    else:
                        accept = False

                if not accept:
                    state[idx] = old_val  # revert

                T *= alpha

                # Track best
                count = np.sum(state)
                S_idx = np.where(state)[0]
                if len(S_idx) > 1:
                    sub_compat = compat[np.ix_(S_idx, S_idx)]
                    violated = np.sum(~sub_compat) // 2
                else:
                    violated = 0
                energy = -count + BIG * violated
                if energy < best_energy:
                    best_energy = energy
                    best_set = state.copy()

        return best_set

    # ------------------------------------------------------------------
    # 4) Faster SA using vectorized energy computation
    # ------------------------------------------------------------------
    def fast_simulated_annealing(compat, n_cand, n_restarts=30, time_budget=100.0):
        """
        Optimized SA with precomputed row sums for faster energy updates.
        """
        BIG = n_cand + 10
        best_set = np.zeros(n_cand, dtype=bool)
        best_energy = BIG * n_cand

        rng = np.random.RandomState(123)
        arange_n = np.arange(n_cand)

        for restart in range(n_restarts):
            if is_deadline():
                break

            # Greedy random initialization
            state = np.zeros(n_cand, dtype=bool)
            order = rng.permutation(n_cand)
            for idx in order:
                if state[idx]:
                    continue
                if not np.any(state & ~compat[idx]):
                    state[idx] = True

            # Precompute for fast updates:
            # For each i, viol_with[i] = number of selected j != i such that not compat[i,j]
            # We maintain this incrementally.
            # viol_with[i] = sum over j in S, j != i of (1 - compat[i,j])
            #              = |S| - 1 - sum over j in S, j != i of compat[i,j]

            S_count = np.sum(state)
            # compat_selected[i] = sum over j in S of compat[i,j]
            compat_selected = compat[state].sum(axis=1) if S_count > 0 else np.zeros(n_cand)
            # viol_with[i] for i not in S: S_count - compat_selected[i]
            # For i in S: S_count - 1 - (compat_selected[i] - 1) = S_count - compat_selected[i]
            # Actually: for i in S, compat_selected[i] includes compat[i,i]=1
            # So violations for i in S: (S_count - 1) - (compat_selected[i] - 1) = S_count - compat_selected[i]
            # For i not in S: S_count - compat_selected[i]
            # So in both cases: viol_with[i] = S_count - compat_selected[i]

            viol_with = S_count - compat_selected

            # Total violated pairs = sum over i in S of viol_with[i] / 2
            total_violated = np.sum(viol_with[state]) // 2
            energy = -S_count + BIG * total_violated

            if energy < best_energy:
                best_energy = energy
                best_set = state.copy()

            # SA loop
            T0 = 3.0
            Tend = 0.005
            n_steps = 8000
            alpha = (Tend / T0) ** (1.0 / n_steps)
            T = T0

            for step in range(n_steps):
                if is_deadline():
                    break

                idx = rng.randint(n_cand)

                if not state[idx]:
                    # Adding idx
                    new_viol = viol_with[idx]  # violations idx would have with current S
                    delta = -1 + BIG * new_viol
                else:
                    # Removing idx
                    old_viol = viol_with[idx]  # violations idx currently has with S\{idx}
                    delta = 1 - BIG * old_viol

                if delta <= 0:
                    accept = True
                else:
                    if T > 1e-9:
                        accept = rng.rand() < np.exp(-delta / T)
                    else:
                        accept = False

                if accept:
                    if not state[idx]:
                        # Add idx
                        state[idx] = True
                        S_count += 1
                        # Update compat_selected and viol_with for all i
                        # compat_selected[i] += compat[i, idx]
                        compat_selected += compat[:, idx]
                        # viol_with[i] = S_count - compat_selected[i] (but S_count changed)
                        viol_with = S_count - compat_selected
                        # For idx itself: viol_with[idx] should be 0 (no self)
                        # Actually viol_with[idx] = S_count - compat_selected[idx]
                        # compat_selected[idx] includes compat[idx,idx]=1 and all compat[idx,j] for j in S
                        # So viol_with[idx] = S_count - (1 + sum_{j in S, j!=idx} compat[idx,j])
                        # = (S_count - 1) - sum_{j in S, j!=idx} compat[idx,j] = violations of idx with S\{idx}
                        # That's correct.
                    else:
                        # Remove idx
                        state[idx] = False
                        S_count -= 1
                        compat_selected -= compat[:, idx]
                        viol_with = S_count - compat_selected

                # Track best
                total_violated = np.sum(viol_with[state]) // 2
                energy = -S_count + BIG * total_violated
                if energy < best_energy:
                    best_energy = energy
                    best_set = state.copy()

                T *= alpha

        return best_set

    # ------------------------------------------------------------------
    # 5) Main: generate candidates, build compat, run SA
    # ------------------------------------------------------------------
    # Start with a safe baseline: all norm^2=2 vectors
    base_vecs = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base_vecs.append(v)
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            base_vecs.append(v)
    base_arr = np.array(base_vecs, dtype=np.int64)
    best = base_arr
    best_k = base_arr.shape[0]

    if is_deadline():
        return best

    # Generate full candidate pool
    all_vecs = generate_candidates()
    V = np.array(all_vecs, dtype=np.int64)
    n_cand = len(all_vecs)
    norms_sq = np.array([np.dot(v, v) for v in all_vecs], dtype=np.int64)

    if n_cand == 0:
        return best

    # Build compatibility matrix
    compat = build_compatibility(V, norms_sq)

    # Run fast simulated annealing
    if not is_deadline():
        sa_result = fast_simulated_annealing(compat, n_cand, n_restarts=25)
        if np.sum(sa_result) > best_k:
            best = V[sa_result]
            best_k = np.sum(sa_result)

    # Additional restarts with different seeds if time permits
    if not is_deadline():
        # Run a second SA with different initialization strategy
        # Focus on smaller-norm vectors first
        rng_extra = np.random.RandomState(789)
        BIG = n_cand + 10
        arange_n = np.arange(n_cand)

        for extra_restart in range(15):
            if is_deadline():
                break

            state = np.zeros(n_cand, dtype=bool)
            # Prioritize norm^2=2 vectors
            order = np.argsort(norms_sq)
            # Shuffle within same norm groups
            for idx in order:
                if state[idx]:
                    continue
                if not np.any(state & ~compat[idx]):
                    state[idx] = True

            S_count = np.sum(state)
            compat_selected = compat[state].sum(axis=1) if S_count > 0 else np.zeros(n_cand)
            viol_with = S_count - compat_selected
            total_violated = np.sum(viol_with[state]) // 2
            energy = -S_count + BIG * total_violated

            T0 = 2.0
            Tend = 0.005
            n_steps = 6000
            alpha = (Tend / T0) ** (1.0 / n_steps)
            T = T0

            for step in range(n_steps):
                if is_deadline():
                    break
                idx = rng_extra.randint(n_cand)
                if not state[idx]:
                    new_viol = viol_with[idx]
                    delta = -1 + BIG * new_viol
                else:
                    old_viol = viol_with[idx]
                    delta = 1 - BIG * old_viol

                if delta <= 0:
                    accept = True
                else:
                    if T > 1e-9:
                        accept = rng_extra.rand() < np.exp(-delta / T)
                    else:
                        accept = False

                if accept:
                    if not state[idx]:
                        state[idx] = True
                        S_count += 1
                        compat_selected += compat[:, idx]
                        viol_with = S_count - compat_selected
                    else:
                        state[idx] = False
                        S_count -= 1
                        compat_selected -= compat[:, idx]
                        viol_with = S_count - compat_selected

                total_violated = np.sum(viol_with[state]) // 2
                energy = -S_count + BIG * total_violated
                if energy < BIG * n_cand - best_k:  # i.e., -count + BIG*viol < -best_k
                    # This means count - BIG*viol > best_k, which for valid sets (viol=0) means count > best_k
                    if total_violated == 0 and S_count > best_k:
                        best = V[state]
                        best_k = S_count

                T *= alpha

    # Final verification: ensure the returned set is valid
    if best is not None and best.shape[0] > 0:
        # Verify all pairs satisfy the constraint
        n = best.shape[0]
        norms_best = np.array([np.dot(v, v) for v in best], dtype=np.int64)
        G_best = best.astype(np.int64) @ best.astype(np.int64).T
        min_norms_best = np.minimum(norms_best[:, None], norms_best[None, :])
        allowed_best = min_norms_best // 2
        if not np.all(G_best <= allowed_best):
            # Fallback to base
            best = base_arr
            best_k = base_arr.shape[0]

    return best
