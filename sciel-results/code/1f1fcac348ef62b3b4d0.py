import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    TARGET_MIN = 128
    TARGET_MAX = 256
    targets = np.arange(TARGET_MIN, TARGET_MAX + 1)  # shape (129,)

    def compute_mults(A_arr):
        """A_arr: sorted 1D int array. Returns multiplicities for each target in [128,256]."""
        n = len(A_arr)
        if n == 0:
            return np.zeros(129, dtype=np.int32)
        # All pairwise sums (i <= j)
        ii = np.triu_indices(n)
        pair_sums = A_arr[ii[0]] + A_arr[ii[1]]
        # Only keep those in [128, 256]
        mask = (pair_sums >= TARGET_MIN) & (pair_sums <= TARGET_MAX)
        valid_sums = pair_sums[mask]
        # Count occurrences for each target
        mults = np.bincount(valid_sums, minlength=TARGET_MAX + 1)[TARGET_MIN:TARGET_MAX + 1]
        return mults.astype(np.int32)

    def peak_of(mults):
        return int(np.max(mults)) if len(mults) > 0 else 0

    def is_valid(mults):
        return np.all(mults > 0)

    def score_tuple(mults):
        if not is_valid(mults):
            return None
        peak = peak_of(mults)
        sq = int(np.sum(mults * mults))
        return (peak, sq)

    # Initial solution: seed
    A = np.array(list(range(64, 129)), dtype=np.int32)
    mults = compute_mults(A)
    best_score = score_tuple(mults)
    best_A = A.copy()
    best_peak = best_score[0]

    # Phase 1: Greedy removal — try removing each element, keep if valid and improves score
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n = len(A)
        for i in range(n):
            if time.time() > deadline - 5:
                break
            A_try = np.delete(A, i)
            mults_try = compute_mults(A_try)
            sc = score_tuple(mults_try)
            if sc is not None and sc < best_score:
                A = A_try
                best_score = sc
                best_A = A.copy()
                best_peak = sc[0]
                improved = True
                break  # restart scan after modification

    # Phase 2: Swap search — for each element to remove, try all candidates to add
    # Use deterministic pass with shuffled order for semi-randomness
    random.seed(42)
    max_swap_rounds = 200
    for rnd in range(max_swap_rounds):
        if time.time() > deadline - 3:
            break
        n = len(A)
        A_list = list(A)
        random.shuffle(A_list)
        found_swap = False
        for a_out in A_list:
            if time.time() > deadline - 3:
                break
            idx = np.where(A == a_out)[0][0]
            A_no = np.delete(A, idx)
            # Try each candidate to add
            best_in = None
            best_sc = best_score
            for a_in in range(0, 129):
                if a_in in A_no:
                    continue
                A_try = np.sort(np.concatenate([A_no, [a_in]]))
                mults_try = compute_mults(A_try)
                sc = score_tuple(mults_try)
                if sc is not None and sc < best_sc:
                    best_sc = sc
                    best_in = a_in
            if best_in is not None:
                A = np.sort(np.concatenate([A_no, [best_in]]))
                best_score = best_sc
                best_A = A.copy()
                best_peak = best_score[0]
                found_swap = True
                break
        if not found_swap:
            # One more pass without shuffle to be thorough
            A_list2 = list(A)
            found_swap2 = False
            for a_out in A_list2:
                if time.time() > deadline - 3:
                    break
                idx = np.where(A == a_out)[0][0]
                A_no = np.delete(A, idx)
                best_in = None
                best_sc = best_score
                for a_in in range(0, 129):
                    if a_in in A_no:
                        continue
                    A_try = np.sort(np.concatenate([A_no, [a_in]]))
                    mults_try = compute_mults(A_try)
                    sc = score_tuple(mults_try)
                    if sc is not None and sc < best_sc:
                        best_sc = sc
                        best_in = a_in
                if best_in is not None:
                    A = np.sort(np.concatenate([A_no, [best_in]]))
                    best_score = best_sc
                    best_A = A.copy()
                    best_peak = best_score[0]
                    found_swap2 = True
                    break
            if not found_swap2:
                break

    # Phase 3: Final greedy removal pass
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        n = len(A)
        for i in range(n):
            if time.time() > deadline - 1:
                break
            A_try = np.delete(A, i)
            mults_try = compute_mults(A_try)
            sc = score_tuple(mults_try)
            if sc is not None and sc < best_score:
                A = A_try
                best_score = sc
                best_A = A.copy()
                best_peak = sc[0]
                improved = True
                break

    return list(best_A)
