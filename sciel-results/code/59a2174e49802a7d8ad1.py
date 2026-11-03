import time
import random
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 142.0          # hard safety margin before 150s kill
    total = q ** n

    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    # Precompute per-position, per-value index deltas for building neighbor indices.
    pow_q = [q ** (n - 1 - p) for p in range(n)]

    # Build neighbor masks (python ints) and neighbor index lists.
    val_mask = [[0] * q for _ in range(n)]
    for i in range(total):
        t = idx_to_tuple(i)
        bit = 1 << i
        for p in range(n):
            val_mask[p][t[p]] |= bit

    neighbors_mask = [0] * total
    neighbor_indices = [[] for _ in range(total)]
    for i in range(total):
        t = idx_to_tuple(i)
        mask = 1 << i
        idxs = [i]
        for p in range(n):
            base = i - t[p] * pow_q[p]
            for v in range(q):
                if v != t[p]:
                    j = base + v * pow_q[p]
                    mask |= (1 << j)
                    idxs.append(j)
        neighbors_mask[i] = mask
        neighbor_indices[i] = idxs

    # coverers: for each word, list of codeword-indices whose neighborhood covers it
    coverers = [[] for _ in range(total)]
    for i in range(total):
        for w in neighbor_indices[i]:
            coverers[w].append(i)

    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << total) - 1
        gain = [0] * total
        for i in range(total):
            g = (neighbors_mask[i] & uncovered).bit_count()
            if g:
                gain[i] = g
        code = []
        while uncovered:
            best_gain = max(gain)
            top = [i for i in range(total) if gain[i] == best_gain]
            chosen = random.choice(top)
            code.append(chosen)
            newly = neighbors_mask[chosen] & uncovered
            uncovered &= ~neighbors_mask[chosen]
            m = newly
            while m:
                lsb = m & (-m)
                w = lsb.bit_length() - 1
                m ^= lsb
                for c in coverers[w]:
                    gain[c] -= 1
        return code

    def reduce_code_fast(code):
        counts = np.zeros(total, dtype=np.int16)
        for c in code:
            counts[neighbor_indices[c]] += 1
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbor_indices[c]] -= 1
                if np.any(counts == 0):
                    counts[neighbor_indices[c]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search_improve(code, deadline):
        """Simulated annealing on the binary selection vector.
        State = selected set S (bitmask). Energy = W*uncovered + |S|.
        Moves: add / remove / swap. Accept worse states by temperature T.
        """
        sel = 0
        for c in code:
            sel |= (1 << c)
        counts = np.zeros(total, dtype=np.int16)
        for c in code:
            counts[neighbor_indices[c]] += 1
        size = len(code)
        uncovered = int(np.sum(counts == 0))
        W = 100000
        energy = W * uncovered + size
        best_energy = energy
        best_sel = sel
        best_size = size

        T0 = 12.0
        T1 = 0.05
        total_steps = 60000
        steps_done = 0
        rng = random.Random(12345 + len(code))

        while steps_done < total_steps:
            if time.time() >= deadline:
                break
            steps_done += 1
            T = T0 * (T1 / T0) ** (steps_done / total_steps)
            sel_list = [c for c in range(total) if (sel >> c) & 1]
            r = rng.random()

            if r < 0.40:
                if not sel_list:
                    continue
                cand = rng.randrange(total)
                if (sel >> cand) & 1:
                    continue
                before_unc = uncovered
                counts[neighbor_indices[cand]] += 1
                after_unc = int(np.sum(counts == 0))
                dE = W * (after_unc - before_unc) + 1
                if dE <= 0 or rng.random() < np.exp(-dE / T):
                    sel |= (1 << cand)
                    size += 1
                    uncovered = after_unc
                    energy = W * uncovered + size
                else:
                    counts[neighbor_indices[cand]] -= 1
            elif r < 0.75:
                if not sel_list:
                    continue
                cand = rng.choice(sel_list)
                before_unc = uncovered
                counts[neighbor_indices[cand]] -= 1
                after_unc = int(np.sum(counts == 0))
                dE = W * (after_unc - before_unc) - 1
                if dE <= 0 or rng.random() < np.exp(-dE / T):
                    sel &= ~(1 << cand)
                    size -= 1
                    uncovered = after_unc
                    energy = W * uncovered + size
                else:
                    counts[neighbor_indices[cand]] += 1
            else:
                if not sel_list:
                    continue
                cand_out = rng.choice(sel_list)
                cand_in = rng.randrange(total)
                if cand_in == cand_out or (sel >> cand_in) & 1:
                    continue
                before_unc = uncovered
                counts[neighbor_indices[cand_out]] -= 1
                counts[neighbor_indices[cand_in]] += 1
                after_unc = int(np.sum(counts == 0))
                dE = W * (after_unc - before_unc)
                if dE <= 0 or rng.random() < np.exp(-dE / T):
                    sel &= ~(1 << cand_out)
                    sel |= (1 << cand_in)
                    uncovered = after_unc
                    energy = W * uncovered + size
                else:
                    counts[neighbor_indices[cand_in]] -= 1
                    counts[neighbor_indices[cand_out]] += 1

            if energy < best_energy:
                best_energy = energy
                best_sel = sel
                best_size = size

        final_code = [c for c in range(total) if (best_sel >> c) & 1]
        counts2 = np.zeros(total, dtype=np.int16)
        for c in final_code:
            counts2[neighbor_indices[c]] += 1
        if np.any(counts2 == 0):
            final_code = [c for c in range(total) if (sel >> c) & 1]
            counts3 = np.zeros(total, dtype=np.int16)
            for c in final_code:
                counts3[neighbor_indices[c]] += 1
            if np.any(counts3 == 0):
                return code
        return final_code

    best_code = None
    best_size = float('inf')
    seed = 0
    while time.time() < deadline - 2.0:
        code = greedy_construct(seed)
        code = reduce_code_fast(code)
        if len(code) < best_size:
            if time.time() < deadline - 1.0:
                code2 = local_search_improve(code, deadline - 0.5)
                code2 = reduce_code_fast(code2)
                if len(code2) < len(code):
                    code = code2
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        seed += 1

    if best_code is None:
        best_code = list(range(total))

    result = [list(idx_to_tuple(idx)) for idx in best_code]
    return result
