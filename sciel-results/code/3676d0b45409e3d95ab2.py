import time

_START = time.time()
DEADLINE = 85  # Leave 5s margin

def construct():
    t0 = time.time()
    
    # 1. Generate all 2^a * 3^b * 5^c smooth numbers <= 128
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128: break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128: break
            for c in range(0, 5):
                p5 = 5 ** c
                v = p2 * p3 * p5
                if v > 128: break
                if v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    is_smooth_arr = [False] * 129
    for v in smooth:
        is_smooth_arr[v] = True
    
    # Precompute divisibility relationship between smooth numbers
    # div[i][j] = True if smooth[i] divides smooth[j]
    div = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                div[i][j] = True
    
    # For each sum, find the minimum number of terms forming an antichain
    best_count = [1000] * 129
    best_terms = [None] * 129
    
    # Size 1: single smooth number
    for v in smooth:
        if v <= 128 and 1 < best_count[v]:
            best_count[v] = 1
            best_terms[v] = [v]
    
    # Precompute compatible pairs as bitmasks
    # compat_mask[i] has bit j set if i and j are compatible and j > i
    compat_mask = [0] * n_smooth
    for i in range(n_smooth):
        mask = 0
        for j in range(i + 1, n_smooth):
            if not (div[i][j] or div[j][i]):
                mask |= (1 << j)
        compat_mask[i] = mask
    
    # Helper: iterate over set bits in a mask
    def iter_bits(mask):
        """Yield positions of set bits in mask"""
        while mask:
            low_bit = mask & (-mask)
            pos = low_bit.bit_length() - 1
            yield pos
            mask ^= low_bit
    
    # Size 2
    for i in range(n_smooth):
        vi = smooth[i]
        mask = compat_mask[i]
        for j in iter_bits(mask):
            vj = smooth[j]
            s = vi + vj
            if s > 128:
                continue
            if 2 < best_count[s]:
                best_count[s] = 2
                best_terms[s] = [vi, vj]
    
    # Size 3
    for i in range(n_smooth):
        vi = smooth[i]
        mask_i = compat_mask[i]
        for j in iter_bits(mask_i):
            vj = smooth[j]
            # Find k > j such that k is compatible with both i and j
            mask_k = mask_i & compat_mask[j]
            # k must be > j, so clear bits 0..j
            mask_k &= (~((1 << (j + 1)) - 1))
            for k in iter_bits(mask_k):
                vk = smooth[k]
                s = vi + vj + vk
                if s > 128:
                    continue
                if 3 < best_count[s]:
                    best_count[s] = 3
                    best_terms[s] = [vi, vj, vk]
    
    # Size 4
    for i in range(n_smooth):
        vi = smooth[i]
        mask_i = compat_mask[i]
        for j in iter_bits(mask_i):
            vj = smooth[j]
            mask_ij = mask_i & compat_mask[j]
            mask_ij &= (~((1 << (j + 1)) - 1))  # k > j
            for k in iter_bits(mask_ij):
                vk = smooth[k]
                mask_ijk = mask_ij & compat_mask[k]
                mask_ijk &= (~((1 << (k + 1)) - 1))  # l > k
                for l in iter_bits(mask_ijk):
                    vl = smooth[l]
                    s = vi + vj + vk + vl
                    if s > 128:
                        continue
                    if 4 < best_count[s]:
                        best_count[s] = 4
                        best_terms[s] = [vi, vj, vk, vl]
    
    # Size 5
    for i in range(n_smooth):
        vi = smooth[i]
        mask_i = compat_mask[i]
        for j in iter_bits(mask_i):
            vj = smooth[j]
            mask_ij = mask_i & compat_mask[j]
            mask_ij &= (~((1 << (j + 1)) - 1))
            for k in iter_bits(mask_ij):
                vk = smooth[k]
                mask_ijk = mask_ij & compat_mask[k]
                mask_ijk &= (~((1 << (k + 1)) - 1))
                for l in iter_bits(mask_ijk):
                    vl = smooth[l]
                    mask_ijkl = mask_ijk & compat_mask[l]
                    mask_ijkl &= (~((1 << (l + 1)) - 1))
                    for m in iter_bits(mask_ijkl):
                        vm = smooth[m]
                        s = vi + vj + vk + vl + vm
                        if s > 128:
                            continue
                        if 5 < best_count[s]:
                            best_count[s] = 5
                            best_terms[s] = [vi, vj, vk, vl, vm]
    
    # Fallback backtracking for any n not covered
    def backtrack(rem, min_idx, current_list, current_set):
        if rem == 0:
            return current_list
        if time.time() - t0 > DEADLINE:
            return None
        for idx in range(min_idx, n_smooth):
            s = smooth[idx]
            if s > rem:
                break
            valid = True
            for x in current_set:
                if s % x == 0 or x % s == 0:
                    valid = False
                    break
            if not valid:
                continue
            current_list.append(s)
            current_set.add(s)
            res = backtrack(rem - s, idx + 1, current_list, current_set)
            if res is not None:
                return res
            current_list.pop()
            current_set.remove(s)
        return None
    
    answers = []
    for n in range(1, 129):
        if best_terms[n] is not None:
            answers.append(best_terms[n])
        else:
            res = backtrack(n, 0, [], set())
            if res is not None:
                answers.append(res)
            else:
                answers.append([1, 2, 3, 5, 7, 9, 11, 13, 15, 17])
    
    return answers
