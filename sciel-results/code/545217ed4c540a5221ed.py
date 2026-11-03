import time
import random

def construct(q, n, radius):
    deadline = time.monotonic() + 118.0
    total = q ** n
    rng = random.Random(20240607)

    # ---- Precompute each codeword's coverage as a bitmask over the word space ----
    covers = [0] * total
    for w in range(total):
        digits = [0] * n
        t = w
        for i in range(n):
            digits[i] = t % q
            t //= q
        mask = 1 << w
        if radius >= 1:
            for i in range(n):
                orig = digits[i]
                for nd in range(q):
                    if nd == orig:
                        continue
                    digits[i] = nd
                    nw = 0
                    for j in range(n):
                        nw = nw * q + digits[j]
                    mask |= (1 << nw)
                digits[i] = orig
        covers[w] = mask

    all_mask = (1 << total) - 1

    def greedy_cover():
        covered = 0
        code = []
        used = set()
        remaining = all_mask
        while remaining:
            best_gain = 0
            best_cands = []
            for c in range(total):
                if c in used:
                    continue
                g = (covers[c] & remaining).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_cands = [c]
                elif g == best_gain and g > 0:
                    best_cands.append(c)
            if not best_cands:
                break
            c = best_cands[rng.randrange(len(best_cands))]
            code.append(c)
            used.add(c)
            covered |= covers[c]
            remaining = all_mask & ~covered
        return code, covered

    def local_search(code, covered):
        cur_code = code[:]
        cur_cover = covered
        cur_set = set(code)
        best_code = code[:]
        best_cover = covered
        best_size = len(code)
        no_improve = 0
        while time.monotonic() < deadline - 0.4:
            r = rng.random()
            if r < 0.75:
                if len(cur_code) == 0:
                    break
                idx = rng.randrange(len(cur_code))
                old = cur_code[idx]
                new = rng.randrange(total)
                if new == old or new in cur_set:
                    continue
                new_cover = (cur_cover | covers[new]) & ~covers[old]
                if new_cover == all_mask:
                    cur_code[idx] = new
                    cur_set.discard(old)
                    cur_set.add(new)
                    cur_cover = new_cover
                    no_improve = 0
                    if len(cur_code) < best_size:
                        best_size = len(cur_code)
                        best_code = cur_code[:]
                        best_cover = cur_cover
                else:
                    no_improve += 1
            elif r < 0.95:
                if len(cur_code) == 0:
                    break
                idx = rng.randrange(len(cur_code))
                c = cur_code[idx]
                new_cover = cur_cover & ~covers[c]
                if new_cover == all_mask:
                    cur_code.pop(idx)
                    cur_set.discard(c)
                    cur_cover = new_cover
                    no_improve = 0
                    if len(cur_code) < best_size:
                        best_size = len(cur_code)
                        best_code = cur_code[:]
                        best_cover = cur_cover
                else:
                    no_improve += 1
            else:
                new = rng.randrange(total)
                if new not in cur_set:
                    cur_code.append(new)
                    cur_set.add(new)
                    cur_cover = cur_cover | covers[new]
                    no_improve = 0
            if no_improve > 400:
                no_improve = 0
                if len(cur_code) > best_size + 3:
                    cur_code = best_code[:]
                    cur_cover = best_cover
                    cur_set = set(cur_code)
        # final cleanup
        bs = set(best_code)
        for c in list(bs):
            if best_cover & ~covers[c] == all_mask:
                bs.discard(c)
                best_cover = best_cover & ~covers[c]
        best_code = list(bs)
        return best_code, best_cover

    best_final_code = None
    best_final_size = total + 1

    while time.monotonic() < deadline - 2.0:
        code, covered = greedy_cover()
        remaining = all_mask & ~covered
        while remaining:
            for c in range(total):
                if covers[c] & remaining:
                    code.append(c)
                    covered |= covers[c]
                    remaining = all_mask & ~covered
                    break
        code, covered = local_search(code, covered)
        if len(code) < best_final_size:
            best_final_size = len(code)
            best_final_code = code[:]

    if best_final_code is None:
        best_final_code = list(range(0, total, q))
    result = []
    for idx in best_final_code:
        word = [0] * n
        t = idx
        for i in range(n):
            word[i] = t % q
            t //= q
        result.append(word)
    return result
