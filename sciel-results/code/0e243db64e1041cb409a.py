    def local_search(blocks, time_limit):
        if blocks is None:
            return None
        cur = list(blocks)
        n = len(cur)
        masks = [block_mask(b) for b in cur]
        total = 0
        for m in masks:
            total |= m
        if total != allmask:
            return cur
        repl = 0
        max_repl = 40
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit or repl >= max_repl:
                    break
                others = 0
                for j in range(n):
                    if j != i:
                        others |= masks[j]
                gap = allmask & ~others
                if gap == 0:
                    cur.pop(i)
                    masks.pop(i)
                    n -= 1
                    improved = True
                    break
                rng = random.Random((i * 7919 + n) & 0xffffffff)
                block = list(sub_by_idx[(gap & -gap).bit_length() - 1])
                bset = set(block)
                while len(block) < k:
                    bx = None
                    bg = -1
                    for x in pts:
                        if x in bset:
                            continue
                        d = bit_xS[x]
                        m = 0
                        for S in itertools.combinations(block, t - 1):
                            m |= d[S]
                        g = (m & gap).bit_count()
                        if g > bg or (g == bg and bx is not None and rng.random() < 0.5):
                            bg = g
                            bx = x
                    if bx is None:
                        break
                    block.append(bx)
                    bset.add(bx)
                block.sort()
                newm = block_mask(block)
                if (newm & gap) == gap and newm != masks[i]:
                    cur[i] = list(block)
                    masks[i] = newm
                    repl += 1
                    improved = True
        return cur
