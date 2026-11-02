import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    DEADLINE = 110
    
    def elapsed():
        return time.time() - t0
    
    def ok():
        return elapsed() < DEADLINE
    
    # Precompute
    all_subsets = list(itertools.combinations(range(v), t))
    n = len(all_subsets)
    idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Greedy with better heuristic
    def greedy():
        cov = np.zeros(n, dtype=np.bool_)
        blocks = []
        
        while np.any(~cov) and ok():
            unc = np.where(~cov)[0]
            if len(unc) == 0:
                break
            # Pick uncovered subset
            ti = unc[0]
            target = all_subsets[ti]
            blk = list(target)
            bs = set(blk)
            
            while len(blk) < k and ok():
                bx, bg = -1, -1
                for x in range(v):
                    if x in bs:
                        continue
                    g = 0
                    for r in itertools.combinations(blk, t-1):
                        s = tuple(sorted(r + (x,)))
                        ii = idx.get(s)
                        if ii is not None and not cov[ii]:
                            g += 1
                    if g > bg:
                        bg, bx = g, x
                if bx >= 0:
                    blk.append(bx)
                    bs.add(bx)
                else:
                    av = [x for x in range(v) if x not in bs]
                    if av:
                        bx = random.choice(av)
                        blk.append(bx)
                        bs.add(bx)
                    else:
                        break
            
            blk.sort()
            blocks.append(blk)
            for s in itertools.combinations(blk, t):
                ii = idx.get(s)
                if ii is not None:
                    cov[ii] = True
        
        return blocks, cov
    
    # Remove redundant
    def dedup(blocks, cov):
        if not np.all(cov):
            return blocks, cov
        cc = np.zeros(n, dtype=np.int32)
        for b in blocks:
            for s in itertools.combinations(b, t):
                cc[idx[s]] += 1
        ch = True
        while ch and ok():
            ch = False
            for i in range(len(blocks)-1, -1, -1):
                b = blocks[i]
                ok_rm = all(cc[idx[s]] > 1 for s in itertools.combinations(b, t))
                if ok_rm:
                    for s in itertools.combinations(b, t):
                        cc[idx[s]] -= 1
                    blocks.pop(i)
                    ch = True
                    break
        cov = np.zeros(n, dtype=np.bool_)
        for b in blocks:
            for s in itertools.combinations(b, t):
                cov[idx[s]] = True
        return blocks, cov
    
    # Local search
    def lsearch(blocks, cov, cc, budget=50):
        if not np.all(cov) or not ok():
            return blocks
        best = [list(b) for b in blocks]
        blen = len(blocks)
        end = t0 + budget
        while time.time() < end and ok():
            # Try remove
            rm = False
            for i in range(len(blocks)-1, -1, -1):
                b = blocks[i]
                if all(cc[idx[s]] > 1 for s in itertools.combinations(b, t)):
                    for s in itertools.combinations(b, t):
                        cc[idx[s]] -= 1
                    blocks.pop(i)
                    rm = True
                    if len(blocks) < blen:
                        best = [list(b) for b in blocks]
                        blen = len(blocks)
                    break
            if not rm and blocks:
                # Try swap
                bi = random.randint(0, len(blocks)-1)
                b = blocks[bi]
                pi = random.randint(0, len(b)-1)
                op = b[pi]
                bs = set(b)
                bx, bd = -1, 0
                for x in range(v):
                    if x in bs:
                        continue
                    d = 0
                    for s in itertools.combinations(b, t):
                        if op in s and cc[idx[s]] <= 1:
                            d += 1
                    if d < bd or (d == bd and bx < 0):
                        bd, bx = d, x
                if bx >= 0 and bd == 0:
                    b[pi] = bx
                    b.sort()
                    ob = list(b)
                    ob[ob.index(bx)] = op
                    ob.sort()
                    for s in itertools.combinations(ob, t):
                        cc[idx[s]] -= 1
                    for s in itertools.combinations(b, t):
                        cc[idx[s]] += 1
                    bad = False
                    for s in itertools.combinations(ob, t):
                        if cc[idx[s]] == 0:
                            bad = True
                            break
                    if not bad:
                        if len(blocks) < blen:
                            best = [list(b) for b in blocks]
                            blen = len(blocks)
                    else:
                        b[pi] = op
                        b.sort()
                        for s in itertools.combinations(b, t):
                            cc[idx[s]] -= 1
                        for s in itertools.combinations(ob, t):
                            cc[idx[s]] += 1
        return best
    
    # Run
    blocks, cov = greedy()
    blocks, cov = dedup(blocks, cov)
    
    if np.all(cov) and ok():
        cc = np.zeros(n, dtype=np.int32)
        for b in blocks:
            for s in itertools.combinations(b, t):
                cc[idx[s]] += 1
        better = lsearch(blocks, cov, cc, budget=40)
        if len(better) < len(blocks):
            blocks = better
            cov = np.zeros(n, dtype=np.bool_)
            for b in blocks:
                for s in itertools.combinations(b, t):
                    cov[idx[s]] = True
    
    if not np.all(cov):
        blocks, cov = greedy()
        blocks, cov = dedup(blocks, cov)
    
    return [list(b) for b in blocks]
