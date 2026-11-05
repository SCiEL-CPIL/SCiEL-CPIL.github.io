import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # Fallback for non-specific cases
    if q != 3 or n != 6 or radius != 1:
        total = q ** n
        if total > 10000:
            return [[0] * n]
        coords = []
        for i in range(total):
            c = [0] * n
            x = i
            for j in range(n - 1, -1, -1):
                c[j] = x % q
                x //= q
            coords.append(c)
        
        nbrs = []
        for c in coords:
            nbs = [c]
            for i in range(n):
                orig = c[i]
                for v in range(q):
                    if v != orig:
                        nc = c[:]
                        nc[i] = v
                        nbs.append(nc)
            nbrs.append(nbs)
            
        coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
        nbr_idx = []
        for nbs in nbrs:
            nbr_idx.append([coord_to_idx[tuple(nb)] for nb in nbs])
        
        uncovered = np.ones(total, dtype=np.int8)
        code = []
        while uncovered.sum() > 0:
            if time.time() > deadline:
                break
            counts = np.sum(uncovered[np.array(nbr_idx)], axis=1)
            best = int(np.argmax(counts))
            if counts[best] == 0:
                break
            code.append(best)
            uncovered[nbr_idx[best]] = 0
        return [coords[i] for i in code]

    # Specific case: K_3(6,1)
    total = 729
    powers = [1, 3, 9, 27, 81, 243]
    coords = [[(i // p) % 3 for p in powers] for i in range(total)]
    
    # Precompute neighbors as indices
    nbr = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        nbr[i, 0] = i
        k = 1
        for p in range(6):
            base = (i // powers[p]) % 3
            for alt in range(3):
                if alt != base:
                    nbr[i, k] = i + (alt - base) * powers[p]
                    k += 1

    def get_direct_sum():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]

    ds = get_direct_sum()
    
    # Initialize with Greedy
    uncovered = np.ones(total, dtype=np.int8)
    code = []
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        counts = np.sum(uncovered[nbr], axis=1)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        code.append(best)
        uncovered[nbr[best]] = 0
        
    def try_reduce(cw):
        if not cw:
            return cw
        cov = np.zeros(total, dtype=np.int32)
        for idx in cw:
            cov[nbr[idx]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(cw) - 1, -1, -1):
                idx = cw[i]
                if np.min(cov[nbr[idx]]) >= 2:
                    cov[nbr[idx]] -= 1
                    cw.pop(i)
                    changed = True
                    break
        return cw

    code = try_reduce(code)
    best_size = len(code)
    best_code = code[:]
    
    # Local Search: Remove and Repair with Simulated Annealing acceptance
    rng = np.random.default_rng(42)
    no_improve = 0
    max_no_improve = 2000
    temperature = 1.0
    cooling = 0.999
    
    while time.time() < deadline - 2.0 and no_improve < max_no_improve:
        if not code:
            break
            
        cov = np.zeros(total, dtype=np.int32)
        for idx in code:
            cov[nbr[idx]] += 1
            
        i = int(rng.integers(len(code)))
        c_remove = code[i]
        code.pop(i)
        cov[nbr[c_remove]] -= 1
        
        uncovered_mask = (cov == 0)
        
        if not np.any(uncovered_mask):
            code = try_reduce(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                no_improve = 0
            else:
                no_improve += 1
            continue
            
        # Repair: Find best word to add back
        counts_add = np.sum(uncovered_mask[nbr], axis=1)
        best_add = int(np.argmax(counts_add))
        
        current_set = set(code)
        if best_add not in current_set:
            code.append(best_add)
            cov[nbr[best_add]] += 1
            code = try_reduce(code)
            
            new_size = len(code)
            if new_size < best_size:
                best_size = new_size
                best_code = code[:]
                no_improve = 0
                temperature = 1.0
            elif new_size == best_size:
                # Accept equal-size moves with probability based on temperature
                if rng.random() < np.exp(-1.0 / temperature):
                    no_improve += 1
                else:
                    # Revert to best_code
                    code = best_code[:]
                    no_improve += 1
            else:
                # Revert to best_code
                code = best_code[:]
                no_improve += 1
        else:
            code.append(c_remove)
            cov[nbr[c_remove]] += 1
            no_improve += 1
            
        temperature *= cooling

    # If we found a code smaller than 81, use it.
    if best_code and len(best_code) <= 81:
        return [coords[idx] for idx in best_code]
    
    return ds
