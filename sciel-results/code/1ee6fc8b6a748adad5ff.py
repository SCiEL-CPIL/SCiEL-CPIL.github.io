import math
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # Safety margin before 90s kill

    max_m = 128

    # Precompute gcd for even moduli (2..128 step 2).
    evens = list(range(2, max_m + 1, 2))
    n_even = len(evens)
    gcds = [[0] * n_even for _ in range(n_even)]
    for i in range(n_even):
        mi = evens[i]
        row = gcds[i]
        for j in range(i, n_even):
            g = math.gcd(mi, evens[j])
            row[j] = g
            gcds[j][i] = g

    # Ordering: descending modulus (largest first) for a strong forward-checking order.
    order = list(range(n_even - 1, -1, -1))  # indices into evens, largest modulus first

    def valid_residues(k, chosen):
        """Return list of residues r in [0, m) for modulus evens[k] that are disjoint
        from every already-chosen class. chosen is a list of (r, m) pairs."""
        m = evens[k]
        half = m // 2
        forbidden = bytearray(half)  # forbidden[a] for a in 0..half-1
        for (r_i, m_i) in chosen:
            g = gcds[k][m_i_index[m_i]]
            # r is invalid if r % g == r_i % g.  r = 2a, so 2a % g == r_i % g
            # <=> a % (g/2) == (r_i % g) // 2   (since g even and r_i%g even).
            h = g // 2
            b = (r_i % g) // 2
            for a in range(b, half, h):
                forbidden[a] = 1
        res = []
        for a in range(half):
            if not forbidden[a]:
                res.append(2 * a)
        return res

    # Map modulus value -> index in evens (for gcd lookup).
    m_i_index = {m: i for i, m in enumerate(evens)}

    # ---------- Fast deterministic greedy (descending + MRV) ----------
    def greedy_deterministic():
        chosen = []
        remaining = list(order)
        while remaining:
            # MRV: pick the remaining modulus with the fewest valid residues.
            best_k = None
            best_valid = None
            best_count = None
            for k in remaining:
                vr = valid_residues(k, chosen)
                c = len(vr)
                if c == 0:
                    return None  # dead end
                if best_count is None or c < best_count:
                    best_count = c
                    best_k = k
                    best_valid = vr
                    if c == 1:
                        break
            chosen.append((best_valid[0], evens[best_k]))
            remaining.remove(best_k)
        return chosen

    result = greedy_deterministic()
    if result is not None and len(result) == n_even:
        # Success: 64 pairwise-disjoint classes with distinct moduli.
        # Try local search to see if we can do better by adding odd moduli
        # or by swapping residues to free up space
        best_solution = [[r, m] for (r, m) in result]
        best_count = len(best_solution)
        
        # Local search: try to add odd moduli or improve via swaps
        used_mods = set(m for _, m in best_solution)
        
        for iteration in range(20000):
            if time.time() > deadline - 3.0:
                break
            
            # Try to add an unused modulus (odd or even not yet used)
            added = False
            for m in range(2, max_m + 1):
                if m in used_mods:
                    continue
                if m % 2 == 0 and m in m_i_index:
                    # Even modulus, use fast path
                    k = m_i_index[m]
                    vr = valid_residues(k, [(r, mi) for r, mi in best_solution])
                    if vr:
                        r = vr[0]
                        best_solution.append([r, m])
                        used_mods.add(m)
                        best_count += 1
                        added = True
                        break
                else:
                    # Odd modulus, need general check
                    # Check gcd with all used moduli
                    compatible = True
                    for _, m_i in best_solution:
                        if math.gcd(m, m_i) == 1:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    # Find valid residue
                    valid_r = None
                    for r in range(m):
                        ok = True
                        for r_i, m_i in best_solution:
                            g = math.gcd(m, m_i)
                            if g > 1 and r % g == r_i % g:
                                ok = False
                                break
                        if ok:
                            valid_r = r
                            break
                    if valid_r is not None:
                        best_solution.append([valid_r, m])
                        used_mods.add(m)
                        best_count += 1
                        added = True
                        break
            
            if added:
                continue
            
            # Try a swap: remove one, try to add another
            if not best_solution:
                break
            
            rm_idx = random.randint(0, len(best_solution) - 1)
            r_old, m_old = best_solution[rm_idx]
            new_solution = [p for i, p in enumerate(best_solution) if i != rm_idx]
            new_used = set(m for _, m in new_solution)
            
            # Try several unused moduli
            unused = [m for m in range(2, max_m + 1) if m not in new_used]
            if not unused:
                continue
            
            tried = random.sample(unused, min(10, len(unused)))
            best_swap = None
            best_swap_count = len(new_solution)
            
            for cand in tried:
                if time.time() > deadline - 3.0:
                    break
                # Check gcd compatibility
                compatible = True
                for _, m_i in new_solution:
                    if math.gcd(cand, m_i) == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                # Find valid residue
                valid_r = None
                if cand % 2 == 0 and cand in m_i_index:
                    k = m_i_index[cand]
                    vr = valid_residues(k, [(r, mi) for r, mi in new_solution])
                    if vr:
                        valid_r = vr[0]
                else:
                    for r in range(cand):
                        ok = True
                        for r_i, m_i in new_solution:
                            g = math.gcd(cand, m_i)
                            if g > 1 and r % g == r_i % g:
                                ok = False
                                break
                        if ok:
                            valid_r = r
                            break
                
                if valid_r is None:
                    continue
                
                trial = new_solution + [[valid_r, cand]]
                trial_used = new_used | {cand}
                
                # Try to add more
                for m2 in range(2, max_m + 1):
                    if time.time() > deadline - 3.0:
                        break
                    if m2 in trial_used:
                        continue
                    comp2 = True
                    for _, m_i2 in trial:
                        if math.gcd(m2, m_i2) == 1:
                            comp2 = False
                            break
                    if not comp2:
                        continue
                    vr2 = None
                    if m2 % 2 == 0 and m2 in m_i_index:
                        k2 = m_i_index[m2]
                        vrs = valid_residues(k2, [(r, mi) for r, mi in trial])
                        if vrs:
                            vr2 = vrs[0]
                    else:
                        for r in range(m2):
                            ok = True
                            for r_i, m_i in trial:
                                g = math.gcd(m2, m_i)
                                if g > 1 and r % g == r_i % g:
                                    ok = False
                                    break
                            if ok:
                                vr2 = r
                                break
                    if vr2 is not None:
                        trial.append([vr2, m2])
                        trial_used.add(m2)
                
                if len(trial) > best_swap_count:
                    best_swap_count = len(trial)
                    best_swap = trial
            
            if best_swap is not None:
                best_solution = best_swap
                best_count = len(best_solution)
                used_mods = set(m for _, m in best_solution)
        
        return best_solution

    # ---------- Lightweight backtracking (only if greedy failed) ----------
    def backtrack(remaining, chosen):
        if time.time() > deadline - 3.0:
            return None
        if not remaining:
            return chosen
        best_k = None
        best_valid = None
        best_count = None
        for k in remaining:
            vr = valid_residues(k, chosen)
            c = len(vr)
            if c == 0:
                return None
            if best_count is None or c < best_count:
                best_count = c
                best_k = k
                best_valid = vr
                if c == 1:
                    break
        remaining.remove(best_k)
        for r in best_valid:
            res = backtrack(remaining, chosen + [(r, evens[best_k])])
            if res is not None:
                return res
        remaining.append(best_k)
        return None

    remaining = list(order)
    bt = backtrack(remaining, [])
    if bt is not None and len(bt) == n_even:
        return [[r, m] for (r, m) in bt]

    # ---------- Fallback: randomized greedy to guarantee a valid nonempty result ----------
    best = []
    for _ in range(20000):
        if time.time() > deadline - 1.0:
            break
        chosen = []
        shuffled = evens[:]
        random.shuffle(shuffled)
        for m in shuffled:
            k = m_i_index[m]
            vr = valid_residues(k, chosen)
            if vr:
                chosen.append((random.choice(vr), m))
        if len(chosen) > len(best):
            best = chosen
            if len(best) == n_even:
                break

    return [[r, m] for (r, m) in best]
