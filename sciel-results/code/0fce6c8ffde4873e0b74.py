# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_T1 = time.time()
_HARD = _T1 + 150.0


def _deadline(margin):
    return (time.time() + margin) < _HARD


def _fallback_base(q, n, radius):
    """Direct-sum of tetracode: 9 * q^(n-4) words, radius-1 for n>=4, q=3."""
    if q != 3 or n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tail_len = n - 4
    out = []
    for tail in itertools.product(range(q), repeat=tail_len):
        for h in tetracode:
            out.append(h + list(tail))
    return out


def _build_neighbors_np(q, n):
    """Return all_words (list of tuples), word_idx (dict), and nrm (list of lists of indices) using NumPy for speed."""
    N = q ** n
    # Generate all words as a 2D numpy array
    arr = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    word_idx = {tuple(arr[i]): i for i in range(N)}
    
    # Build neighbor indices
    nrm = [[] for _ in range(N)]
    for i in range(N):
        w = arr[i]
        nrm[i].append(i)
        for j in range(n):
            for d in (1, 2):
                b = w.copy()
                b[j] = (b[j] + d) % q
                nrm[i].append(word_idx[tuple(b)])
    return arr, word_idx, nrm


def _local_search_swap(q, n, radius, base):
    """
    Perform local search with swapping to minimize code size.
    1. Greedy removal of redundant codewords.
    2. Swap codewords to enable further removals or maintain coverage with better structure.
    """
    if radius < 1:
        return base
    
    N = q ** n
    all_words, word_idx, nrm = _build_neighbors_np(q, n)

    # Initialize selection
    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)
            
    sel_arr = np.array(sel, dtype=np.int32)
    sel_set = set(sel)
    
    # Coverage count
    cov = np.zeros(N, dtype=np.int32)
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1
            
    # Precompute neighbor arrays for vectorized operations if possible, 
    # but lists are fine for n=8, N=6561.
    
    deadline_guard = _HARD - 15.0
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline_guard:
        improved = False
        iterations += 1
        
        # --- Phase 1: Remove redundant codewords ---
        # A codeword is redundant if removing it leaves all words covered (cov[nb] >= 1 for all nb in nrm[ci])
        # i.e., min(cov[nrm[ci]]) >= 1
        # Actually, we want to remove if min(cov[nrm[ci]]) >= 2? No, if min is 1, removing it makes some 0.
        # We want to remove if min(cov[nrm[ci]]) >= 1 is false? No.
        # If min(cov[nrm[ci]]) >= 1, it's covered. If we remove it, those with cov==1 become 0.
        # So we can only remove if min(cov[nrm[ci]]) >= 2? No, that's not right.
        # If min(cov[nrm[ci]]) == 1, removing it creates holes.
        # If min(cov[nrm[ci]]) >= 2, removing it is safe.
        
        # Let's check for safe removals (min cov >= 2)
        removable = []
        for ci in sel:
            if min(cov[nb] for nb in nrm[ci]) >= 2:
                removable.append(ci)
                
        if removable:
            # Remove all safe ones at once? Or one by one? 
            # Removing one might make another safe. 
            # Let's remove one by one to be safe, or batch if independent.
            # For efficiency, let's try to remove as many as possible.
            # Simple approach: iterate and remove if still safe.
            for ci in removable:
                if ci in sel_set:
                    # Check if still safe (cov might have changed if we removed others)
                    if min(cov[nb] for nb in nrm[ci]) >= 2:
                        for nb in nrm[ci]:
                            cov[nb] -= 1
                        sel.remove(ci)
                        sel_set.discard(ci)
                        improved = True
            continue # Restart loop to re-evaluate
            
        # --- Phase 2: Swapping ---
        # If no safe removals, try swapping.
        # For each ci in sel, let U = {nb in nrm[ci] | cov[nb] == 1}.
        # We want to find cj not in sel such that nrm[cj] covers U.
        # If found, replace ci with cj.
        
        # To speed up, we can limit the number of swaps per iteration or sample candidates.
        # Given N=6561, checking all cj for each ci is 6561 * 6561 ~ 43M operations. 
        # In Python, this is too slow. 
        # Optimization: 
        # 1. Identify "critical" words U for ci.
        # 2. Only consider cj that are neighbors of words in U?
        #    If cj covers U, then cj is within distance 1 of all words in U.
        #    Actually, if w in U, w is in nrm[ci]. We need cj such that w in nrm[cj].
        #    So cj must be in nrm[w] for all w in U.
        #    So cj is in intersection of nrm[w] for w in U.
        
        swapped = False
        # Sample a subset of codewords to try swapping to save time
        # Or just iterate all if time permits.
        # Let's iterate all, but break early if time is tight.
        
        for ci in sel:
            if not _deadline(5.0):
                break
                
            U = [nb for nb in nrm[ci] if cov[nb] == 1]
            if not U:
                # Should have been removed in Phase 1? 
                # Maybe min was 1 but we didn't check >=2. 
                # If min cov == 1, U is not empty.
                # If min cov >= 2, U is empty. 
                # So if U is empty, it was safe to remove. Why didn't Phase 1 catch it?
                # Phase 1 checks min >= 2. If min is 1, U is not empty.
                # If min is 0, code is invalid.
                # So U empty implies min >= 2. 
                # So this block should rarely trigger if Phase 1 is thorough.
                continue
                
            # Find candidates cj in intersection of nrm[w] for w in U
            # Start with nrm[U[0]]
            candidates = set(nrm[U[0]])
            for w in U[1:]:
                candidates &= set(nrm[w])
                if not candidates:
                    break
                    
            if not candidates:
                continue
                
            # Remove ci from candidates (we don't want to swap with self)
            candidates.discard(ci)
            
            # We want a cj that is NOT in sel
            # And ideally, adding cj doesn't break anything else (it only adds coverage, so it's safe)
            # But we must ensure cj is not already in sel.
            
            best_cj = -1
            # Heuristic: Pick cj that covers the most currently covered words? 
            # Or just pick the first one not in sel?
            # Picking the first one is fast.
            # To improve quality, maybe pick one that maximizes "redundancy" elsewhere?
            # Let's just pick the first valid one.
            
            for cj in candidates:
                if cj not in sel_set:
                    best_cj = cj
                    break
                    
            if best_cj != -1:
                # Perform swap
                # Remove ci
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                
                # Add cj
                for nb in nrm[best_cj]:
                    cov[nb] += 1
                sel.append(best_cj)
                sel_set.add(best_cj)
                
                improved = True
                swapped = True
                break # Restart loop to re-evaluate removals
                
        if not improved:
            # No swaps found? Try a more aggressive search?
            # If we are stuck, we might be in a local minimum.
            # We can try random swaps or perturbations.
            # For now, if no simple swap works, we stop.
            pass
            
    return [list(all_words[i]) for i in sel]


def construct(q, n, radius):
    global _HARD
    _HARD = time.time() + 150.0
    if n < 0:
        return []
    if n == 0:
        return [[]]

    # --- Build a guaranteed-valid base code ---
    base = _fallback_base(q, n, radius)
    
    # For q=3, n=8, radius=1: try to get a better code via local search
    if q == 3 and n == 8 and radius == 1 and _deadline(100.0):
        # Run local search with swapping
        improved_base = _local_search_swap(3, 8, 1, base)
        if len(improved_base) < len(base):
            base = improved_base
            
    # --- Guarantee validity: cover all words (repair if base was not perfect) ---
    if radius >= 1:
        N = q ** n
        all_words, word_idx, nrm = _build_neighbors_np(q, n)
        cov = np.zeros(N, dtype=np.int32)
        sel = []
        seen = set()
        for cw in base:
            i = word_idx[tuple(cw)]
            if i not in seen:
                seen.add(i)
                sel.append(i)
                for nb in nrm[i]:
                    cov[nb] += 1
        uncovered = np.where(cov == 0)[0]
        # repair: add uncovered words as codewords
        while len(uncovered) > 0 and _deadline(30.0):
            uc = int(uncovered[0])
            if uc not in seen:
                seen.add(uc)
                sel.append(uc)
                for nb in nrm[uc]:
                    cov[nb] += 1
                # recompute remaining uncovered efficiently
                if len(uncovered) > 0:
                    uncovered = np.where(cov == 0)[0]

        base = [list(all_words[i]) for i in sel]

    return [list(c) for c in base]
# EVOLVE-BLOCK-END
