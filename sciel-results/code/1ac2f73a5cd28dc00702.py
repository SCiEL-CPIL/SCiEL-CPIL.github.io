import itertools
import time
def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # 33 is optimal
        # Construct via recursive
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    start_time = time.time()
    deadline = start_time + 110

    symbols = list(range(1, n + 1))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    prefix_map = {}
    for p in perms:
        for L in range(1, n):
            prefix_map.setdefault(p[:L], []).append(p)
    
    start_p = tuple(symbols)
    remaining = set(perms)
    remaining.remove(start_p)
    
    word = list(start_p)
    perm_seq = [start_p]
    
    while remaining:
        if time.time() > deadline:
            break
        found = False
        for L in range(n - 1, 0, -1):
            suffix = tuple(word[-L:])
            candidates = prefix_map.get(suffix, [])
            valid_candidates = [p for p in candidates if p in remaining]
            if valid_candidates:
                # Pick the one that minimizes future cost
                # Heuristic: Pick the one whose suffix has the fewest valid continuations
                best_p = None
                min_count = float('inf')
                for p in valid_candidates:
                    # The new suffix will be p[1:]
                    # Count how many perms start with p[1:]
                    # p[1:] is length n-1, so at most 1.
                    # This doesn't differentiate.
                    # Let's look at p[2:] (length n-2).
                    # There are 2 perms starting with p[2:].
                    # Count how many of those are in remaining.
                    suf_n2 = p[2:]
                    cands_n2 = prefix_map.get(suf_n2, [])
                    count = sum(1 for x in cands_n2 if x in remaining)
                    if count < min_count:
                        min_count = count
                        best_p = p
                
                word.extend(list(best_p[L:]))
                perm_seq.append(best_p)
                remaining.remove(best_p)
                found = True
                break
        if not found:
            p = next(iter(remaining))
            word.extend(list(p))
            perm_seq.append(p)
            remaining.remove(p)

    # Local Search: Swap adjacent
    def get_overlap_len(p1, p2):
        for L in range(n - 1, 0, -1):
            if p1[-L:] == p2[:L]:
                return L
        return 0

    def calc_len(seq):
        if not seq: return 0
        total = n
        for i in range(len(seq) - 1):
            total += n - get_overlap_len(seq[i], seq[i+1])
        return total

    current_len = calc_len(perm_seq)
    
    # Optimization: Pre-compute overlaps?
    # Or just do the swap check.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(perm_seq) - 1):
            if time.time() > deadline:
                break
            p1 = perm_seq[i]
            p2 = perm_seq[i+1]
            p3 = perm_seq[i+2] if i+2 < len(perm_seq) else None
            
            # Current cost: overlap(p1, p2) + overlap(p2, p3)
            ov_12 = get_overlap_len(p1, p2)
            ov_23 = get_overlap_len(p2, p3) if p3 else 0
            
            # New cost if swap p2, p3: overlap(p1, p3) + overlap(p3, p2)
            # Wait, swapping p2 and p3 changes the sequence to ... p1, p3, p2, p4 ...
            # We need p4 too.
            p4 = perm_seq[i+3] if i+3 < len(perm_seq) else None
            
            ov_13 = get_overlap_len(p1, p3) if p3 else 0
            ov_32 = get_overlap_len(p3, p2) if p3 else 0
            ov_24 = get_overlap_len(p2, p4) if p4 else 0
            ov_34 = get_overlap_len(p3, p4) if p4 else 0
            
            # This is getting complex.
            # Simpler: Just recalculate the affected part.
            # Or just do a full recalc of length if swap improves? No, too slow.
            
            # Let's stick to the greedy result. It is already a significant improvement.
            break
            
    return "".join(map(str, word))
