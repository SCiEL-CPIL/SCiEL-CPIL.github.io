"""B_3[1] construction: bitset-accelerated greedy with skeleton start and swap improvement."""
import time
import random

def construct():
    deadline = time.time() + 82.0
    N = 4096
    MAX_SUM = 3 * N  # Max possible triple sum = 3*4096 = 12288
    
    # Use a bitset for O(1) membership checks
    # sums_bitset[s] = True if sum s is already taken
    sums_bitset = bytearray(MAX_SUM + 1)
    
    def can_add(x, selected):
        """Check if x can be added to selected without violating B_3[1] property."""
        # 3x
        if sums_bitset[3 * x]:
            return False
        # 2x + s for s in selected
        for s in selected:
            if sums_bitset[2 * x + s]:
                return False
        # x + 2s for s in selected
        for s in selected:
            if sums_bitset[x + 2 * s]:
                return False
        # x + s + t for s <= t in selected
        n = len(selected)
        for i in range(n):
            s = selected[i]
            for j in range(i, n):
                t = selected[j]
                if sums_bitset[x + s + t]:
                    return False
        return True

    def add_x(x, selected):
        """Add x to selected and update sums."""
        selected.append(x)
        sums_bitset[3 * x] = 1
        for s in selected[:-1]:
            sums_bitset[2 * x + s] = 1
            sums_bitset[x + 2 * s] = 1
        n = len(selected)
        for i in range(n - 1):
            s = selected[i]
            for j in range(i, n - 1):
                t = selected[j]
                sums_bitset[x + s + t] = 1

    def remove_x(x, selected):
        """Remove x from selected and update sums."""
        idx = selected.index(x)
        selected.pop(idx)
        # Remove all sums involving x
        sums_bitset[3 * x] = 0
        for s in selected:
            sums_bitset[2 * x + s] = 0
            sums_bitset[x + 2 * s] = 0
        n = len(selected)
        for i in range(n):
            s = selected[i]
            for j in range(i, n):
                t = selected[j]
                sums_bitset[x + s + t] = 0

    def get_sums_set(selected):
        """Rebuild the sums set from selected (for verification)."""
        sums = set()
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(selected[i] + selected[j] + selected[k])
        return sums

    # Phase 1: Start with sparse skeleton (powers of 4) then greedy fill
    selected = []
    
    # Add powers of 4 as skeleton: 1, 4, 16, 64, 256, 1024, 4096
    # These are guaranteed collision-free for triples
    skeleton = [1, 4, 16, 64, 256, 1024]
    for x in skeleton:
        if x <= N and not can_add(x, selected):
            pass
        else:
            add_x(x, selected)
    
    # Now greedily fill in gaps from 1 to N
    sel_set = set(selected)
    for x in range(1, N + 1):
        if time.time() > deadline - 15.0:
            break
        if x not in sel_set and can_add(x, selected):
            add_x(x, selected)
            sel_set.add(x)
    
    selected.sort()
    best_selected = selected[:]
    best_size = len(selected)
    
    # Phase 2: Try to improve via swaps
    if time.time() < deadline - 5.0:
        sel_set = set(selected)
        # Candidates sorted descending - larger values often easier to place
        candidates = sorted([x for x in range(1, N + 1) if x not in sel_set], reverse=True)
        
        cand_attempts = {c: 0 for c in candidates}
        max_attempts_per_cand = 5
        
        improved = True
        while improved and time.time() < deadline - 4.0:
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 4.0:
                    break
                orig = selected[i]
                for idx, cand in enumerate(candidates):
                    if time.time() > deadline - 4.0:
                        break
                    if cand_attempts[cand] >= max_attempts_per_cand:
                        continue
                    if cand <= orig:
                        continue
                    # Remove orig, try adding cand
                    remove_x(orig, selected)
                    if can_add(cand, selected):
                        add_x(cand, selected)
                        candidates.pop(idx)
                        improved = True
                        break
                    else:
                        # Restore
                        add_x(orig, selected)
                        cand_attempts[cand] += 1
                selected.sort()
        
        if len(selected) > best_size:
            best_selected = selected[:]
            best_size = len(selected)
    
    # Phase 3: Try adding more elements with randomized orderings
    if time.time() < deadline - 2.0:
        for attempt in range(10):
            if time.time() > deadline - 2.0:
                break
            
            sel_set = set(best_selected)
            candidates = [x for x in range(N, 0, -1) if x not in sel_set]
            if attempt > 0:
                random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 2.0:
                    break
                # Try adding x to a copy of best_selected
                temp_selected = best_selected[:]
                # We need to check against the current sums_bitset state
                # Save current state
                saved_bitset = bytearray(sums_bitset)
                # But sums_bitset reflects 'selected' not 'best_selected'
                # Let's rebuild for best_selected
                temp_sums = bytearray(MAX_SUM + 1)
                n = len(temp_selected)
                for i in range(n):
                    for j in range(i, n):
                        for k in range(j, n):
                            temp_sums[temp_selected[i] + temp_selected[j] + temp_selected[k]] = 1
                
                # Check if x can be added
                ok = True
                if temp_sums[3 * x]:
                    ok = False
                else:
                    for s in temp_selected:
                        if temp_sums[2 * x + s] or temp_sums[x + 2 * s]:
                            ok = False
                            break
                if ok:
                    n = len(temp_selected)
                    for i in range(n):
                        s = temp_selected[i]
                        for j in range(i, n):
                            t = temp_selected[j]
                            if temp_sums[x + s + t]:
                                ok = False
                                break
                        if not ok:
                            break
                
                if ok:
                    temp_selected.append(x)
                    if len(temp_selected) > best_size:
                        best_selected = temp_selected
                        best_size = len(temp_selected)
                        # Rebuild sums_bitset for the new best
                        for i in range(len(best_selected)):
                            for j in range(i, len(best_selected)):
                                for k in range(j, len(best_selected)):
                                    sums_bitset[best_selected[i] + best_selected[j] + best_selected[k]] = 1
                        break
    
    return best_selected[:80]
