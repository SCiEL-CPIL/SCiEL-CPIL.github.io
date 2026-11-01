import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121" # 12, 21. Len 3. Seed: 1!+2!=3.
    if n == 3:
        # Seed: 1!+2!+3! = 1+2+6=9.
        # Optimal?
        # Cycles: (123, 231, 312), (132, 321, 213).
        # Greedy:
        # Start 123. L=123.
        # C1: 231 (ov 2), 312 (ov 1), 123 (visited). Best 231.
        # Append 231. L=231.
        # C1 done? No, we visit whole cycle.
        # My algo visits whole cycle.
        # So from 123, we go 123 -> 231 -> 312.
        # L=312.
        # Next cycle C2: 132, 321, 213.
        # Overlap with 312:
        # 132: 312 vs 132. Suff 12, Pref 13. No. Suff 2, Pref 1. No. Ov 0?
        # 321: 312 vs 321. Suff 12, Pref 32. No. Suff 2, Pref 3. No. Ov 0?
        # 213: 312 vs 213. Suff 12, Pref 21. No. Suff 2, Pref 2. Yes (1). Ov 1.
        # Best is 213 (ov 1).
        # Append 213 -> 132 -> 321.
        # Word: 123 21 3 213 1 321?
        # 12321321321?
        # Let's just run the algo.
        pass

    # Generate permutations
    symbols = list(range(1, n + 1))
    all_perms = list(itertools.permutations(symbols))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Build cycles
    visited = [False] * len(all_perms)
    cycles = []
    
    for i, p in enumerate(all_perms):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                # Next is shift left
                curr = perm_to_idx[tuple(all_perms[curr][1:] + all_perms[curr][:1])]
            cycles.append(cycle)
            
    # Precompute shift indices for fast traversal?
    # cycle is list of indices.
    # To get next in cycle: (pos + 1) % n
    
    # Greedy TSP on cycles
    best_word = None
    best_len = float('inf')
    
    num_starts = 20
    random.seed(42)
    
    for start_iter in range(num_starts):
        if time.time() > deadline:
            break
            
        # Pick random starting cycle and position
        start_cycle_idx = random.randint(0, len(cycles) - 1)
        start_pos_in_cycle = random.randint(0, n - 1)
        
        word = []
        # Add first cycle
        c = cycles[start_cycle_idx]
        # Rotate cycle to start at start_pos_in_cycle
        # The cycle list is in order of shifts.
        # We need to start at c[start_pos_in_cycle] and follow shifts.
        seq = []
        for k in range(n):
            idx_in_c = (start_pos_in_cycle + k) % n
            seq.append(all_perms[c[idx_in_c]])
        
        word.extend(seq[0])
        for p in seq[1:]:
            # Append with max overlap
            # Since they are in cycle, overlap is n-1
            word.extend(p[-1:]) # Just the last char?
            # Wait, overlap n-1 means suffix n-1 of prev == prefix n-1 of curr.
            # So we only append the last char of curr.
            
        last_perm = seq[-1]
        
        unvisited_cycles = set(range(len(cycles)))
        unvisited_cycles.remove(start_cycle_idx)
        
        while unvisited_cycles:
            if time.time() > deadline:
                break
                
            best_overlap = -1
            best_cycle = None
            best_entry_pos = -1
            best_entry_perm = None
            
            for ci in unvisited_cycles:
                c = cycles[ci]
                # Find best entry in this cycle
                for pos in range(n):
                    p = all_perms[c[pos]]
                    # Calculate overlap between last_perm and p
                    # Suffix of last_perm, Prefix of p
                    max_ov = 0
                    for k in range(n - 1, -1, -1):
                        if last_perm[-k:] == p[:k]:
                            max_ov = k
                            break
                    if max_ov > best_overlap:
                        best_overlap = max_ov
                        best_cycle = ci
                        best_entry_pos = pos
                        best_entry_perm = p
                        
            if best_cycle is None:
                break # Should not happen
                
            unvisited_cycles.remove(best_cycle)
            
            # Append the cycle starting at best_entry_pos
            c = cycles[best_cycle]
            for k in range(n):
                idx_in_c = (best_entry_pos + k) % n
                p = all_perms[c[idx_in_c]]
                if k == 0:
                    # Append with overlap best_overlap
                    # last_perm suffix best_overlap == p prefix best_overlap
                    # So append p[best_overlap:]
                    word.extend(p[best_overlap:])
                else:
                    # Overlap is n-1
                    word.extend(p[-1:])
                
                last_perm = p
            
        if len(word) < best_len:
            best_len = len(word)
            best_word = word.copy()
            
    if best_word is None:
        # Fallback to seed
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
        
    return "".join(map(str, best_word))
