import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    # Precompute APs for each index (0-indexed)
    # ap_list[i] contains tuples of 4 indices (1-indexed) of APs that include i+1
    ap_list = [[] for _ in range(n)]
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            for idx in ap:
                ap_list[idx - 1].append(ap)

    def check_change(coloring, idx, new_color):
        """Check if changing coloring[idx] to new_color maintains validity for all APs containing idx."""
        for ap in ap_list[idx]:
            c0 = coloring[ap[0]-1]
            c1 = coloring[ap[1]-1]
            c2 = coloring[ap[2]-1]
            c3 = coloring[ap[3]-1]
            
            if ap[0] == idx + 1: c0 = new_color
            elif ap[1] == idx + 1: c1 = new_color
            elif ap[2] == idx + 1: c2 = new_color
            elif ap[3] == idx + 1: c3 = new_color
            
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True

    def greedy_construct(randomize=False):
        coloring = [0] * n
        for i in range(n):
            forbidden = set()
            for d in range(1, (i + 1) // 3 + 1):
                a = i + 1 - 3 * d
                b = i + 1 - 2 * d
                c = i + 1 - d
                if a < 1: continue
                ca = coloring[a - 1]
                cb = coloring[b - 1]
                cc = coloring[c - 1]
                S = {ca, cb, cc}
                if len(S) <= 2:
                    forbidden.update(S)
            
            candidates = [c for c in range(128) if c not in forbidden]
            if not candidates:
                return None # Should not happen
            
            if randomize and candidates:
                coloring[i] = random.choice(candidates)
            else:
                coloring[i] = candidates[0]
        return coloring

    def local_search(coloring):
        current_colors = set(coloring)
        
        while time.time() < deadline:
            freq = {}
            for c in coloring:
                freq[c] = freq.get(c, 0) + 1
            sorted_colors = sorted(freq.keys(), key=lambda x: freq[x])
            
            improved = False
            for target_color in sorted_colors:
                if target_color not in current_colors:
                    continue
                
                positions = [i for i in range(n) if coloring[i] == target_color]
                orig_coloring = coloring[:]
                
                success = True
                random.shuffle(positions)
                
                for pos in positions:
                    found = False
                    other_colors = [c for c in current_colors if c != target_color]
                    random.shuffle(other_colors)
                    for new_c in other_colors:
                        if check_change(coloring, pos, new_c):
                            coloring[pos] = new_c
                            found = True
                            break
                    if not found:
                        success = False
                        break
                
                if success:
                    current_colors.remove(target_color)
                    improved = True
                else:
                    coloring = orig_coloring[:]
            
            if not improved:
                break
                
        return coloring

    best_coloring = None
    best_count = 128
    
    # Try deterministic greedy first
    coloring = greedy_construct(randomize=False)
    if coloring:
        coloring = local_search(coloring)
        count = len(set(coloring))
        if count < best_count:
            best_count = count
            best_coloring = coloring[:]
    
    # Try randomized greedy restarts if time permits
    while time.time() < deadline - 5.0:
        coloring = greedy_construct(randomize=True)
        if coloring:
            # Quick check if it's already better before expensive local search?
            # Local search is fast enough.
            coloring = local_search(coloring)
            count = len(set(coloring))
            if count < best_count:
                best_count = count
                best_coloring = coloring[:]
                # If we hit a very low count, we can stop early?
                # But we want to minimize.
    
    if best_coloring is None:
        # Fallback to seed-like behavior if something went wrong
        # But greedy_construct should always succeed.
        return [0] * n
        
    return best_coloring
