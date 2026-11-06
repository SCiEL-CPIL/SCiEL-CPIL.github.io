import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_structure(n):
        m = n * (n - 1) // 2
        edge_lookup = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_lookup[i][j] = idx
                edge_lookup[j][i] = idx
                idx += 1

        cycles_list = []
        verts = list(range(n))
        
        # Generate all C6 cycles
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_lookup[u][v])
                cycles_list.append(cycle_edges)
        
        N = len(cycles_list)
        if N == 0:
            return [], [], 0

        # Map edge to list of cycle indices
        edge_to_cycles = [[] for _ in range(m)]
        for i in range(N):
            for e in cycles_list[i]:
                edge_to_cycles[e].append(i)
        
        return cycles_list, edge_to_cycles, N

    def spectral_init(n, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        m = n * (n - 1) // 2
        # Simple structured initialization: c(i,j) = (i - j) % 4
        # This is a known good start for many Ramsey problems.
        colors = np.zeros(m, dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                colors[idx] = (i - j) % 4
                idx += 1
        return colors

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)

        cycles_list, edge_to_cycles, N = build_structure(n)

        if N == 0:
            colors = np.zeros(m, dtype=np.int8)
            return {'n': n, 'colors': colors.tolist()}

        colors = spectral_init(n, seed=seed_offset)
        
        # Initialize counts
        # cnt[cycle_idx][color] = number of edges in cycle with that color
        cnt = np.zeros((N, 4), dtype=np.int32)
        is_bad = np.zeros(N, dtype=bool)
        bad_count = 0
        
        for c_idx in range(N):
            ce = cycles_list[c_idx]
            for c in range(4):
                cnt[c_idx, c] = np.sum(colors[ce] == c)
            if np.any(cnt[c_idx] == 6):
                is_bad[c_idx] = True
                bad_count += 1
                
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        # SA Parameters
        T_start = 2.0
        T_end = 0.01
        start_iter_time = time.time()
        
        while time.time() < start_iter_time + time_budget:
            if time.time() > deadline - 5:
                break

            e = np.random.randint(0, m)
            c_old = colors[e]
            
            # Pick new color
            c_new = np.random.randint(0, 4)
            while c_new == c_old:
                c_new = np.random.randint(0, 4)

            cycs = edge_to_cycles[e]
            
            # Calculate Delta
            delta = 0
            # Store state to revert if needed
            old_states = []
            
            for c_idx in cycs:
                # Check if it was bad
                was_bad = is_bad[c_idx]
                
                # Update counts
                old_cnt_old = cnt[c_idx, c_old]
                old_cnt_new = cnt[c_idx, c_new]
                
                cnt[c_idx, c_old] -= 1
                cnt[c_idx, c_new] += 1
                
                # Check if it is bad now
                is_bad_now = False
                # Only c_old or c_new can change bad status
                if cnt[c_idx, c_old] == 6:
                    is_bad_now = True
                elif cnt[c_idx, c_new] == 6:
                    is_bad_now = True
                
                if was_bad and not is_bad_now:
                    delta -= 1
                elif not was_bad and is_bad_now:
                    delta += 1
                
                is_bad[c_idx] = is_bad_now
                
                old_states.append((c_idx, was_bad, old_cnt_old, old_cnt_new))

            # Acceptance
            accept = False
            if delta <= 0:
                accept = True
            else:
                elapsed = time.time() - start_iter_time
                progress = min(1.0, elapsed / time_budget)
                T = T_start * (T_end / T_start) ** progress
                
                if delta / T > 50:
                    accept = False
                else:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob

            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
                    if best_bad == 0:
                        return {'n': n, 'colors': colors.tolist()}
            else:
                # Revert
                for c_idx, was_bad, old_cnt_old, old_cnt_new in old_states:
                    is_bad[c_idx] = was_bad
                    cnt[c_idx, c_old] = old_cnt_old
                    cnt[c_idx, c_new] = old_cnt_new

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try targets
    targets = [17, 16, 15]
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        tb = 30 if n == 17 else 15
        result = search_n(n, tb, seed_offset=n)
        if result:
            best_result = result
            return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
