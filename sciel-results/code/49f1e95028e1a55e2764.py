"""
Vectorized class-based local search for 4-AP 3-color constraint.
Uses numpy for fast constraint evaluation and color-class swap moves.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80
    N = 128

    # Precompute all 4-APs as index arrays for vectorized ops
    ap_list = []
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Store as numpy array: shape (num_aps, 4)
    ap_arr = np.array(ap_list, dtype=np.int32)
    num_aps = ap_arr.shape[0]
    
    # For each position, which APs does it appear in?
    pos_ap_map = [[] for _ in range(N)]
    for ap_id, ap in enumerate(ap_list):
        for pos in ap:
            pos_ap_map[pos].append(ap_id)
    
    def count_violations_vec(coloring):
        """Vectorized violation count: number of APs with <3 distinct colors."""
        # coloring is a numpy array of shape (N,)
        # For each AP, get the 4 colors
        c0 = coloring[ap_arr[:, 0]]
        c1 = coloring[ap_arr[:, 1]]
        c2 = coloring[ap_arr[:, 2]]
        c3 = coloring[ap_arr[:, 3]]
        
        # Count distinct colors: an AP violates if it has <=2 distinct colors
        # This means: (c0==c1==c2==c3) or (exactly 2 colors used)
        # Exactly 2 colors means: all pairs are either equal or the set has size 2
        # Simpler: count how many of the 4 values are equal to c0, and how many equal c1 (if c1!=c0)
        
        # An AP has <3 distinct colors iff:
        # - all 4 same (monochromatic), OR
        # - exactly 2 distinct values used
        # 
        # Check: let unique = {c0,c1,c2,c3}. |unique| < 3 iff:
        # (c0==c1 and c0==c2 and c0==c3) OR
        # (c0==c1 and c0==c2 and c1==c3) OR
        # (c0==c1 and c2==c3) OR
        # (c0==c2 and c1==c3) OR
        # (c0==c3 and c1==c2) OR
        # (c1==c2 and c1==c3) OR
        # (c1==c3 and c2==c3) OR
        # (c2==c3 and c0==c1) -- already covered
        # 
        # Simpler formulation: |{c0,c1,c2,c3}| <= 2
        # iff: (c0==c1) & (c0==c2) & (c0==c3)  [all same]
        #     OR: (c0==c1) & (c0==c2) & (c1==c3)  [c0=c1=c2, c3 different or same]
        #     ... this gets complex. Let me use a different approach.
        
        # Count distinct: for each AP, the number of distinct colors is:
        # 1 + (c1!=c0) + (c2 not in {c0,c1}) + (c3 not in {c0,c1,c2})
        # But that's sequential. Let's just check if <3 directly.
        
        # <3 distinct means the 4 values can be covered by 2 colors.
        # This is equivalent to: NOT (at least 3 distinct values)
        # At least 3 distinct: there exist 3 positions with pairwise different colors.
        
        # Direct check: violations = APs where the multiset has <=2 distinct values
        # = APs where (c0,c1,c2,c3) can be written with 2 values.
        
        # Efficient: count how many are "bad"
        # bad if: c0==c1==c2==c3 (mono)
        #    or: exactly 2 values appear
        # exactly 2 values: the set {c0,c1,c2,c3} has size exactly 2
        # This happens when: not all same, but some pair is same and the other pair is also same (possibly same value)
        
        # Let me just compute it straightforwardly with broadcasting
        # For each AP, check if all 4 are same, or if they use exactly 2 colors
        
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # Exactly 2 colors: not all same, but the 4 values fall into 2 groups
        # This is: (c0==c1 and c2==c3 and c0!=c2) OR
        #           (c0==c2 and c1==c3 and c0!=c1) OR
        #           (c0==c3 and c1==c2 and c0!=c1) OR
        #           (c0==c1 and c2!=c3 and c1!=c2 and c0!=c2 and c0!=c3) -- no, that's 3 colors
        # 
        # Actually, exactly 2 distinct values in 4 positions means the partition is 2+2, 3+1, or 1+3
        # 2+2: two pairs of equal values, the pairs having different values
        # 3+1: three of one value, one of another
        # 1+3: same as 3+1
        # 
        # Let me use a simpler criterion:
        # |{c0,c1,c2,c3}| <= 2 iff:
        # (c0==c1 and c0==c2 and c0==c3) OR
        # (c0==c1 and c0==c2 and c1==c3) OR  -- this is same as first if c3==c0
        # 
        # OK let me just use: for each AP, count distinct by checking all pairs
        # |set| <= 2 iff: for every pair (i,j), c_i == c_j OR c_i == c_k for some k != j
        # This is getting complicated. Let me use a lookup-free approach:
        
        # Number of distinct = 4 - (number of "collisions" that reduce count)
        # Actually, let me just directly check:
        # distinct < 3 iff: the 4 values have at most 2 unique values
        # iff: (c0==c1 or c0==c2 or c0==c3) AND (c1==c2 or c1==c3) AND (c2==c3)
        # No wait, that's not right either.
        
        # Let me think again. |{a,b,c,d}| <= 2.
        # Case 1: all same: a=b=c=d
        # Case 2: exactly 2: one value appears 3 times, another 1 time; or 2 and 2.
        # 
        # For 2+2: (a==b and c==d and a!=c) or (a==c and b==d and a!=b) or (a==d and b==c and a!=b)
        # For 3+1: (a==b and a==c and a!=d) or (a==b and a==d and a!=c) or (a==c and a==d and a!=b) or (b==c and b==d and b!=a)
        # 
        # This is getting unwieldy. Let me just use a simple approach:
        # For each AP, compute the sorted tuple and check.
        # But that's slow in numpy. 
        #
        # Alternative: use the fact that |{c0,c1,c2,c3}| <= 2 iff
        # (c0-c1)*(c0-c2)*(c0-c3) == 0 AND (c1-c2)*(c1-c3)*(c2-c3) == 0
        # No, that's not right.
        #
        # Let me just use a direct boolean formula:
        # |{a,b,c,d}| <= 2 iff:
        # (a==b) & (a==c) & (a==d)  -- all same
        # OR (a==b) & (c==d) & (a!=c)  -- 2+2 split as (ab)(cd)
        # OR (a==c) & (b==d) & (a!=b)  -- 2+2 split as (ac)(bd)
        # OR (a==d) & (b==c) & (a!=b)  -- 2+2 split as (ad)(bc)
        # OR (a==b) & (a==c) & (a!=d)  -- 3+1: abc same, d different
        # OR (a==b) & (a==d) & (a!=c)  -- 3+1: abd same, c different
        # OR (a==c) & (a==d) & (a!=b)  -- 3+1: acd same, b different
        # OR (b==c) & (b==d) & (b!=a)  -- 3+1: bcd same, a different
        
        two_plus_two = ((c0 == c1) & (c2 == c3) & (c0 != c2)) | \
                       ((c0 == c2) & (c1 == c3) & (c0 != c1)) | \
                       ((c0 == c3) & (c1 == c2) & (c0 != c1))
        
        three_plus_one = ((c0 == c1) & (c0 == c2) & (c0 != c3)) | \
                         ((c0 == c1) & (c0 == c3) & (c0 != c2)) | \
                         ((c0 == c2) & (c0 == c3) & (c0 != c1)) | \
                         ((c1 == c2) & (c1 == c3) & (c1 != c0))
        
        violations = np.sum(all_same | two_plus_two | three_plus_one)
        return int(violations)
    
    def validate_vec(coloring):
        return count_violations_vec(coloring) == 0
    
    def try_k_colors_class_sa(k, time_budget, rng):
        """Class-based simulated annealing: moves are swapping entire color classes."""
        start_t = time.time()
        best_violations = float('inf')
        best_coloring = None
        
        num_restarts = 4
        time_per_restart = time_budget / num_restarts
        
        for restart in range(num_restarts):
            if time.time() - start_t > time_budget - 0.5:
                break
            
            # Initialize
            if restart == 0:
                coloring = np.array([i % k for i in range(N)], dtype=np.int32)
            elif restart == 1:
                coloring = np.array([rng.randint(0, k-1) for _ in range(N)], dtype=np.int32)
            elif restart == 2:
                coloring = np.array([bin(i).count('1') % k for i in range(N)], dtype=np.int32)
            else:
                coloring = np.array([(i * 7 + 3) % k for i in range(N)], dtype=np.int32)
            
            current_violations = count_violations_vec(coloring)
            if current_violations == 0:
                return coloring.copy()
            if current_violations < best_violations:
                best_violations = current_violations
                best_coloring = coloring.copy()
            
            T = 3.0
            cooling = 0.9995
            steps = 0
            max_steps = 200000
            
            while current_violations > 0 and steps < max_steps and time.time() - start_t < time_budget - 0.5:
                steps += 1
                
                # Move: pick a position and change its color
                idx = rng.randint(0, N - 1)
                old_c = coloring[idx]
                new_c = rng.randint(0, k - 1)
                if old_c == new_c:
                    continue
                
                # Compute delta violations incrementally
                # Only APs containing idx are affected
                delta = 0
                for ap_id in pos_ap_map[idx]:
                    ap = ap_list[ap_id]
                    c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                    old_size = len({c0, c1, c2, c3})
                    
                    nc0 = new_c if ap[0] == idx else c0
                    nc1 = new_c if ap[1] == idx else c1
                    nc2 = new_c if ap[2] == idx else c2
                    nc3 = new_c if ap[3] == idx else c3
                    new_size = len({nc0, nc1, nc2, nc3})
                    
                    if old_size < 3 and new_size >= 3:
                        delta -= 1
                    elif old_size >= 3 and new_size < 3:
                        delta += 1
                
                coloring[idx] = new_c
                current_violations += delta
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring.copy()
                
                if current_violations > 0:
                    # Maybe revert
                    if delta > 0:
                        if T > 0.001:
                            accept = rng.random() < np.exp(-delta / T)
                        else:
                            accept = False
                        if not accept:
                            coloring[idx] = old_c
                            current_violations -= delta
                    elif delta == 0:
                        if rng.random() > 0.5:
                            coloring[idx] = old_c
                            current_violations -= delta
                
                T *= cooling
            
            if best_violations == 0:
                return best_coloring.copy()
        
        if best_violations == 0:
            return best_coloring.copy()
        return None
    
    def try_k_colors_pair_swap(k, time_budget, rng):
        """Try to find valid coloring using pair-swap and triple-swap moves."""
        start_t = time.time()
        best_violations = float('inf')
        best_coloring = None
        
        # Initialize with structured pattern
        coloring = np.array([i % k for i in range(N)], dtype=np.int32)
        current_violations = count_violations_vec(coloring)
        if current_violations == 0:
            return coloring.copy()
        best_violations = current_violations
        best_coloring = coloring.copy()
        
        T = 2.0
        cooling = 0.9993
        steps = 0
        max_steps = 300000
        
        while current_violations > 0 and steps < max_steps and time.time() - start_t < time_budget - 0.5:
            steps += 1
            
            # Random move type: 70% single position, 20% pair swap, 10% triple
            move_type = rng.random()
            
            if move_type < 0.7:
                # Single position recolor
                idx = rng.randint(0, N - 1)
                old_c = coloring[idx]
                new_c = rng.randint(0, k - 1)
                if old_c == new_c:
                    continue
                
                delta = 0
                for ap_id in pos_ap_map[idx]:
                    ap = ap_list[ap_id]
                    c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                    old_size = len({c0, c1, c2, c3})
                    nc0 = new_c if ap[0] == idx else c0
                    nc1 = new_c if ap[1] == idx else c1
                    nc2 = new_c if ap[2] == idx else c2
                    nc3 = new_c if ap[3] == idx else c3
                    new_size = len({nc0, nc1, nc2, nc3})
                    if old_size < 3 and new_size >= 3:
                        delta -= 1
                    elif old_size >= 3 and new_size < 3:
                        delta += 1
                
                coloring[idx] = new_c
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring.copy()
                
                if current_violations > 0 and delta > 0:
                    if T > 0.001:
                        if rng.random() < np.exp(-delta / T):
                            pass  # accept
                        else:
                            coloring[idx] = old_c
                            current_violations -= delta
                    else:
                        coloring[idx] = old_c
                        current_violations -= delta
                
            elif move_type < 0.9:
                # Pair swap: swap colors of two random positions
                i1, i2 = rng.sample(range(N), 2)
                c1, c2 = coloring[i1], coloring[i2]
                if c1 == c2:
                    continue
                
                delta = 0
                # Check all APs containing i1 or i2
                affected_aps = set(pos_ap_map[i1]) | set(pos_ap_map[i2])
                for ap_id in affected_aps:
                    ap = ap_list[ap_id]
                    c0, c1v, c2v, c3v = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                    old_size = len({c0, c1v, c2v, c3v})
                    
                    nc0 = c2 if ap[0] == i1 else (c1 if ap[0] == i2 else c0)
                    nc1 = c2 if ap[1] == i1 else (c1 if ap[1] == i2 else c1v)
                    nc2 = c2 if ap[2] == i1 else (c1 if ap[2] == i2 else c2v)
                    nc3 = c2 if ap[3] == i1 else (c1 if ap[3] == i2 else c3v)
                    new_size = len({nc0, nc1, nc2, nc3})
                    
                    if old_size < 3 and new_size >= 3:
                        delta -= 1
                    elif old_size >= 3 and new_size < 3:
                        delta += 1
                
                coloring[i1] = c2
                coloring[i2] = c1
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring.copy()
                
                if current_violations > 0 and delta > 0:
                    if T > 0.001:
                        if rng.random() < np.exp(-delta / T):
                            pass
                        else:
                            coloring[i1] = c1
                            coloring[i2] = c2
                            current_violations -= delta
                    else:
                        coloring[i1] = c1
                        coloring[i2] = c2
                        current_violations -= delta
            
            else:
                # Triple recolor: change 3 random positions
                idxs = rng.sample(range(N), 3)
                old_colors = [coloring[i] for i in idxs]
                new_colors = [rng.randint(0, k-1) for _ in idxs]
                
                delta = 0
                affected_aps = set()
                for i in idxs:
                    affected_aps.update(pos_ap_map[i])
                
                for ap_id in affected_aps:
                    ap = ap_list[ap_id]
                    c0, c1v, c2v, c3v = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                    old_size = len({c0, c1v, c2v, c3v})
                    
                    nc0 = c1v; nc1 = c1v; nc2 = c2v; nc3 = c3v
                    for j, idx in enumerate(idxs):
                        if ap[0] == idx: nc0 = new_colors[j]
                        if ap[1] == idx: nc1 = new_colors[j]
                        if ap[2] == idx: nc2 = new_colors[j]
                        if ap[3] == idx: nc3 = new_colors[j]
                    new_size = len({nc0, nc1, nc2, nc3})
                    
                    if old_size < 3 and new_size >= 3:
                        delta -= 1
                    elif old_size >= 3 and new_size < 3:
                        delta += 1
                
                for j, idx in enumerate(idxs):
                    coloring[idx] = new_colors[j]
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring.copy()
                
                if current_violations > 0 and delta > 0:
                    if T > 0.001:
                        if rng.random() < np.exp(-delta / T):
                            pass
                        else:
                            for j, idx in enumerate(idxs):
                                coloring[idx] = old_colors[j]
                            current_violations -= delta
                    else:
                        for j, idx in enumerate(idxs):
                            coloring[idx] = old_colors[j]
                        current_violations -= delta
            
            T *= cooling
        
        if best_violations == 0:
            return best_coloring.copy()
        return None
    
    def reduce_colors(coloring, max_time, rng):
        """Try to reduce number of distinct colors."""
        start_t = time.time()
        coloring = coloring.copy()
        n_colors = len(set(coloring.tolist()))
        
        # Get color frequencies
        def try_eliminate_color(coloring, target, max_time, rng):
            """Try to eliminate a specific color by recoloring its positions."""
            positions = [i for i in range(N) if coloring[i] == target]
            if not positions:
                return coloring, False
            
            other_colors = [c for c in set(coloring.tolist()) if c != target]
            
            # Try greedy: recolor each position to a valid other color
            success = True
            for p in positions:
                found = False
                for nc in other_colors:
                    coloring[p] = nc
                    if count_violations_vec(coloring) == 0:
                        found = True
                        break
                if not found:
                    # Try random search for this position
                    for _ in range(50):
                        nc = rng.choice(other_colors)
                        coloring[p] = nc
                        if count_violations_vec(coloring) == 0:
                            found = True
                            break
                if not found:
                    success = False
                    break
            
            if success:
                return coloring, True
            return coloring, False
        
        # Try eliminating rarest colors first
        for _ in range(10):
            if time.time() - start_t > max_time:
                break
            if n_colors <= 3:
                break
            
            color_counts = {}
            for c in coloring:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            # Sort by frequency (rarest first)
            sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x])
            
            eliminated = False
            for target in sorted_colors:
                if color_counts[target] >= n_colors - 3:
                    break  # Can't eliminate enough
                if time.time() - start_t > max_time:
                    break
                
                coloring, ok = try_eliminate_color(coloring, target, max_time, rng)
                if ok:
                    n_colors = len(set(coloring.tolist()))
                    eliminated = True
                    print(f"Eliminated color, now {n_colors} colors", file=sys.stderr)
                    break
            
            if not eliminated:
                break
        
        return coloring
    
    # Main strategy: try k=3, 4, 5 with different methods
    rng = random.Random(12345)
    best_colors = None
    best_count = N + 1
    
    # Time allocation
    time_alloc = {
        3: 25.0,
        4: 15.0,
        5: 10.0,
        6: 8.0
    }
    
    for k in [3, 4, 5, 6]:
        remaining = deadline - time.time() - 15
        if remaining < 3:
            break
        
        t_budget = min(time_alloc[k], remaining)
        print(f"Trying k={k} with budget {t_budget:.1f}s", file=sys.stderr)
        
        # Try multiple methods
        result = None
        
        # Method 1: Class-based SA
        if result is None and t_budget > 3:
            result = try_k_colors_class_sa(k, t_budget * 0.4, rng)
        
        # Method 2: Pair-swap SA
        if result is None and t_budget > 3:
            result = try_k_colors_pair_swap(k, t_budget * 0.6, rng)
        
        if result is not None and validate_vec(result):
            best_colors = result
            best_count = k
            print(f"Found valid coloring with k={k}", file=sys.stderr)
            break
    
    # Fallback: greedy construction with many colors
    if best_colors is None:
        print("SA failed, using greedy fallback", file=sys.stderr)
        coloring = np.zeros(N, dtype=np.int32)
        for i in range(N):
            # Find a color that doesn't create violations
            for c in range(N):
                coloring[i] = c
                if count_violations_vec(coloring) == 0:
                    break
            else:
                coloring[i] = i  # Use unique color as last resort
        best_colors = coloring
        best_count = len(set(coloring.tolist()))
    
    # Try to reduce colors
    if best_count > 3 and time.time() < deadline - 10:
        reduce_time = min(15.0, deadline - time.time() - 8)
        if reduce_time > 2:
            reduced = reduce_colors(best_colors, reduce_time, rng)
            new_count = len(set(reduced.tolist()))
            if new_count < best_count:
                best_colors = reduced
                best_count = new_count
                print(f"Reduced to {best_count} colors", file=sys.stderr)
    
    # Relabel to consecutive
    unique = sorted(set(best_colors.tolist()))
    mapping = {c: i for i, c in enumerate(unique)}
    result = np.array([mapping[c] for c in best_colors], dtype=np.int32)
    
    # Final validation
    if not validate_vec(result):
        print("WARNING: final validation failed", file=sys.stderr)
        # Emergency: use 4-color periodic
        result = np.array([i % 4 for i in range(N)], dtype=np.int32)
    
    return result.tolist()
# EVOLVE-BLOCK-END
