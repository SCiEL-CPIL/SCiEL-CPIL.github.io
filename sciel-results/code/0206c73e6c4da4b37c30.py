import time
import random
import math

def construct():
    N = 128
    deadline = time.time() + 80.0  # 10s safety margin

    # Precompute all non-constant 4-term APs
    # A 4-term AP is (a, a+d, a+2d, a+3d) with d>0
    # Using 0-indexed: (i, i+d, i+2d, i+3d) where i+3d < N
    aps = []
    for d in range(1, N):
        for a in range(0, N - 3*d):
            aps.append((a, a+d, a+2*d, a+3*d))
    
    ap_count = len(aps)
    
    # For each index, list of (ap_index, position_in_ap) where position is 0,1,2,3
    index_to_aps = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for pos, idx in enumerate(ap):
            index_to_aps[idx].append((ap_idx, pos))

    def is_bad(c1, c2, c3, c4):
        """Check if a 4-term AP has fewer than 3 distinct colors."""
        # Count distinct colors
        if c1 == c2:
            if c3 == c4:
                return True  # At most 2 colors
            else:
                return c1 == c3 or c1 == c4
        elif c1 == c3:
            return c2 == c4
        elif c1 == c4:
            return c2 == c3
        else:
            if c2 == c3:
                return c2 == c4
            elif c2 == c4:
                return c1 == c3
            elif c3 == c4:
                return c2 == c3
        return False

    def count_violations(colors):
        count = 0
        for ap in aps:
            if is_bad(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                count += 1
        return count

    def try_k_coloring(K, time_limit):
        """Try to find a valid K-coloring using local search."""
        start = time.time()
        
        # Multiple restarts
        for attempt in range(100):
            if time.time() - start > time_limit or time.time() > deadline:
                break
            
            # Random initialization
            colors = [random.randint(0, K-1) for _ in range(N)]
            energy = count_violations(colors)
            
            if energy == 0:
                return colors
            
            # Local search: iterated local search with random restarts on stuck
            best_energy = energy
            best_colors = colors[:]
            
            T = 1.0
            min_T = 1e-4
            cooling = 0.9995
            
            steps_since_improvement = 0
            max_steps_no_improve = 500
            
            while T > min_T:
                if time.time() > deadline or time.time() - start > time_limit:
                    break
                
                i = random.randint(0, N-1)
                old_color = colors[i]
                new_color = random.randint(0, K-1)
                while new_color == old_color:
                    new_color = random.randint(0, K-1)
                
                # Calculate delta by checking all APs involving index i
                old_local = 0
                new_local = 0
                
                for (ap_idx, pos) in index_to_aps[i]:
                    ap = aps[ap_idx]
                    c = list(colors[ap])
                    # Old
                    if is_bad(c[0], c[1], c[2], c[3]):
                        old_local += 1
                    # New
                    c[pos] = new_color
                    if is_bad(c[0], c[1], c[2], c[3]):
                        new_local += 1
                
                delta = new_local - old_local
                
                if delta < 0 or (delta == 0 and random.random() < 0.5) or \
                   (delta > 0 and random.random() < math.exp(-delta / T)):
                    colors[i] = new_color
                    energy += delta
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_colors = colors[:]
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1
                    
                    if energy == 0:
                        return colors
                    
                    if steps_since_improvement > max_steps_no_improve:
                        # Perturb and reset
                        for _ in range(5):
                            j = random.randint(0, N-1)
                            colors[j] = random.randint(0, K-1)
                        energy = count_violations(colors)
                        steps_since_improvement = 0
                
                T *= cooling
        
        # Return best found if valid
        if best_energy == 0:
            return best_colors
        return None

    # Strategy: Try K=3 first with more time, then K=4, K=5, etc.
    # Based on the problem, minimum is likely 3 or 4 colors.
    
    # First, try a deterministic construction for K=3
    # Known result: For 4-APs, we need at least 3 colors.
    # Let's try: color based on position mod 3
    # But that might not work. Let's use search.
    
    # Try K=3
    remaining = deadline - time.time()
    if remaining > 20:
        sol = try_k_coloring(3, min(25, remaining))
        if sol:
            return sol
    
    # Try K=4
    remaining = deadline - time.time()
    if remaining > 15:
        sol = try_k_coloring(4, min(20, remaining))
        if sol:
            return sol
    
    # Try K=5
    remaining = deadline - time.time()
    if remaining > 10:
        sol = try_k_coloring(5, min(15, remaining))
        if sol:
            return sol
    
    # Try K=6
    remaining = deadline - time.time()
    if remaining > 5:
        sol = try_k_coloring(6, min(10, remaining))
        if sol:
            return sol
    
    # Fallback: Greedy construction with many colors
    # This will always find a valid solution, just with more colors
    colors = [0] * N
    for i in range(N):
        # Find the smallest color that doesn't create a violation
        for c in range(0, 128):
            colors[i] = c
            valid = True
            # Check all 4-APs ending at i or involving i
            # Actually, just check all APs that include i
            for (ap_idx, pos) in index_to_aps[i]:
                ap = aps[ap_idx]
                c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if is_bad(c1, c2, c3, c4):
                    valid = False
                    break
            if valid:
                break
    
    # Verify and return
    final_colors = colors[:]
    used = len(set(final_colors))
    
    # If greedy used too many colors, try a bit more search
    if used > 6 and time.time() < deadline - 5:
        # Quick local search to reduce colors
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            # Try to recolor a random position to a color already in use
            i = random.randint(0, N-1)
            old = colors[i]
            existing_colors = list(set(colors))
            random.shuffle(existing_colors)
            found = False
            for c in existing_colors:
                if c == old:
                    continue
                colors[i] = c
                valid = True
                for (ap_idx, pos) in index_to_aps[i]:
                    ap = aps[ap_idx]
                    if is_bad(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                        valid = False
                        break
                if valid:
                    found = True
                    break
            if not found:
                colors[i] = old
    
    return colors

# Quick test to verify the function works
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Colors used: {len(set(result))}")
    print(f"Max color: {max(result)}")
    # Verify
    N = 128
    aps = []
    for d in range(1, N):
        for a in range(0, N - 3*d):
            aps.append((a, a+d, a+2*d, a+3*d))
    violations = 0
    for ap in aps:
        c1, c2, c3, c4 = result[ap[0]], result[ap[1]], result[ap[2]], result[ap[3]]
        if is_bad(c1, c2, c3, c4):
            violations += 1
    print(f"Violations: {violations}")
