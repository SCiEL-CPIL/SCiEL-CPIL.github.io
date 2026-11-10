import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Precompute index_to_aps for delta calculation
    index_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i in ap:
            index_to_aps[i].append(idx)

    def score_full(colors):
        """Calculate exact number of monochromatic 4-APs."""
        count = 0
        for ap in aps:
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            if c0 == c1 == c2 == c3:
                count += 1
        return count

    # 2. Structured 2D Initialization (16x16 Grid)
    # We treat the sequence 1..256 as a 16x16 matrix.
    # Index i corresponds to matrix[row][col] where i = row * 16 + col (0-based).
    # We construct rows to minimize intra-row and intra-column 4-APs.
    
    # A known good pattern for 2-coloring to avoid 4-APs is based on Thue-Morse or similar,
    # but for 2D, we can try to balance rows and columns.
    
    # Strategy:
    # 1. Generate a base pattern that is low-density of 4-APs in 1D.
    # 2. Arrange it in 2D such that column APs are also minimized.
    
    # Let's use a simple heuristic: 
    # Start with a random 16x16 grid.
    # Then, for each row, try to flip the row (0->1, 1->0) or permute? 
    # No, permutation breaks the index mapping.
    
    # Better:
    # Use a known 1D sequence with few 4-APs, e.g., the "van der Corput" sequence or a specific binary string.
    # However, the simplest robust initialization that is different from random is to use a 
    # "checkerboard" like pattern or a "block" pattern, then refine.
    
    # Let's try a specific 2D construction:
    # Let colors[i] = (i // 16) ^ (i % 16) % 2 ? No.
    # Let's try: colors[i] = (i // 16) % 2 ^ (i % 16) % 2
    # This creates a checkerboard. 
    # In a checkerboard, a 4-AP a, a+d, a+2d, a+3d:
    # If d is odd, the colors alternate 0,1,0,1 or 1,0,1,0. No mono.
    # If d is even, the colors are 0,0,0,0 or 1,1,1,1. All mono.
    # So checkerboard is bad for even d.
    
    # Let's try: colors[i] = (i // 4) % 2. (Blocks of 4)
    # 0000 1111 0000 1111...
    # 4-AP with d=1: 0,0,0,0 (mono) or 1,1,1,1 (mono). Bad.
    
    # Let's use a Thue-Morse sequence. It is 4-AP free? 
    # Thue-Morse avoids squares, but not necessarily 4-APs. 
    # Actually, Thue-Morse is known to be cube-free, but does it avoid 4-APs?
    # A 4-AP in Thue-Morse: t(n), t(n+d), t(n+2d), t(n+3d).
    # It is known that Thue-Morse contains no 4-term arithmetic progression of the same value? 
    # Wait, the Thue-Morse sequence is 2-coloring of integers. 
    # It is a well-known fact that the Thue-Morse sequence is 4-AP free? 
    # Let's check: 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0
    # Indices: 0..15
    # d=1: 0,1,1,0 (no), 1,1,0,1 (no), 1,0,1,0 (no), 0,1,0,0 (no), 1,0,0,1 (no), 0,0,1,1 (no), 0,1,1,0 (no), 1,1,0,1 (no), 1,0,0,1 (no), 0,0,1,0 (no), 0,1,0,1 (no), 1,0,1,1 (no), 0,1,1,0 (no)
    # d=2: 0,1,1,0 (no), 1,0,0,1 (no), 1,1,0,0 (no), 0,0,1,1 (no), 1,1,0,0 (no), 0,0,1,1 (no), 1,0,0,1 (no), 0,1,1,0 (no), 1,0,0,1 (no), 0,1,1,0 (no), 1,0,0,1 (no), 0,1,1,0 (no)
    # d=3: 0,1,0,1 (no), 1,0,1,0 (no), 1,1,0,0 (no), 0,0,1,1 (no), 1,1,0,0 (no), 0,0,1,1 (no), 1,0,0,1 (no), 0,1,1,0 (no), 1,0,0,1 (no), 0,1,1,0 (no)
    # d=4: 0,1,1,0 (no), 1,0,0,1 (no), 1,1,0,0 (no), 0,0,1,1 (no), 1,1,0,0 (no), 0,0,1,1 (no), 1,0,0,1 (no), 0,1,1,0 (no)
    # d=5: 0,0,1,1 (no), 1,1,0,0 (no), 1,0,0,1 (no), 0,0,1,1 (no), 1,1,0,0 (no), 0,0,1,1 (no), 1,0,0,1 (no), 0,1,1,0 (no)
    # d=6: 0,0,0,0 -> 0,0,0,0 is mono! Indices 0,6,12,18? 
    # t(0)=0, t(6)=0, t(12)=0, t(18)=? 
    # t(18) = t(16+2) = t(16) xor t(2) = 0 xor 1 = 1. 
    # So 0,0,0,1 is not mono.
    # It is a known result that the Thue-Morse sequence is 4-AP free? 
    # Actually, the Thue-Morse sequence is known to be "cube-free" but it DOES contain 4-APs? 
    # Let's check online knowledge: "The Thue-Morse sequence is 4-AP free" is a common claim but might be false for 4-APs. 
    # Wait, the sequence 0,1,1,0,1,0,0,1... 
    # Let's just generate it and score it. If it's good, use it. If not, use random.
    
    def generate_thue_morse(n):
        tm = [0]
        for i in range(n - 1):
            tm.append(1 - tm[-1]) # This is wrong, this is just alternating.
        # Correct Thue-Morse:
        tm = [0]
        for i in range(1, n):
            # t(n) = t(n//2) + (n%2) mod 2
            # Or simpler: t(n) = popcount(n) % 2
            pass
        # Use popcount
        return [bin(i).count('1') % 2 for i in range(n)]

    # Let's try multiple initialization strategies and pick the best after a quick local search.
    
    def quick_local_search(colors, max_iters=100):
        colors = colors.copy()
        is_mono = np.zeros(total_aps, dtype=bool)
        for idx, ap in enumerate(aps):
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                is_mono[idx] = True
        current_score = np.sum(is_mono)
        
        for _ in range(max_iters):
            improved = False
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                new_val = 1 - colors[i]
                delta = 0
                for ap_idx in index_to_aps[i]:
                    ap = aps[ap_idx]
                    others = [j for j in ap if j != i]
                    was_mono = is_mono[ap_idx]
                    is_new_mono = (new_val == colors[others[0]] == colors[others[1]] == colors[others[2]])
                    if was_mono and not is_new_mono:
                        delta -= 1
                    elif not was_mono and is_new_mono:
                        delta += 1
                if delta < 0:
                    colors[i] = new_val
                    for ap_idx in index_to_aps[i]:
                        ap = aps[ap_idx]
                        others = [j for j in ap if j != i]
                        new_is_mono = (colors[i] == colors[others[0]] == colors[others[1]] == colors[others[2]])
                        if new_is_mono != is_mono[ap_idx]:
                            is_mono[ap_idx] = new_is_mono
                            current_score += 1 if new_is_mono else -1
                    improved = True
            if not improved:
                break
        return colors, current_score

    # Candidate 1: Thue-Morse
    tm_colors = np.array(generate_thue_morse(N), dtype=np.int8)
    # Candidate 2: Random
    rand_colors = np.random.randint(0, 2, size=N, dtype=np.int8)
    # Candidate 3: 2D Checkerboard with offset
    grid = np.zeros((16, 16), dtype=np.int8)
    for r in range(16):
        for c in range(16):
            grid[r, c] = (r + c) % 2
    check_colors = grid.flatten().astype(np.int8)
    
    # Evaluate and refine candidates
    best_init_colors, best_init_score = None, float('inf')
    
    for cand in [tm_colors, rand_colors, check_colors]:
        refined_c, s = quick_local_search(cand, max_iters=20)
        if s < best_init_score:
            best_init_score = s
            best_init_colors = refined_c

    colors = best_init_colors
    current_score = best_init_score
    
    # Re-initialize is_mono for the chosen best
    is_mono = np.zeros(total_aps, dtype=bool)
    for idx, ap in enumerate(aps):
        if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
            is_mono[idx] = True
            
    best_score = current_score
    best_colors = colors.copy()

    # 3. Multi-bit Simulated Annealing
    # The directive suggests 2D grid, but we are in 1D index space. 
    # However, we can define moves that flip 2 or 3 bits to escape local minima.
    
    T = 1.0
    alpha = 0.999
    
    # Precompute pairs of indices for 2-bit flips? 
    # Just pick 2 random indices.
    
    def apply_flip(indices, new_vals):
        nonlocal current_score
        # Check if any index is same
        # Calculate delta
        delta = 0
        affected_aps = set()
        for i, nv in zip(indices, new_vals):
            if colors[i] != nv:
                for ap_idx in index_to_aps[i]:
                    affected_aps.add(ap_idx)
        
        if not affected_aps:
            return 0, False
            
        # For each affected AP, calculate change
        # We need to be careful: if an AP contains multiple flipped indices, 
        # we must calculate the final state vs initial state.
        
        initial_states = {}
        final_states = {}
        
        for ap_idx in affected_aps:
            ap = aps[ap_idx]
            # Get initial colors
            init_c = [colors[j] for j in ap]
            # Get final colors
            fin_c = init_c.copy()
            for i, nv in zip(indices, new_vals):
                if i in ap:
                    fin_c[ap.index(i)] = nv
            
            init_mono = (init_c[0] == init_c[1] == init_c[2] == init_c[3])
            fin_mono = (fin_c[0] == fin_c[1] == fin_c[2] == fin_c[3])
            
            if init_mono and not fin_mono:
                delta -= 1
            elif not init_mono and fin_mono:
                delta += 1
        
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            # Apply
            for i, nv in zip(indices, new_vals):
                colors[i] = nv
            # Update is_mono
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                new_is_mono = (colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]])
                if new_is_mono != is_mono[ap_idx]:
                    is_mono[ap_idx] = new_is_mono
                    current_score += 1 if new_is_mono else -1
            return delta, True
        return delta, False

    while time.time() < deadline:
        # Decide move type: 1-bit (60%), 2-bit (30%), 3-bit (10%)
        r = random.random()
        if r < 0.6:
            k = 1
        elif r < 0.9:
            k = 2
        else:
            k = 3
            
        indices = random.sample(range(N), k)
        new_vals = [1 - colors[i] for i in indices]
        
        d, accepted = apply_flip(indices, new_vals)
        
        if accepted:
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
                
        T *= alpha
        if T < 0.01:
            T = 1.0 # Restart

    return best_colors.tolist()

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Result length: {len(res)}")
    assert len(res) == 256
    assert all(c in [0, 1] for c in res)
    print("Valid")
