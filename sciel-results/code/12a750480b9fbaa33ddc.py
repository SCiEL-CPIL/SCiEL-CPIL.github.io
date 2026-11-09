import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (limit is 150s, but let's be safe)

    # 1. Precomputation
    # Edge index mapping: (i, j) -> idx
    # i, j in 0..15, i < j
    # Total edges: 16*15/2 = 120
    def get_idx(i, j):
        return i * 15 - i * (i - 1) // 2 + (j - i - 1)
        
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))
            
    edge_uv = [None] * 120
    for idx, (i, j) in enumerate(edges):
        edge_uv[idx] = (i, j)

    edge_idx_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx_map[(i, j)] = idx
        edge_idx_map[(j, i)] = idx

    # Precompute K4s
    # A K4 is a set of 4 vertices. It has 6 edges.
    # We need every K4 to have at least 5 distinct colors.
    # This is equivalent to: in every K4, at most 2 edges can share the same color? No.
    # If a K4 has 6 edges and 5 distinct colors, the color counts in the K4 must be like:
    # 1,1,1,1,2 (sum 6) -> 5 distinct colors.
    # Or 1,1,1,2,1 (same)
    # Can it be 1,1,2,2? That's 4 distinct colors. Invalid.
    # So in any K4, no color can appear 3 or more times?
    # If a color appears 3 times, remaining 3 edges must have 3 distinct colors (to reach 5 total? No, 3 edges + 1 color = 4 distinct colors max if all distinct. Wait.
    # If colors are c1,c1,c1,c2,c3,c4 -> distinct: {c1,c2,c3,c4} = 4. Invalid.
    # So max frequency of any color in a K4 is 2.
    # And if a color appears 2 times, the other 4 edges must have 4 distinct colors different from the first one?
    # c1,c1,c2,c3,c4,c5 -> distinct: {c1,c2,c3,c4,c5} = 5. Valid.
    # c1,c1,c2,c3,c4,c2 -> distinct: {c1,c2,c3,c4} = 4. Invalid.
    # So if a color appears twice, the other 4 edges must be all distinct and different from the first color.
    # Actually, simpler constraint:
    # For any K4, let the multiset of colors be M.
    # |set(M)| >= 5.
    # Since |M|=6, this means:
    # - No color appears >= 3 times.
    # - If a color appears exactly 2 times, the other 4 edges must have 4 distinct colors, none of which is the repeated color.
    #   (If one of the other 4 was the repeated color, it would appear 3 times total, which is already ruled out.
    #    If two of the other 4 were the same color, say c2, then we have c1,c1,c2,c2,... distinct count <= 4.
    #    So yes, if a color repeats, all other 4 must be distinct and different from it.)
    
    # Let's precompute the 6 edge indices for each K4.
    k4s_all = []
    edge_k4s = [[] for _ in range(120)]
    
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        e_indices = [
            edge_idx_map[(a, b)],
            edge_idx_map[(a, c)],
            edge_idx_map[(a, d)],
            edge_idx_map[(b, c)],
            edge_idx_map[(b, d)],
            edge_idx_map[(c, d)]
        ]
        k4_idx = len(k4s_all)
        k4s_all.append(e_indices)
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)

    # 2. Initial Solution
    # Use the (i+j) % 17 construction.
    # Is it valid?
    # In K16, vertices 0..15.
    # Color of (i,j) is (i+j) % 17.
    # Consider a K4 {a,b,c,d}.
    # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d).
    # Colors: a+b, a+c, a+d, b+c, b+d, c+d (mod 17).
    # We need 5 distinct colors.
    # Let's check if this construction always works.
    # Actually, the prompt says "Minimize the number of colours used. A valid colouring certifies f(16) <= colours used."
    # And "The general bounds give 13 <= f(16) <= 15".
    # The seed program uses 17 colors and claims it's valid.
    # Wait, if f(16) <= 15, then 17 is not minimal.
    # The seed program returns 17 colors. Score 1.0.
    # The goal is to find a valid coloring with <= 15 colors.
    # Actually, the evaluator measures "raw value (lower is better)".
    # So if I return a valid coloring with 15 colors, the value is 15.
    # If I return a valid coloring with 17 colors, the value is 17.
    # Score = seed/value. Seed value is 17? Or is seed value 1.0?
    # "1.0 equals the seed program".
    # If seed program returns 17 colors, and score is 1.0, then "value" is likely the number of colors.
    # So I want to minimize the number of distinct colors used in the returned array.
    
    # Let's try to find a valid coloring with K=15, 14, 13 colors.
    # Strategy:
    # 1. Start with a valid coloring using K colors (e.g., K=17).
    # 2. Try to reduce K to 16, 15, etc.
    # 3. Use local search to fix violations when reducing colors.

    # Better Strategy:
    # Start with K=15 colors.
    # Initialize randomly with K=15 colors.
    # Use Simulated Annealing to minimize the number of invalid K4s.
    # Once we have 0 invalid K4s, we have a valid coloring with <= 15 colors.
    # Then try to compress to 14, 13.

    # Let's implement a fast validity check.
    # For a given coloring `colors` (list of 120 ints), count invalid K4s.
    # An invalid K4 is one with < 5 distinct colors.
    
    def count_invalid_k4s(colors):
        invalid_count = 0
        for e_indices in k4s_all:
            # e_indices is a list of 6 edge indices
            c0 = colors[e_indices[0]]
            c1 = colors[e_indices[1]]
            c2 = colors[e_indices[2]]
            c3 = colors[e_indices[3]]
            c4 = colors[e_indices[4]]
            c5 = colors[e_indices[5]]
            
            # Check distinct count
            # Fast path: if all 6 are distinct, valid.
            # If 5 distinct, valid.
            # If < 5, invalid.
            
            # We can just use a set. 1820 K4s * 6 edges = ~11k operations.
            # With Python sets, this might be slow in the inner loop of SA.
            # Let's optimize.
            
            # Alternative: 
            # Sort the 6 colors? No.
            # Just count distinct.
            # Since 6 is small, manual comparison might be faster than set?
            # Or use a small array for counts?
            
            # Let's try the set approach first, it's clean.
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                invalid_count += 1
                
        return invalid_count

    # To speed up SA, we need to evaluate the change in invalid_count when changing one edge.
    # Changing edge e affects only K4s containing e.
    # There are 91 such K4s.
    # For each affected K4, we check if it was valid before, and if it is valid now.
    # delta = (invalid_before - valid_before) - (invalid_after - valid_after)
    # Actually, let's just maintain a global count `total_invalid`.
    # When we change edge e from c_old to c_new:
    # For each K4 in edge_k4s[e]:
    #   check if K4 was valid before.
    #   check if K4 is valid now.
    #   update total_invalid.
    
    # We need a fast function to check if a specific K4 is valid given the current colors.
    def is_k4_valid(k4_idx, colors):
        e_indices = k4s_all[k4_idx]
        c0 = colors[e_indices[0]]
        c1 = colors[e_indices[1]]
        c2 = colors[e_indices[2]]
        c3 = colors[e_indices[3]]
        c4 = colors[e_indices[4]]
        c5 = colors[e_indices[5]]
        return len({c0, c1, c2, c3, c4, c5}) >= 5

    # Let's try to find a valid solution with K=15.
    # If we find one, return it.
    # If not, try K=16.
    # If not, return the best valid one found (or the initial 17-color one if no valid one found with fewer).
    # Actually, the initial 17-color one is guaranteed valid?
    # Let's verify the (i+j)%17 construction.
    # If it's valid, we can always fall back to it.
    
    def generate_initial_coloring(K):
        # Random initialization
        colors = [random.randint(0, K - 1) for _ in range(120)]
        return colors

    def try_find_valid(K, time_limit):
        # Simulated Annealing
        colors = generate_initial_coloring(K)
        
        # Calculate initial invalid count
        total_invalid = count_invalid_k4s(colors)
        
        best_colors = colors[:]
        best_invalid = total_invalid
        
        if best_invalid == 0:
            return best_colors
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        # Precompute affected K4s for each edge for faster access?
        # edge_k4s is already there.
        
        while time.time() < time_limit:
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            new_color = random.randint(0, K - 1)
            
            if new_color == old_color:
                continue
            
            # Calculate delta in invalid count
            delta_invalid = 0
            
            # For each K4 containing e_idx
            for k4_idx in edge_k4s[e_idx]:
                # Check validity before
                valid_before = is_k4_valid(k4_idx, colors)
                
                # Apply change temporarily? No, we can just check with new_color.
                # But is_k4_valid uses `colors`.
                # We need to check if K4 is valid with e_idx having new_color.
                # Let's write a helper that takes the edge index and new color.
                
                # Actually, modifying the array and reverting is safer and probably fast enough for 91 K4s.
                # But we can optimize by just checking the 6 colors.
                
                e_indices = k4s_all[k4_idx]
                
                # Get the 6 colors, replacing the one at e_idx with new_color
                # This is a bit messy. Let's just modify the array.
                
                colors[e_idx] = new_color
                valid_after = is_k4_valid(k4_idx, colors)
                
                if valid_before and not valid_after:
                    delta_invalid += 1
                elif not valid_before and valid_after:
                    delta_invalid -= 1
            
            # Revert for now if we don't accept?
            # Or keep applied if we accept.
            
            new_total_invalid = total_invalid + delta_invalid
            
            # Acceptance criteria
            accept = False
            if new_total_invalid < total_invalid:
                accept = True
            elif new_total_invalid == total_invalid:
                # Accept with some probability to escape plateaus
                accept = random.random() < 0.5
            else:
                # Accept worse move with probability exp(-delta / T)
                delta = new_total_invalid - total_invalid
                if random.random() < pow(2.71828, -delta / T):
                    accept = True
            
            if accept:
                # The change is already applied in the loop above (colors[e_idx] = new_color)
                # Wait, I set colors[e_idx] = new_color inside the loop.
                # So it is applied.
                total_invalid = new_total_invalid
                
                if total_invalid < best_invalid:
                    best_invalid = total_invalid
                    best_colors = colors[:]
                    
                    if best_invalid == 0:
                        return best_colors
            else:
                # Revert
                colors[e_idx] = old_color
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_colors

    # Try K=15
    # Give it 30 seconds
    t_start = time.time()
    colors_15 = try_find_valid(15, t_start + 30)
    
    if count_invalid_k4s(colors_15) == 0:
        # Compress colors to 0..K-1
        unique_colors = sorted(set(colors_15))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        return [mapping[c] for c in colors_15]
        
    # Try K=16
    t_start = time.time()
    colors_16 = try_find_valid(16, t_start + 30)
    
    if count_invalid_k4s(colors_16) == 0:
        unique_colors = sorted(set(colors_16))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        return [mapping[c] for c in colors_16]
        
    # Fallback to K=17 (Seed construction)
    # Verify it's valid
    colors_17 = [0] * 120
    for idx, (i, j) in enumerate(edges):
        colors_17[idx] = (i + j) % 17
        
    if count_invalid_k4s(colors_17) == 0:
        unique_colors = sorted(set(colors_17))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        return [mapping[c] for c in colors_17]
        
    # If even 17 fails (it shouldn't), return random
    return colors_17

# EVOLVE-BLOCK-END
