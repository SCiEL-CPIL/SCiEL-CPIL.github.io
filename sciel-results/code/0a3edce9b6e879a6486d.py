import time
import random

# Precompute the K4s: for each 4-tuple, store the 6 edge indices
def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    # Correct lexicographic index for upper triangle of 16x16 matrix
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

def _precompute_k4s():
    """
    Returns a list of tuples. Each tuple represents a K4 (set of 4 vertices)
    and the 6 edge indices corresponding to the edges of that K4.
    """
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    verts = (a, b, c, d)
                    edges = [
                        (a, b), (a, c), (a, d),
                        (b, c), (b, d),
                        (c, d)
                    ]
                    idxs = []
                    for u, v in edges:
                        idx = _edge_idx(u, v)
                        idxs.append(idx)
                    k4s.append((verts, tuple(idxs)))
    return k4s

# Global precomputation
_K4S = _precompute_k4s()
_NUM_K4S = len(_K4S)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 30s safety margin for 150s limit

    # Initial coloring: (i + j) % 17
    # Edge (i, j) has color (i + j) % 17.
    # This uses 17 colors.
    
    # We represent the coloring by assigning a color label to each edge.
    # To allow merging, we maintain a "base_color" for each edge, 
    # and a mapping from base_color -> final_color.
    
    # Base colors are in 0..16.
    base_colors = []
    for i in range(16):
        for j in range(i + 1, 16):
            base_colors.append((i + j) % 17)
    base_colors = tuple(base_colors)
    
    # Ensure base_colors length is 120
    assert len(base_colors) == 120

    # Initial mapping: identity
    # color_map[c] = final color for base color c
    color_map = list(range(17))
    
    # Current number of colors used
    current_colors = 17

    # Helper to check if a coloring is valid
    def is_valid(cmap):
        # cmap: list mapping base color -> final color
        # We check all K4s
        # For each K4, collect the set of final colors
        # If size < 5, invalid
        for k4 in _K4S:
            verts, idxs = k4
            colors_set = set()
            for idx in idxs:
                bc = base_colors[idx]
                fc = cmap[bc]
                colors_set.add(fc)
                # Early exit if size is already 5? No, we need to check if it's AT LEAST 5.
                # Wait, the condition is "at least five distinct colours".
                # So if we have 5, it's valid. We just need to ensure we don't have < 5.
                # But if we have 6, it's also valid.
                # So we just need to check if len(colors_set) >= 5 at the end?
                # Actually, if we find a K4 with < 5, it's invalid.
                # So we can early exit if we are done processing edges and len < 5.
                # Or we can just collect and check at the end.
                # Given the small size, collecting is fine.
            if len(colors_set) < 5:
                return False
        return True

    # Greedy Merging
    # We want to merge base colors to reduce the number of distinct final colors.
    # Merging base colors u and v means setting color_map[v] = color_map[u] (or vice versa, effectively making them same).
    # We iterate through pairs of base colors that are currently distinct.
    
    # To make this robust, we can randomize the order of merges or restart.
    # But greedy sequential merging is often effective.
    
    # Let's store the best solution found
    best_colors = current_colors
    best_map = color_map[:]

    # We will try to merge colors.
    # A merge is valid if the new coloring is valid.
    # We can use a simple loop:
    # while True:
    #   merged = False
    #   for u in range(17):
    #     if color_map[u] == u: # u is a "root"
    #       for v in range(17):
    #         if v != u and color_map[v] == v: # v is a distinct root
    #           # Try merging v into u
    #           old_v_map = color_map[v]
    #           color_map[v] = u
    #           # Check validity
    #           if is_valid(color_map):
    #             current_colors -= 1
    #             best_colors = current_colors
    #             best_map = color_map[:]
    #             merged = True
    #             break
    #           else:
    #             color_map[v] = old_v_map
    #         if merged:
    #             break
    #     if merged:
    #       continue
    #   if not merged:
    #     break
    
    # However, checking validity every single merge is O(1820 * 6).
    # 1820 * 6 ~ 10k operations.
    # Number of merges ~ 17.
    # Number of pairs to try ~ 17*17/2 ~ 150.
    # Total ops ~ 150 * 10k = 1.5M. Very fast.
    
    # Let's optimize is_valid slightly by precomputing which base colors appear in each K4.
    # For each K4, we have a set of base colors.
    # If we merge u and v, any K4 containing both u and v will lose a color.
    # K4s containing u but not v are unaffected (unless v was mapping to same, but they are distinct roots).
    # So we only need to check K4s that contain BOTH u and v.
    
    # Precompute for each pair (u, v) the list of K4s that contain both base color u and base color v.
    k4s_containing = {}
    for k4_idx, (verts, idxs) in enumerate(_K4S):
        bcs = set()
        for idx in idxs:
            bcs.add(base_colors[idx])
        bcs = tuple(sorted(bcs))
        # Add this K4 to all pairs in bcs
        for i in range(len(bcs)):
            for j in range(i + 1, len(bcs)):
                pair = (bcs[i], bcs[j])
                if pair not in k4s_containing:
                    k4s_containing[pair] = []
                k4s_containing[pair].append(k4_idx)

    def is_valid_fast(cmap, affected_k4s):
        for k4_idx in affected_k4s:
            verts, idxs = _K4S[k4_idx]
            colors_set = set()
            for idx in idxs:
                bc = base_colors[idx]
                fc = cmap[bc]
                colors_set.add(fc)
            if len(colors_set) < 5:
                return False
        return True

    # Greedy loop with time check
    while True:
        if time.time() > deadline:
            break
        
        merged = False
        # Randomize order of checks to escape local minima? 
        # Or just fixed order. Fixed order is deterministic.
        # Let's try fixed order first.
        
        for u in range(17):
            if color_map[u] != u:
                continue
            for v in range(u + 1, 17):
                if color_map[v] != v:
                    continue
                
                # Check if u and v appear together in any K4
                pair = (u, v)
                if pair in k4s_containing:
                    affected = k4s_containing[pair]
                else:
                    affected = [] # No K4 has both, so merging is always safe? 
                                  # Wait, if no K4 has both, then no K4 loses a color due to this merge.
                                  # So it is always valid.
                
                # Try merge v -> u
                color_map[v] = u
                valid = is_valid_fast(color_map, affected)
                
                if valid:
                    current_colors -= 1
                    best_colors = current_colors
                    best_map = color_map[:]
                    merged = True
                    break
                else:
                    color_map[v] = v
            if merged:
                break
        
        if not merged:
            break

    # If we have time, try random restarts or different merge orders?
    # The greedy might get stuck. 
    # We can try shuffling the order of v for each u.
    
    # Let's see how much time we have.
    # If we finished quickly, we can try more.
    
    # Actually, let's just return the best found so far.
    # But wait, the greedy above only does one pass. 
    # "While True" loop continues as long as merges happen.
    # So it will keep merging until stuck.
    
    # Is it possible to get a better result with randomization?
    # Yes. Let's add a few random restarts if time permits.
    
    # However, the deterministic greedy is likely sufficient for 17 -> 15.
    # Let's stick to the deterministic one for simplicity and speed, 
    # unless we are very fast.
    
    # Re-verify the best_map is valid (paranoia)
    # Actually, we updated best_map only when valid.
    
    # Construct the output
    result = []
    for bc in base_colors:
        result.append(best_map[bc])
    
    return result

# Test locally if __name__ == '__main__'
if __name__ == '__main__':
    res = construct()
    print(f"Colors used: {len(set(res))}")
    print(f"Max color: {max(res)}")
    # Quick validation
    for k4 in _K4S:
        verts, idxs = k4
        colors = set()
        for idx in idxs:
            colors.add(res[idx])
        if len(colors) < 5:
            print(f"Invalid K4: {verts} with colors {colors}")
            break
    else:
        print("All K4s valid.")
